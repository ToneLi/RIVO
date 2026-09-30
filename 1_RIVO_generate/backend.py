import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import glob, random, json, pickle
from abc import ABC, abstractmethod
from threading import Lock
from typing import Dict, List
import duckdb
import faiss
import numpy as np
import torch
from loguru import logger
from pydantic import BaseModel, Field
from pyserini.search.lucene import LuceneSearcher
from tevatron.retriever.arguments import ModelArguments
from tevatron.retriever.driver.encode import DenseModel
from tevatron.retriever.searcher import FaissFlatSearcher
from tqdm import tqdm
from transformers import AutoTokenizer

faiss.omp_set_num_threads(int(os.getenv("FAISS_NUM_THREADS", "1")))
torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS", "1")))


class Corpus:
    def __init__(self, parquet_path: str):
        self.parquet_path = parquet_path
        self.id2url: Dict[str, str] = {}
        self.url2id: Dict[str, str] = {}
        self.docid_to_text: Dict[str, str] = {}
        self.load()

    def load(self):
        logger.info(f"Loading document mappings from {self.parquet_path}...")
        try:
            con = duckdb.connect(database=':memory:', read_only=False)
            total_docs = con.execute(
                f"SELECT count(*) FROM read_parquet('{self.parquet_path}')"
            ).fetchone()[0]
            result_iterator = con.execute(
                f"SELECT docid, url, text FROM read_parquet('{self.parquet_path}')"
            )

            self.id2url = {}
            self.url2id = {}
            self.docid_to_text = {}

            batch_size = int(os.getenv("CORPUS_LOAD_BATCH_SIZE", "100000"))
            with tqdm(total=total_docs, desc="Loading corpus docs", unit="doc") as pbar:
                while True:
                    rows = result_iterator.fetchmany(batch_size)
                    if not rows:
                        break
                    for docid, url, text in rows:
                        docid = str(docid)
                        self.id2url[docid] = url
                        self.url2id[url] = docid
                        self.docid_to_text[docid] = text
                    pbar.update(len(rows))

            logger.info(f"Document mappings loaded successfully. Found {len(self.id2url)} mappings.")
        except Exception as e:
            logger.error(f"Failed to load document mappings: {e}")
            raise
    
    def get_url_from_id(self, docid: str) -> str | None:
        return self.id2url.get(str(docid))

    def get_id_from_url(self, url: str) -> str | None:
        return self.url2id.get(url)

    def get_text_from_id(self, docid: str) -> str | None:
        docid = str(docid)
        return self.docid_to_text.get(docid)

class SearchResult(BaseModel):
    docid: str
    score: float
    text: str | None = Field(default=None)

class BaseSearcher(ABC):
    @abstractmethod
    def search(self, query: str, topn: int) -> List[SearchResult]:
        pass

    def search_batch(self, queries: List[str], topn: int) -> List[List[SearchResult]]:
        return [self.search(query, topn) for query in queries]

class BM25Searcher(BaseSearcher):
    def __init__(self, index_path: str, corpus: Corpus):
        logger.info(f"Initializing BM25Searcher with index: {index_path}")
        self.searcher = LuceneSearcher(index_path)
        self.corpus = corpus

    def search(self, query: str, topn: int) -> List[SearchResult]:
        hits = self.searcher.search(query, k=topn)
        return [
            SearchResult(
                docid=str(hit.docid),
                score=hit.score,
                text=self.corpus.get_text_from_id(str(hit.docid))
            )
            for hit in hits
        ]

class DenseSearcher(BaseSearcher):
    def __init__(
        self,
        index_path: str,
        model_name: str,
        corpus: Corpus,
        normalize: bool = True,
        pooling: str = 'eos',
        torch_dtype: str = 'float16',
        task_prefix: str= "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:",
        max_length: int = 8192,
        gpu_ids = ["0"],
    ):
        self.index_path_pattern = index_path
        self.model_name = model_name
        self.corpus = corpus
        self.normalize = normalize
        self.pooling = pooling
        self.torch_dtype_str = torch_dtype
        self.task_prefix = task_prefix
        self.max_length = max_length
        self.gpu_ids = gpu_ids
        
        self.retriever = None
        self.model = None
        self.tokenizer = None
        self.lookup = []
        self.search_lock = Lock()

        logger.info("Initializing DenseSearcher...")
        self._load_faiss_index()
        self._load_model()
        logger.info("DenseSearcher initialized successfully.")

    def _load_faiss_index(self) -> None:
        class TqdmFileReader:
            def __init__(self, file_obj, progress):
                self.file_obj = file_obj
                self.progress = progress

            def read(self, size=-1):
                data = self.file_obj.read(size)
                self.progress.update(len(data))
                return data

            def readline(self, size=-1):
                data = self.file_obj.readline(size)
                self.progress.update(len(data))
                return data

            def __getattr__(self, name):
                return getattr(self.file_obj, name)

        def pickle_load(path):
            file_size = os.path.getsize(path)
            logger.info(f"Loading dense index pickle: {path} ({file_size / (1024 ** 3):.2f} GiB)")
            with open(path, 'rb') as f:
                with tqdm(total=file_size, desc=f"Reading {os.path.basename(path)}", unit="B", unit_scale=True, unit_divisor=1024) as pbar:
                    reps, lookup = pickle.load(TqdmFileReader(f, pbar))
            logger.info(f"Converting reps from {path} to float32 numpy array...")
            return np.array(reps, dtype=np.float32), lookup

        cache_dir = os.getenv("DENSE_FAISS_CACHE_DIR")
        if cache_dir:
            cache_index_path = os.path.join(cache_dir, "index.faiss")
            cache_lookup_path = os.path.join(cache_dir, "lookup.pkl")
            if os.path.exists(cache_index_path) and os.path.exists(cache_lookup_path):
                logger.info(f"Loading FAISS index cache from {cache_index_path}...")
                self.retriever = type("CachedFaissSearcher", (), {})()
                self.retriever.index = faiss.read_index(cache_index_path)
                file_size = os.path.getsize(cache_lookup_path)
                with open(cache_lookup_path, "rb") as f:
                    with tqdm(total=file_size, desc="Reading lookup.pkl", unit="B", unit_scale=True, unit_divisor=1024) as pbar:
                        self.lookup = pickle.load(TqdmFileReader(f, pbar))
                self.retriever.search = self.retriever.index.search
                logger.info(f"FAISS cache ready. ntotal={self.retriever.index.ntotal}, lookup_len={len(self.lookup)}")
                return

        index_files = glob.glob(self.index_path_pattern)
        if not index_files:
            raise ValueError(f"No files found matching pattern: {self.index_path_pattern}")

        logger.info(f'Found {len(index_files)} shards; loading into index.')
        p_reps_0, p_lookup_0 = pickle_load(index_files[0])

        # Create CPU index and add incrementally
        logger.info(f"Creating FAISS index with first shard shape={p_reps_0.shape}...")
        self.retriever = FaissFlatSearcher(p_reps_0)
        logger.info("Adding first shard to FAISS index...")
        self.retriever.add(p_reps_0)
        self.lookup.extend(p_lookup_0)

        if len(index_files) > 1:
            for f in tqdm(index_files[1:], desc='Loading shards'):
                p_reps, p_lookup = pickle_load(f)
                self.retriever.add(p_reps)
                self.lookup.extend(p_lookup)

        ntotal = self.retriever.index.ntotal
        if ntotal != len(self.lookup):
            raise RuntimeError(
                f"FAISS index/lookup mismatch: ntotal={ntotal}, lookup_len={len(self.lookup)}. "
                "Check shard loading or lookup building."
            )
        logger.info(f"FAISS index ready. ntotal={ntotal}")

        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            cache_index_path = os.path.join(cache_dir, "index.faiss")
            cache_lookup_path = os.path.join(cache_dir, "lookup.pkl")
            logger.info(f"Writing FAISS index cache to {cache_index_path}...")
            faiss.write_index(self.retriever.index, cache_index_path)
            logger.info(f"Writing lookup cache to {cache_lookup_path}...")
            with open(cache_lookup_path, "wb") as f:
                pickle.dump(self.lookup, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info("FAISS cache written. Next startup can skip corpus.pkl pickle loading.")

    def _load_model(self) -> None:
        logger.info(f"Loading model: {self.model_name}")
        model_args = ModelArguments(
            model_name_or_path=self.model_name,
            normalize=self.normalize,
            pooling=self.pooling,
        )
        dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16}
        torch_dtype = dtype_map.get(self.torch_dtype_str, torch.float32)

        self.model_pool = [DenseModel.load(
            model_args.model_name_or_path, pooling=model_args.pooling,
            normalize=model_args.normalize, torch_dtype=torch_dtype
        ).to(f'cuda:{gpu_idx}').eval() for gpu_idx in self.gpu_ids]
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, padding_side='left'
        )
        logger.info("Model loaded successfully.")

    def search(self, query: str, topn: int) -> List[SearchResult]:
        return self.search_batch([query], topn)[0]

    def search_batch(self, queries: List[str], topn: int) -> List[List[SearchResult]]:
        if not all([self.retriever, self.model_pool, self.tokenizer]):
            raise RuntimeError("DenseSearcher not properly initialized.")

        # Keep dense retrieval single-flight inside one server process. The full
        # OpenResearcher FAISS index is huge, and concurrent BLAS/FAISS calls can
        # exhaust thread-local memory regions on this machine. Batching improves
        # throughput without concurrent FAISS/BLAS calls.
        with self.search_lock:
            encoded_query = self.tokenizer(
                [self.task_prefix + query for query in queries],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            model = random.choice(self.model_pool)
            device = next(model.parameters()).device
            encoded_query = {k: v.to(device) for k, v in encoded_query.items()}
            
            with torch.autocast(
                device_type='cuda', dtype={"float16": torch.float16, "bfloat16": torch.bfloat16}[self.torch_dtype_str]
            ):
                with torch.no_grad():
                    q_reps = model.encode_query(encoded_query).cpu().numpy()

            scores, indices = self.retriever.search(q_reps.astype(np.float32, copy=False), topn)

        batched_results: List[List[SearchResult]] = []
        for row_scores, row_indices in zip(scores, indices):
            results = []
            for score, idx in zip(row_scores, row_indices):
                if idx < 0:
                    continue
                docid = str(self.lookup[idx])
                results.append(
                    SearchResult(
                        docid=docid,
                        score=float(score),
                        text=self.corpus.get_text_from_id(docid),
                    )
                )
            batched_results.append(results)
        return batched_results

def get_searcher(corpus: Corpus) -> BaseSearcher:
    """
    Reads environment variables to determine which searcher to instantiate
    and returns the configured searcher instance.

    Required Environment Variables:
    - SEARCHER_TYPE: "bm25" or "dense"

    For BM25:
    - LUCENE_INDEX_DIR: Path to the Lucene index directory.

    For Dense:
    - DENSE_INDEX_PATH: Glob pattern for FAISS index files (*.pkl).
    - DENSE_MODEL_NAME: HuggingFace model name for the encoder.
    - GPU_IDS (optional): Comma-separated list of GPU IDs to use (e.g., "0,1").
    """
    searcher_type = os.getenv("SEARCHER_TYPE", "bm25").lower()
    logger.info(f"Attempting to initialize searcher of type: '{searcher_type}'")

    if searcher_type == "bm25":
        index_dir = os.getenv("LUCENE_INDEX_DIR")
        if not index_dir:
            raise ValueError("LUCENE_INDEX_DIR environment variable must be set for BM25 searcher.")
        return BM25Searcher(index_path=index_dir, corpus=corpus)

    elif searcher_type == "dense":
        index_path = os.getenv("DENSE_INDEX_PATH")
        model_name = os.getenv("DENSE_MODEL_NAME")

        if not index_path or not model_name:
            raise ValueError("DENSE_INDEX_PATH and DENSE_MODEL_NAME must be set for dense searcher.")

        gpu_ids_str = os.getenv("GPU_IDS", "0")
        gpu_ids = [gid.strip() for gid in gpu_ids_str.split(',')]

        return DenseSearcher(
            index_path=index_path,
            model_name=model_name,
            corpus=corpus,
            gpu_ids=gpu_ids,
        )
    else:
        raise ValueError(f"Unknown SEARCHER_TYPE: '{searcher_type}'. Please choose 'bm25' or 'dense'.")