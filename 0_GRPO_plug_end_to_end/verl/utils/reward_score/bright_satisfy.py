# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import time
import requests
import numpy as np
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import math
import pytrec_eval

# Default retrieval URL, kept consistent with bright_tool_config.yaml:
DEFAULT_BRIGHT_RETRIEVAL_URL = "http://172.16.34.22:8516/batch_retrieve"


def _normalize_question_id(task, sample_id):
    task_str = "" if task is None else str(task)
    sample_id_str = "" if sample_id is None else str(sample_id)
    prefix = f"{task_str}_"
    if task_str and sample_id_str.startswith(prefix):
        return sample_id_str[len(prefix):]
    return sample_id_str


def extract_steps(trajectory: str, max_steps: int = 5):
    """Extract per-step info blocks from a trajectory string.

    Each step is expected to be either:
    1) <think>...</think> + <summary>...</summary> + <information>...</information>
    2) <think>...</think> + <satisfy> yes </satisfy>
    Returns a list of dicts with keys: think, summary, information, satisfy.

    NOTE: The model may place <summary> *inside* a <think> block (e.g. the second
    turn often looks like <think>...<summary>...</summary>...</think>).  The old
    single-regex approach missed those because the <think> non-greedy match would
    consume the inner <summary>.  We now extract all <summary> and <satisfy> tags
    independently so nothing is lost.
    """

    # Collect all <summary>...</summary> spans
    summaries = [
        (m.start(), m.group(1).strip())
        for m in re.finditer(r"<summary>(.*?)</summary>", trajectory, re.DOTALL)
    ]

    # Collect all <satisfy>...</satisfy> spans
    satisfies = [
        (m.start(), m.group(1).strip())
        for m in re.finditer(r"<satisfy>(.*?)</satisfy>", trajectory, re.DOTALL)
    ]

    # Merge into ordered steps (by position in text)
    events = [(pos, "summary", text) for pos, text in summaries] + \
             [(pos, "satisfy", text) for pos, text in satisfies]
    events.sort(key=lambda x: x[0])

    steps = []
    for pos, kind, text in events:
        step = {"think": "", "summary": None, "information": None, "satisfy": None}
        if kind == "summary":
            step["summary"] = text
        else:
            step["satisfy"] = text
        steps.append(step)
        if len(steps) >= max_steps:
            break
    return steps


def extract_expand_query(sequences_str, input_docs, query):
    """Extract the expanded query from the model trajectory.
    
    Uses the last summary as expansion, falls back to initial docs.
    """
    trajectory = sequences_str or ""
    steps = extract_steps(trajectory)

    # Search backwards for the last step that has a valid summary
    for step in reversed(steps):
        if step["summary"] is not None and len(step["summary"]) > 20:
            return query + " " + step["summary"]

    return query + " " + input_docs


def number_duplicate_qids(q_ids):
    counter = defaultdict(int)
    new_qids = []

    for qid in q_ids:
        idx = counter[qid]
        new_qids.append(f"{qid}_{idx}")
        counter[qid] += 1

    return new_qids


def _batch_search(queries, search_host_url, question_ids=None, tasks=None, excluded_ids=None):
    """Batchified search for queries — PARALLELIZED per task via ThreadPoolExecutor."""
    final_scores = {}
    if excluded_ids is None:
        excluded_ids = [["N/A"] for _ in queries]

    task2datas = defaultdict(lambda: {
        "q_ids": [],
        "questions": [],
        "excluded_ids": []
    })

    for qid, query, task, excluded in zip(question_ids, queries, tasks, excluded_ids):
        task2datas[task]["q_ids"].append(qid)
        task2datas[task]["questions"].append(query)
        task2datas[task]["excluded_ids"].append(excluded)

    def _search_one_task(task, id_query):
        ids = number_duplicate_qids(id_query["q_ids"])
        raw_excluded_ids = id_query["excluded_ids"]
        normalized_excluded_ids = []
        for ex_ids in raw_excluded_ids:
            if ex_ids is None:
                normalized_excluded_ids.append(["N/A"])
            elif not isinstance(ex_ids, (list, tuple, set)):
                normalized_excluded_ids.append(["N/A"])
            elif len(ex_ids) == 0:
                normalized_excluded_ids.append(["N/A"])
            else:
                normalized_excluded_ids.append(ex_ids)

        if len(normalized_excluded_ids) < len(ids):
            normalized_excluded_ids.extend([["N/A"] for _ in range(len(ids) - len(normalized_excluded_ids))])

        path_excluded_ids = {qid: ex_ids for qid, ex_ids in zip(ids, normalized_excluded_ids)}
       
        payload = {
            "task": task,
            "q_id_list": ids,
            "q_text_list": id_query["questions"],
            "excluded_ids": path_excluded_ids,
            "num_hits": 20,
        }
        max_retries = 5
        for attempt in range(max_retries):
            try:
                resp = requests.post(search_host_url, json=payload)
                resp.raise_for_status()
                return task, resp.json()
            except Exception as e:
                wait_time = 2 ** attempt
                print(f"[_batch_search] task={task}, attempt {attempt+1}/{max_retries} failed: {e}. "
                      f"Retrying in {wait_time}s...")
                time.sleep(wait_time)
        print(f"[_batch_search] WARNING: All {max_retries} retries failed for task={task}")
        return task, None



    with ThreadPoolExecutor(max_workers=min(12, len(task2datas))) as executor:
        futures = {
            executor.submit(_search_one_task, task, id_query): task
            for task, id_query in task2datas.items()
        }
        for future in as_completed(futures):
            task, data = future.result()
            if data is None:
                continue
            id_doc_scores = data["scores"]
            for inst_id, (qid, docs_score) in enumerate(id_doc_scores.items()):
                final_scores[(task, qid, inst_id)] = docs_score

    return final_scores


def dcg_at_k(relevances, k=10):
    """relevances: list of relevance scores in ranked order"""
    dcg = 0.0
    for i in range(min(k, len(relevances))):
        rel = relevances[i]
        dcg += (2**rel - 1) / math.log2(i + 2)  # i starts from 0, so rank = i+1
    return dcg


def ndcg_at_k(retrieved_scores: dict, ground_truth_ids: list, k=10):
    """
    retrieved_scores: dict {doc_id: score} (higher is better)
    ground_truth_ids: list of relevant doc_ids
    k: cutoff (default 10)

    Uses binary relevance: rel=1 if doc_id in ground_truth_ids else 0
    """
    ranked_docs = sorted(retrieved_scores.items(), key=lambda x: x[1], reverse=True)
    ranked_doc_ids = [doc_id for doc_id, _ in ranked_docs[:k]]

    relevances = [1 if doc_id in ground_truth_ids else 0 for doc_id in ranked_doc_ids]

    dcg = dcg_at_k(relevances, k)

    ideal_relevances = [1] * min(len(ground_truth_ids), k)
    idcg = dcg_at_k(ideal_relevances, k)

    if idcg == 0:
        return 0.0

    return dcg / idcg


def calculate_retrieval_metrics(results, qrels, k_values=[10]):
    """Calculate retrieval metrics with pytrec_eval (BEIR-style)."""
    map_string = "map_cut." + ",".join([str(k) for k in k_values])
    ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
    recall_string = "recall." + ",".join([str(k) for k in k_values])
    precision_string = "P." + ",".join([str(k) for k in k_values])

    evaluator = pytrec_eval.RelevanceEvaluator(
        qrels,
        {map_string, ndcg_string, recall_string, precision_string, "recip_rank"},
    )
    return evaluator.evaluate(results)


def compute_score_batch(solution_strs, ground_truths, extra_infos_list, **kwargs):
    """Batch version of compute_score. Makes ONE _batch_search call for ALL samples.

    Instead of calling _batch_search N times with 1 query each, this collects
    all N queries and sends them in a single call (one HTTP POST per unique task).

    Args:
        solution_strs: list of decoded model response strings
        ground_truths: list of ground truths (unused for BRIGHT, kept for API compat)
        extra_infos_list: list of extra_info dicts

    Returns:
        list of float scores (NDCG@10), one per sample
    """
    n_solution = len(solution_strs)
    n_extra = len(extra_infos_list)
    if n_solution != n_extra:
        print(
            f"[compute_score_batch] Warning: length mismatch solution_strs={n_solution}, extra_infos_list={n_extra}; "
            f"using first {min(n_solution, n_extra)} samples"
        )

    n = min(n_solution, n_extra)
    scores = [0.0] * n

    # Step 1: Prepare expanded queries for ALL samples
    expanded_queries = []
    question_ids = []
    tasks_list = []
    ground_truth_ids_list = []
    valid_indices = []  # maps position in parallel arrays -> original sample index
    excluded_ids_list = []

    for i in range(n):
        extra_info = extra_infos_list[i]
        if extra_info is None:
            continue
        try:
            input_docs = extra_info["initial_doc"]
            query = extra_info["question"]
            task = extra_info["task"]
            sample_id = extra_info["id"]
            excluded_ids= extra_info.get("excluded_ids") # for excample , ["N/A"]
            ground_truth = extra_info["interaction_kwargs"]["ground_truth"]

            sequences_str = extract_expand_query(solution_strs[i], input_docs, query)
            question_id = _normalize_question_id(task, sample_id)

            # Normalize ground-truth doc ids to a flat list of strings.
            if isinstance(ground_truth, np.ndarray):
                ground_truth = ground_truth.tolist()
            elif isinstance(ground_truth, (tuple, set)):
                ground_truth = list(ground_truth)
            elif not isinstance(ground_truth, list):
                ground_truth = [ground_truth]

            expanded_queries.append(sequences_str)
            question_ids.append(question_id)
            tasks_list.append(task)
            ground_truth_ids_list.append(ground_truth)
            valid_indices.append(i)
            excluded_ids_list.append(excluded_ids)
        except Exception as e:
            print(f"[compute_score_batch] Failed to prepare sample {i}: {e}")
            continue

    if not expanded_queries:
        return scores

    # Step 2: ONE _batch_search call for ALL queries
    retrieval_url = os.getenv("BRIGHT_RETRIEVAL_URL", DEFAULT_BRIGHT_RETRIEVAL_URL)
    print(f"[compute_score_batch] Batch searching {len(expanded_queries)} queries -> {len(set(tasks_list))} tasks")

    final_all_scores = _batch_search(
        expanded_queries,
        search_host_url=retrieval_url,
        question_ids=question_ids,
        tasks=tasks_list,
        excluded_ids=excluded_ids_list
    )

    # Step 3: Build qrels/results (merge_scores.py style), evaluate with pytrec_eval,
    # then map ndcg_cut_10 back to each sample.
    task_to_positions = defaultdict(list)
    for pos, task in enumerate(tasks_list):
        task_to_positions[task].append(pos)

    per_task_results = defaultdict(dict)
    per_task_qrels = defaultdict(dict)
    per_task_eval_qid_to_pos = defaultdict(dict)

    # Rebuild duplicated qids exactly as _batch_search does (per-task numbering),
    # so we can map returned qids back to original sample positions robustly.
    task_qid_counter = defaultdict(lambda: defaultdict(int))
    task_dup_qid_to_pos = defaultdict(dict)
    for pos, (task, qid) in enumerate(zip(tasks_list, question_ids)):
        dup_idx = task_qid_counter[task][qid]
        task_qid_counter[task][qid] += 1
        dup_qid = f"{qid}_{dup_idx}"
        task_dup_qid_to_pos[task][dup_qid] = pos

    for (task, returned_qid, inst_id), did_scores in final_all_scores.items():
        pos = task_dup_qid_to_pos[task].get(returned_qid)
        # Fallback to inst_id mapping if qid is unexpectedly missing.
        if pos is None:
            if inst_id >= len(task_to_positions[task]):
                continue
            pos = task_to_positions[task][inst_id]
        gt_ids = ground_truth_ids_list[pos]

        # Use a unique eval qid so repeated prompts in the same batch do not collide.
        eval_qid = f"{question_ids[pos]}__idx{pos}"

        per_task_results[task][eval_qid] = did_scores
        per_task_qrels[task][eval_qid] = {str(gid): 1 for gid in gt_ids}
        per_task_eval_qid_to_pos[task][eval_qid] = pos

    for task in per_task_results.keys():
        if not per_task_results[task]:
            continue

        task_metrics = calculate_retrieval_metrics(
            results=per_task_results[task],
            qrels=per_task_qrels[task],
            k_values=[10],
        )

        for eval_qid, metric in task_metrics.items():
            pos = per_task_eval_qid_to_pos[task].get(eval_qid)
            if pos is None:
                continue
            orig_idx = valid_indices[pos]
            ndcg = float(metric.get("ndcg_cut_10", 0.0))
            print("NDCG10", ndcg)
            scores[orig_idx] = ndcg

    return scores


# def compute_score(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0, extra_info=None, **kwargs):
#     """Scoring function for BRIGHT: re-retrieve with expanded query and compute NDCG@10.

#     Args:
#         solution_str: the solution text (model's trajectory)
#         ground_truth: the ground truth
#         extra_info: dict with keys: initial_doc, question, task, id, interaction_kwargs.ground_truth
#     """
#     if extra_info is None:
#         return 0

#     input_docs = extra_info["initial_doc"]
#     query = extra_info["question"]

#     # Use model's trajectory plus retrieved docs/query to build expanded query
#     sequences_str = extract_expand_query(solution_str, input_docs, query)

#     ground_truth_doc_ids = [extra_info["interaction_kwargs"]["ground_truth"]]

#     task = extra_info["task"]
#     question_id = _normalize_question_id(task, extra_info["id"])

#     retrieval_url = os.getenv("BRIGHT_RETRIEVAL_URL", DEFAULT_BRIGHT_RETRIEVAL_URL)

#     single_excluded_ids = extra_info.get("excluded_ids")
#     if single_excluded_ids is None:
#         single_excluded_ids = [["N/A"]]
#     else:
#         single_excluded_ids = [single_excluded_ids]

#     final_all_scores = _batch_search(
#         [sequences_str],
#         search_host_url=retrieval_url,
#         question_ids=[question_id],
#         tasks=[task],
#         excluded_ids=single_excluded_ids
#     )

#     scores = []
#     flag = -1
#     for key, value in final_all_scores.items():
#         flag = flag + 1
#         ground_id = ground_truth_doc_ids[flag]
#         retrieved_scores = value
#         NDCG_10 = ndcg_at_k(retrieved_scores, ground_id, k=10)
#         scores.append(NDCG_10)

#     return scores[0] if scores else 0.0
