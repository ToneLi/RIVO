# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
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

import json
import logging
import os
import threading
import time
import traceback
import uuid
from typing import Any, Optional
from collections import defaultdict

import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

DEFAULT_TIMEOUT = 30  # Default search request timeout
MAX_RETRIES = 10
INITIAL_RETRY_DELAY = 1
API_TIMEOUT = 10

logger = logging.getLogger(__name__)


def call_search_api(
    retrieval_service_url: str,
    query_list: list[str],
    topk: int = 3,
    return_scores: bool = True,
    timeout: int = DEFAULT_TIMEOUT,
) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """
    Calls the remote search API to perform retrieval with retry logic for various errors,
    using increasing delay between retries. Logs internal calls with a unique ID.

    Args:
        retrieval_service_url: The URL of the retrieval service API.
        query_list: List of search queries.
        topk: Number of top results to return.
        return_scores: Whether to return scores.
        timeout: Request timeout in seconds.

    Returns:
        A tuple (response_json, error_message).
        If successful, response_json is the API's returned JSON object, error_message is None.
        If failed after retries, response_json is None, error_message contains the error information.
    """
    request_id = str(uuid.uuid4())
    log_prefix = f"[Search Request ID: {request_id}] "

    payload = {"queries": query_list, "topk": topk, "return_scores": return_scores}

    headers = {"Content-Type": "application/json", "Accept": "application/json"}

    last_error = None

    for attempt in range(MAX_RETRIES):
        try:
            logger.info(
                f"{log_prefix}Attempt {attempt + 1}/{MAX_RETRIES}: Calling search API at {retrieval_service_url}"
            )
            response = requests.post(
                retrieval_service_url,
                headers=headers,
                json=payload,
                timeout=timeout,
            )

            # Check for Gateway Timeout (504) and other server errors for retrying
            if response.status_code in [500, 502, 503, 504]:
                last_error = (
                    f"{log_prefix}API Request Error: Server Error ({response.status_code}) on attempt "
                    f"{attempt + 1}/{MAX_RETRIES}"
                )
                logger.warning(last_error)
                if attempt < MAX_RETRIES - 1:
                    delay = INITIAL_RETRY_DELAY * (attempt + 1)
                    logger.info(f"{log_prefix}Retrying after {delay} seconds...")
                    time.sleep(delay)
                continue

            # Check for other HTTP errors (e.g., 4xx)
            response.raise_for_status()

            # If successful (status code 2xx)
            logger.info(f"{log_prefix}Search API call successful on attempt {attempt + 1}")
            return response.json(), None

        except requests.exceptions.ConnectionError as e:
            last_error = f"{log_prefix}Connection Error: {e}"
            logger.warning(last_error)
            if attempt < MAX_RETRIES - 1:
                delay = INITIAL_RETRY_DELAY * (attempt + 1)
                logger.info(f"{log_prefix}Retrying after {delay} seconds...")
                time.sleep(delay)
            continue
        except requests.exceptions.Timeout as e:
            last_error = f"{log_prefix}Timeout Error: {e}"
            logger.warning(last_error)
            if attempt < MAX_RETRIES - 1:
                delay = INITIAL_RETRY_DELAY * (attempt + 1)
                logger.info(f"{log_prefix}Retrying after {delay} seconds...")
                time.sleep(delay)
            continue
        except requests.exceptions.RequestException as e:
            last_error = f"{log_prefix}API Request Error: {e}"
            break  # Exit retry loop on other request errors
        except json.JSONDecodeError as e:
            raw_response_text = response.text if "response" in locals() else "N/A"
            last_error = f"{log_prefix}API Response JSON Decode Error: {e}, Response: {raw_response_text[:200]}"
            break  # Exit retry loop on JSON decode errors
        except Exception as e:
            last_error = f"{log_prefix}Unexpected Error: {e}"
            break  # Exit retry loop on other unexpected errors

    # If loop finishes without returning success, return the last recorded error
    logger.error(f"{log_prefix}Search API call failed. Last error: {last_error}")
    return None, last_error.replace(log_prefix, "API Call Failed: ") if last_error else "API Call Failed after retries"


def _passages2string(retrieval_result):
    """Convert retrieval results to formatted string."""
    format_reference = ""
    for idx, doc_item in enumerate(retrieval_result):
        content = doc_item["document"]["contents"]
        title = content.split("\n")[0]
        text = "\n".join(content.split("\n")[1:])
        format_reference += f"Doc {idx + 1} (Title: {title})\n{text}\n\n"
    return format_reference.strip()


def perform_single_search_batch(
    retrieval_service_url: str,
    query_list: list[str],
    topk: int = 3,
    concurrent_semaphore: Optional[threading.Semaphore] = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> tuple[str, dict[str, Any]]:
    """
    Performs a single batch search for multiple queries (original search tool behavior).

    Args:
        retrieval_service_url: The URL of the retrieval service API.
        query_list: List of search queries.
        topk: Number of top results to return.
        concurrent_semaphore: Optional semaphore for concurrency control.
        timeout: Request timeout in seconds.

    Returns:
        A tuple (result_text, metadata).
        result_text: The search result JSON string.
        metadata: Metadata dictionary for the batch search.
    """
    logger.info(f"Starting batch search for {len(query_list)} queries.")

    api_response = None
    error_msg = None

    try:
        if concurrent_semaphore:
            with concurrent_semaphore:
                api_response, error_msg = call_search_api(
                    retrieval_service_url=retrieval_service_url,
                    query_list=query_list,
                    topk=topk,
                    return_scores=True,
                    timeout=timeout,
                )
        else:
            api_response, error_msg = call_search_api(
                retrieval_service_url=retrieval_service_url,
                query_list=query_list,
                topk=topk,
                return_scores=True,
                timeout=timeout,
            )
    except Exception as e:
        error_msg = f"API Request Exception during batch search: {e}"
        logger.error(f"Batch search: {error_msg}")
        traceback.print_exc()

    metadata = {
        "query_count": len(query_list),
        "queries": query_list,
        "api_request_error": error_msg,
        "api_response": None,
        "status": "unknown",
        "total_results": 0,
        "formatted_result": None,
    }

    result_text = json.dumps({"result": "Search request failed or timed out after retries."}, ensure_ascii=False)

    if error_msg:
        metadata["status"] = "api_error"
        result_text = json.dumps({"result": f"Search error: {error_msg}"}, ensure_ascii=False)
        logger.error(f"Batch search: API error occurred: {error_msg}")
    elif api_response:
        logger.debug(f"Batch search: API Response: {api_response}")
        metadata["api_response"] = api_response

        try:
            raw_results = api_response.get("result", [])
            if raw_results:
                pretty_results = []
                total_results = 0

                for retrieval in raw_results:
                    formatted = _passages2string(retrieval)
                    pretty_results.append(formatted)
                    total_results += len(retrieval) if isinstance(retrieval, list) else 1

                final_result = "\n---\n".join(pretty_results)
                result_text = json.dumps({"result": final_result}, ensure_ascii=False)
                metadata["status"] = "success"
                metadata["total_results"] = total_results
                metadata["formatted_result"] = final_result
                logger.info(f"Batch search: Successful, got {total_results} total results")
            else:
                result_text = json.dumps({"result": "No search results found."}, ensure_ascii=False)
                metadata["status"] = "no_results"
                metadata["total_results"] = 0
                logger.info("Batch search: No results found")
        except Exception as e:
            error_msg = f"Error processing search results: {e}"
            result_text = json.dumps({"result": error_msg}, ensure_ascii=False)
            metadata["status"] = "processing_error"
            logger.error(f"Batch search: {error_msg}")
    else:
        metadata["status"] = "unknown_api_state"
        result_text = json.dumps(
            {"result": "Unknown API state (no response and no error message)."}, ensure_ascii=False
        )
        logger.error("Batch search: Unknown API state.")

    return result_text, metadata


# ---------------------------------------------------------------------------
# BRIGHT-specific batch search utilities
# ---------------------------------------------------------------------------

def _get_env_list(var_name: str, default_value: list[str]) -> list[str]:
    """Return list from env var; accept JSON array or comma-separated string."""
    raw_value = os.getenv(var_name)
    if not raw_value:
        return default_value
    try:
        parsed = json.loads(raw_value)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    except json.JSONDecodeError:
        pass
    return [item.strip() for item in raw_value.split(",") if item.strip()]


DEFAULT_TASKS = [
    "biology",
    "earth_science",
    "economics",
    "psychology",
    "robotics",
    "stackoverflow",
    "sustainable_living",
    "leetcode",
    "pony",
    "aops",
    "theoremqa_theorems",
    "theoremqa_questions",
]
DEFAULT_CACHE_DIR = os.path.join(
    os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")),
    "diver-retriever",
)
DEFAULT_BASE_DOC_DIR = os.environ.get("BRIGHT_BASE_DOC_DIR", "Diver/Retriever/data/BRIGHT/document")


# Module-level document cache: (task, doc_id) -> content
did2content: dict[tuple[str, str], str] = {}
_did2content_loaded = False


def _load_task_documents(tasks=None, base_doc_dir=None, cache_dir=None):
    """Load documents for multiple tasks and build did2content mapping."""
    global _did2content_loaded
    if _did2content_loaded:
        return

    if tasks is None:
        tasks = _get_env_list("BRIGHT_TASKS", DEFAULT_TASKS)
    if base_doc_dir is None:
        base_doc_dir = os.getenv("BRIGHT_BASE_DOC_DIR", DEFAULT_BASE_DOC_DIR)
    if cache_dir is None:
        cache_dir = os.getenv("BRIGHT_CACHE_DIR", DEFAULT_CACHE_DIR)

    from datasets import load_dataset

    for task in tasks:
        # Support both BEIR naming ({task}-documents.parquet) and
        # BRIGHT naming ({task}-00000-of-00001.parquet).
        doc_path = os.path.join(base_doc_dir, f"{task}-documents.parquet")
        if not os.path.exists(doc_path):
            doc_path = os.path.join(base_doc_dir, f"{task}-00000-of-00001.parquet")
        if not os.path.exists(doc_path):
            logger.warning(f"Document file not found for task '{task}' in {base_doc_dir}")
            continue
        doc_pairs = load_dataset("parquet", data_files=doc_path, cache_dir=cache_dir)["train"]
        for dp in doc_pairs:
            doc_id = dp["id"]
            content = dp["content"]
            did2content[(task, doc_id)] = content

    _did2content_loaded = True
    logger.info(f"[BRIGHT] Loaded {len(did2content)} documents across {len(tasks)} tasks")


def _ensure_did2content():
    """Ensure did2content is loaded (lazy initialization)."""
    if not _did2content_loaded:
        _load_task_documents()


def number_duplicate_qids(q_ids):
    counter = defaultdict(int)
    new_qids = []
    for qid in q_ids:
        idx = counter[qid]
        new_qids.append(f"{qid}_{idx}")
        counter[qid] += 1
    return new_qids


def _batch_search(queries, question_ids=None, tasks=None, search_url=None, excluded_ids=None):
    """Batchified search for queries — PARALLELIZED per task via ThreadPoolExecutor."""
    final_scores = {}
    if excluded_ids is None:
        excluded_ids = [["N/A"] for _ in queries]

    if question_ids is None or tasks is None:
        raise ValueError("question_ids and tasks must be provided for _batch_search")

    if not (
        len(queries) == len(question_ids) == len(tasks) == len(excluded_ids)
    ):
        raise ValueError(
            "_batch_search expects one-to-one mapping among queries/question_ids/tasks/excluded_ids, "
            f"but got len(queries)={len(queries)}, len(question_ids)={len(question_ids)}, "
            f"len(tasks)={len(tasks)}, len(excluded_ids)={len(excluded_ids)}"
        )

    task2datas = defaultdict(lambda: {
        "q_ids": [],
        "questions": [],
        "excluded_ids": [],
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
                normalized_excluded_ids.append(list(ex_ids))

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
                resp = requests.post(search_url, json=payload)
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


def perform_single_search_batch_4b(
    retrieval_service_url: str,
    query_list: list[str],
    question_ids,
    tasks,
    excluded_ids=None,
    topk: int = 3,
    concurrent_semaphore: Optional[threading.Semaphore] = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> tuple[str, dict[str, Any]]:
    """BRIGHT-specific batch search: calls _batch_search and returns formatted documents.

    Args:
        retrieval_service_url: URL of the retrieval service.
        query_list: List of search queries.
        question_ids: Question ID(s).
        tasks: Task name(s).
        topk: Number of top results to return.
    """
    _ensure_did2content()

    normalized_qids = question_ids if isinstance(question_ids, list) else [question_ids]
    normalized_tasks = tasks if isinstance(tasks, list) else [tasks]

    if excluded_ids is None:
        normalized_excluded_ids = [["N/A"] for _ in query_list]
    elif isinstance(excluded_ids, list) and len(excluded_ids) == len(query_list):
        normalized_excluded_ids = excluded_ids
    else:
        normalized_excluded_ids = [excluded_ids for _ in query_list]

    if not (
        len(query_list) == len(normalized_qids) == len(normalized_tasks) == len(normalized_excluded_ids)
    ):
        raise ValueError(
            "perform_single_search_batch_4b expects one-to-one mapping among query_list/question_ids/tasks/excluded_ids, "
            f"but got len(query_list)={len(query_list)}, len(question_ids)={len(normalized_qids)}, "
            f"len(tasks)={len(normalized_tasks)}, len(excluded_ids)={len(normalized_excluded_ids)}"
        )

    final_all_scores = _batch_search(
        query_list,
        question_ids=normalized_qids,
        tasks=normalized_tasks,
        search_url=retrieval_service_url,
        excluded_ids=normalized_excluded_ids,
    )

    MAX_DOC_TOKENS = int(os.getenv("BRIGHT_MAX_DOC_TOKENS", "500"))

    def _truncate_doc(text, max_tokens=MAX_DOC_TOKENS):
        tokens = text.split()
        return " ".join(tokens[:max_tokens]) if len(tokens) > max_tokens else text

    final_doc = []
    for qid_task_id, did_scores in final_all_scores.items():
        task = qid_task_id[0]
        top_docs = [
            _truncate_doc(did2content.get((task, qid), "[Document not found]"))
            for qid, score in list(did_scores.items())[:topk]
        ]
        final_doc.append(" ".join(top_docs))

    result_text = json.dumps({"result": final_doc}, ensure_ascii=False)
    metadata = {
        "query_count": len(query_list),
        "queries": query_list,
        "api_request_error": None,
        "api_response": None,
        "status": None,
        "total_results": len(final_doc) * topk,
        "formatted_result": None,
    }

    return result_text, metadata
