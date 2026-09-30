"""
Async batch search gateway: collects concurrent search requests from multiple
agent loop coroutines and batches them into single API calls.

Architecture:
  agent_loop_1 ──┐
  agent_loop_2 ──┤──> AsyncBatchSearchGateway ──> _batch_search(all_queries) ──> HTTP POST(s)
  agent_loop_3 ──┤                                                   one per task
  ...           ──┘

Instead of N individual HTTP requests (one per sample), the gateway sends
~T requests (one per unique task in the batch), where T << N.

This dramatically reduces HTTP overhead when many agent loops run concurrently
(e.g. batch_size=128, n=4 rollouts → up to 512 concurrent searches batched
into ~12 requests, one per BRIGHT task).
"""

import asyncio
import json
import logging
import os
from collections import defaultdict
from typing import Any, Optional

# Maximum tokens (whitespace-split words) per retrieved document
MAX_DOC_TOKENS = int(os.getenv("BRIGHT_MAX_DOC_TOKENS", "500"))


def _truncate_doc(text: str, max_tokens: int = MAX_DOC_TOKENS) -> str:
    """Truncate a document to at most max_tokens whitespace-split tokens."""
    tokens = text.split()
    if len(tokens) <= max_tokens:
        return text
    return " ".join(tokens[:max_tokens])


logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Module-level singleton gateway, one per process (each AgentLoopWorker Ray actor)
_gateway_instance: Optional["AsyncBatchSearchGateway"] = None


class AsyncBatchSearchGateway:
    """Batches concurrent search requests from multiple agent loops into single API calls.

    When multiple agent loop coroutines need to search at roughly the same time
    (which happens naturally since LLM generation is batched), this gateway
    collects their queries and sends them to the retrieval service in a single
    batched request per task.

    Usage:
        gateway = get_batch_search_gateway(search_url, topk)
        result = await gateway.search(query="...", qid="...", task="...")
    """

    def __init__(
        self,
        search_url: str,
        topk: int = 3,
        batch_wait_time: float = 0.02,
        max_batch_size: int = 1024,
    ):
        """Initialize the batch search gateway.

        Args:
            search_url: URL of the retrieval service.
            topk: Number of top document results to return per query.
            batch_wait_time: Seconds to wait for more requests before flushing.
            max_batch_size: Maximum queries per batch before immediate flush.
        """
        self.search_url = search_url
        self.topk = topk
        self.batch_wait_time = batch_wait_time
        self.max_batch_size = max_batch_size
        self._lock = asyncio.Lock()
        self._current_batch: Optional["_PendingBatch"] = None

    async def search(self, query: str, qid: str, task: str, excluded_ids: Optional[list[str]] = None) -> str:
        """Submit a single search request. Returns when the batched result is available.

        This method is called concurrently by multiple agent loop coroutines.
        Requests are accumulated into a batch and flushed either when:
        - The batch reaches max_batch_size, OR
        - batch_wait_time has elapsed since the first request in the batch.

        Args:
            query: Search query text.
            qid: Question ID for the retrieval service.
            task: Task name (e.g. "biology", "economics").

        Returns:
            JSON string containing search results: {"result": [doc_text]}
        """
        loop = asyncio.get_event_loop()
        future = loop.create_future()

        batch_to_execute = None

        async with self._lock:
            if self._current_batch is None:
                self._current_batch = _PendingBatch(self)

            self._current_batch.add(query, qid, task, excluded_ids, future)

            if self._current_batch.size >= self.max_batch_size:
                batch_to_execute = self._current_batch
                self._current_batch = None
            elif self._current_batch.size == 1:
                asyncio.ensure_future(self._delayed_flush(self._current_batch))

        if batch_to_execute is not None:
            asyncio.ensure_future(batch_to_execute.execute())

        return await future

    async def _delayed_flush(self, batch: "_PendingBatch"):
        """Wait for batch_wait_time, then flush if batch is still the current one."""
        await asyncio.sleep(self.batch_wait_time)
        should_execute = False
        async with self._lock:
            if self._current_batch is batch:
                self._current_batch = None
                should_execute = True
        if should_execute and not batch.executed:
            await batch.execute()


class _PendingBatch:
    """A batch of pending search requests that will be executed together."""

    def __init__(self, gateway: AsyncBatchSearchGateway):
        self.gateway = gateway
        self.queries: list[str] = []
        self.qids: list[str] = []
        self.tasks: list[str] = []
        self.excluded_ids: list[Optional[list[str]]] = []
        self.futures: list[asyncio.Future] = []
        self.executed = False

    @property
    def size(self) -> int:
        return len(self.queries)

    def add(self, query: str, qid: str, task: str, excluded_ids: Optional[list[str]], future: asyncio.Future):
        self.queries.append(query)
        self.qids.append(qid)
        self.tasks.append(task)
        self.excluded_ids.append(excluded_ids)
        self.futures.append(future)

    async def execute(self):
        """Execute the batched search and distribute results to all waiting coroutines."""
        if self.executed:
            return
        self.executed = True

        batch_size = self.size
        logger.info(f"[BatchSearchGateway] Executing batch of {batch_size} search queries")

        try:
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(None, self._do_search)

            for i, future in enumerate(self.futures):
                if not future.done():
                    future.set_result(results[i])

            logger.info(
                f"[BatchSearchGateway] Batch of {batch_size} queries completed successfully"
            )
        except Exception as e:
            logger.error(
                f"[BatchSearchGateway] Batch search failed for {batch_size} queries: {e}"
            )
            for future in self.futures:
                if not future.done():
                    future.set_exception(e)

    def _do_search(self) -> list[str]:
        """Execute the actual batch search (blocking, runs in thread pool).

        Calls _batch_search from search_r1_like_utils with all accumulated queries,
        then maps results back to the original request order.

        Returns:
            List of JSON result strings, one per original request, in order.
        """
        from verl.tools.utils.search_r1_like_utils import _batch_search, did2content, _ensure_did2content

        # Ensure document cache is loaded before looking up results
        _ensure_did2content()

        final_scores = _batch_search(
            queries=self.queries,
            question_ids=self.qids,
            tasks=self.tasks,
            search_url=self.gateway.search_url,
            excluded_ids=self.excluded_ids,
        )

        # Map results back to original request order.
        task_indices = defaultdict(list)
        for i, task in enumerate(self.tasks):
            task_indices[task].append(i)

        results = [None] * len(self.queries)
        task_counters = defaultdict(int)

        for (task, qid, inst_id), did_scores in final_scores.items():
            counter = task_counters[task]
            if counter < len(task_indices[task]):
                orig_idx = task_indices[task][counter]
                task_counters[task] += 1

                top_docs = [
                    _truncate_doc(did2content.get((task, did), "[Document not found]"))
                    for did, score in list(did_scores.items())[: self.gateway.topk]
                ]
                result_text = json.dumps(
                    {"result": [" ".join(top_docs)]}, ensure_ascii=False
                )
                results[orig_idx] = result_text

        # Fill any unmapped results (defensive)
        for i in range(len(results)):
            if results[i] is None:
                logger.warning(
                    f"[BatchSearchGateway] No result for query index {i} "
                    f"(qid={self.qids[i]}, task={self.tasks[i]}). Using fallback."
                )
                results[i] = json.dumps({"result": ["No search results found."]})

        return results


def get_batch_search_gateway(
    search_url: str,
    topk: int = 3,
    batch_wait_time: float = 0.02,
    max_batch_size: int = 1024,
) -> AsyncBatchSearchGateway:
    """Get or create the module-level singleton batch search gateway.

    Each AgentLoopWorker process gets its own singleton gateway, which batches
    search requests from all concurrent agent loop coroutines in that process.
    """
    global _gateway_instance
    if _gateway_instance is None:
        _gateway_instance = AsyncBatchSearchGateway(
            search_url=search_url,
            topk=topk,
            batch_wait_time=batch_wait_time,
            max_batch_size=max_batch_size,
        )
        logger.info(
            f"[BatchSearchGateway] Created singleton: url={search_url}, topk={topk}, "
            f"wait={batch_wait_time}s, max_batch={max_batch_size}"
        )
    return _gateway_instance


def reset_batch_search_gateway():
    """Reset the singleton gateway (useful for testing or reconfiguration)."""
    global _gateway_instance
    _gateway_instance = None
