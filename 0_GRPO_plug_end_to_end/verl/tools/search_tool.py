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
from contextlib import ExitStack
from enum import Enum
from typing import Any, Callable, Optional, TypeVar
from uuid import uuid4
import asyncio

import ray
import ray.actor

from verl.tools.utils.search_r1_like_utils import perform_single_search_batch, perform_single_search_batch_4b
from verl.utils.rollout_trace import rollout_trace_op

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

T = TypeVar("T")


# Adapted from verl/tools/sandbox_fusion_tools.py
class PoolMode(Enum):
    """Execution pool mode enumeration."""

    ThreadMode = 1
    ProcessMode = 2


@ray.remote(concurrency_groups={"acquire": 1, "release": 10})
class TokenBucketWorker:
    """Ray actor for rate limiting using token bucket algorithm."""

    def __init__(self, rate_limit: int):
        self.rate_limit = rate_limit
        self.current_count = 0  # For observability
        self._semaphore = threading.Semaphore(rate_limit)

    @ray.method(concurrency_group="acquire")
    def acquire(self):
        """Acquire a token from the bucket."""
        self._semaphore.acquire()
        self.current_count += 1

    @ray.method(concurrency_group="release")
    def release(self):
        """Release a token back to the bucket."""
        self._semaphore.release()
        self.current_count -= 1

    def get_current_count(self):
        """Get current number of acquired tokens."""
        return self.current_count


class SearchExecutionWorker:
    """Worker for executing search operations with optional rate limiting."""

    def __init__(self, enable_global_rate_limit=True, rate_limit=10):
        self.rate_limit_worker = self._init_rate_limit(rate_limit) if enable_global_rate_limit else None

    def _init_rate_limit(self, rate_limit):
        """Initialize singleton rate limiter."""
        return TokenBucketWorker.options(name="rate-limiter", get_if_exists=True).remote(rate_limit)

    def ping(self):
        """Health check method."""
        return True

    def execute(self, fn: Callable[..., T], *fn_args, **fn_kwargs) -> T:
        """Execute function with optional rate limiting."""
        if self.rate_limit_worker:
            with ExitStack() as stack:
                stack.callback(self.rate_limit_worker.release.remote)
                try:
                    ray.get(self.rate_limit_worker.acquire.remote(), timeout=60)
                except ray.exceptions.ActorDiedError as e:
                    logger.warning(f"Rate limiter actor died, fallback to direct execution: {e}")
                    return fn(*fn_args, **fn_kwargs)
                except Exception as e:
                    logger.error(f"Error acquiring rate limit token: {e}")
                    raise
                try:
                    return fn(*fn_args, **fn_kwargs)
                except Exception as e:
                    logger.warning(f"Error when executing search: {e}")
                    raise
        else:
            return fn(*fn_args, **fn_kwargs)


# Module-level strong reference to the singleton SearchExecutionWorker actor.
# CRITICAL: keeps the actor alive even when individual SearchTool instances are GC'd.
_SEARCH_EXECUTION_POOL_HANDLE = None


def init_search_execution_pool(
    num_workers: int, enable_global_rate_limit=True, rate_limit=10, mode: PoolMode = PoolMode.ThreadMode,
    pool_name: str = "search-execution-pool",
):
    """Initialize (or retrieve) the singleton search execution pool actor."""
    global _SEARCH_EXECUTION_POOL_HANDLE
    if mode == PoolMode.ThreadMode:
        # Fast path: already initialized in this process — reuse the SAME object.
        if _SEARCH_EXECUTION_POOL_HANDLE is not None:
            return _SEARCH_EXECUTION_POOL_HANDLE

        # Slow path (first call in this process): get existing named actor or create one.
        try:
            handle = ray.get_actor(pool_name)
            logger.info(f"[SearchTool] Reusing existing named actor '{pool_name}'")
        except ValueError:
            handle = (
                ray.remote(SearchExecutionWorker)
                .options(
                    max_concurrency=num_workers,
                    name=pool_name,
                    get_if_exists=True,
                )
                .remote(enable_global_rate_limit=enable_global_rate_limit, rate_limit=rate_limit)
            )
            logger.info(f"[SearchTool] Created new named actor '{pool_name}' with {num_workers} workers")

        _SEARCH_EXECUTION_POOL_HANDLE = handle
        return handle
    else:
        raise NotImplementedError("Process mode is not implemented yet")


class SearchTool(BaseTool):
    """Search tool for retrieving information using external retrieval services.

    This tool provides search functionality with rate limiting and concurrent execution
    support through Ray. It integrates with external retrieval services to perform
    semantic search operations.

    Methods:
        get_openai_tool_schema: Return the tool schema in OpenAI format
        create: Create a tool instance for a trajectory
        execute: Execute the search tool
        calc_reward: Calculate the reward with respect to tool state
        release: Release the tool instance
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """Initialize SearchTool with configuration and schema.

        Args:
            config: Configuration dictionary containing tool settings
            tool_schema: OpenAI function tool schema definition

        Example tool_schema:
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Searches for relevant information based on queries.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query_list": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of search queries"
                            }
                        },
                        "required": ["query_list"]
                    }
                }
            }
        """
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        self._pool_retries = config.get("pool_retries", 1)

        # Worker and rate limiting configuration
        self.num_workers = config.get("num_workers", 32)
        self.rate_limit = config.get("rate_limit", 32)
        self.timeout = config.get("timeout", 30)

        self.enable_global_rate_limit = False
        if config.get("enable_global_rate_limit", False):
            logger.warning("Ignoring enable_global_rate_limit=True to avoid TokenBucketWorker instability.")
        self.execution_pool = init_search_execution_pool(
            num_workers=self.num_workers,
            enable_global_rate_limit=self.enable_global_rate_limit,
            rate_limit=self.rate_limit,
            mode=PoolMode.ThreadMode,
            pool_name="search-execution-pool",
        )

        # Retrieval service configuration
        self.retrieval_service_url = config.get("retrieval_service_url")
        assert self.retrieval_service_url, "Configuration must include 'retrieval_service_url'"
        self.topk = config.get("topk", 3)
        if self.retrieval_service_url == "":
            raise ValueError("retrieval_service_url is not set")

        logger.info(f"Initialized SearchTool with config: {config}")

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        """Return the OpenAI tool schema."""
        return self.tool_schema

    def _recreate_execution_pool(self):
        """Recreate the execution pool after actor death."""
        global _SEARCH_EXECUTION_POOL_HANDLE
        _SEARCH_EXECUTION_POOL_HANDLE = None
        self.execution_pool = init_search_execution_pool(
            num_workers=self.num_workers,
            enable_global_rate_limit=self.enable_global_rate_limit,
            rate_limit=self.rate_limit,
            mode=PoolMode.ThreadMode,
            pool_name="search-execution-pool",
        )

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        """Create a tool instance.

        Args:
            instance_id: The instance id of the tool.

        Returns:
            The instance id of the tool.
            tool_creation_response: The response of the tool when creating the instance.
        """
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "response": "",
            "reward": [],
        }
        return instance_id, ToolResponse()

    def execute_search(
        self,
        instance_id: str,
        query_list: list,
        question_ids: list,
        tasks: list,
        retrieval_service_url: str,
        topk: int,
        timeout: int,
        excluded_ids: Optional[list[str]] = None,
    ):
        """Execute search operation using BRIGHT retrieval service.

        Args:
            instance_id: Tool instance ID
            query_list: List of search queries
            question_ids: List of question IDs
            tasks: List of tasks
            excluded_ids: Document IDs to exclude from retrieval
            retrieval_service_url: URL of the retrieval service
            topk: Number of top results to return
            timeout: Request timeout in seconds

        Returns:
            Tuple of (result_text, metadata)
        """
        result_text, metadata = perform_single_search_batch_4b(
            retrieval_service_url=retrieval_service_url,
            query_list=query_list,
            question_ids=question_ids,
            tasks=tasks,
            excluded_ids=excluded_ids,
            topk=topk,
            concurrent_semaphore=None,
            timeout=timeout,
        )
        logger.debug(f"Search result for instance {instance_id}: {result_text}")
        return result_text, metadata

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        """Execute the search tool using batch search gateway.

        Instead of sending individual HTTP requests per sample, the gateway collects
        concurrent search requests from all agent loops and batches them into
        a single API call per task.

        Args:
            instance_id: The instance ID of the tool
            parameters: Tool parameters containing query_list, question_ids, tasks

        Returns: tool_response, tool_reward_score, tool_metrics
        """
        query_list_from_params = parameters.get("query_list")
        agent_data = kwargs.get("agent_data")

        if not query_list_from_params or not isinstance(query_list_from_params, list):
            error_msg = "Error: 'query_list' is missing, empty, or not a list in parameters."
            logger.error(f"[SearchTool] {error_msg} Received parameters: {parameters}")
            return ToolResponse(text=json.dumps({"result": error_msg})), 0.0, {}

        # Use the batch search gateway: concurrent requests from all agent loops
        # are automatically batched into a single API call per task.
        try:
            from verl.tools.utils.batch_search_gateway import get_batch_search_gateway

            gateway = get_batch_search_gateway(
                search_url=self.retrieval_service_url,
                topk=self.topk,
                batch_wait_time=float(os.getenv("BRIGHT_BATCH_WAIT_TIME", "0.12")),
                max_batch_size=int(os.getenv("BRIGHT_BATCH_MAX_SIZE", "1024")),
            )

            excluded_ids = kwargs.get("extra_info", {}).get("excluded_ids")
            if excluded_ids is None:
                excluded_ids = parameters.get("excluded_ids")
            if excluded_ids is None and agent_data is not None:
                excluded_ids = getattr(agent_data, "excluded_ids", None)

            result_text = await gateway.search(
                query=query_list_from_params[0],
                qid=agent_data.qids,
                task=agent_data.tasks,
                excluded_ids=excluded_ids,
            )

            # Store results in instance dictionary
            self._instance_dict[instance_id]["reward"].append(result_text.strip())

            metrics = {
                "query_count": 1,
                "status": "success",
                "total_results": self.topk,
                "api_request_error": None,
            }

            return ToolResponse(text=result_text), 0.0, metrics

        except Exception as e:
            error_result = json.dumps({"result": f"Search execution failed: {e}"})
            logger.error(f"[SearchTool] Batch search execution failed: {e}")
            return ToolResponse(text=error_result), 0.0, {"error": str(e)}

    async def calc_reward(self, instance_id: str, **kwargs) -> str:
        return self._instance_dict[instance_id]["reward"]

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
