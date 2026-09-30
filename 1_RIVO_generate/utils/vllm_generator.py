from typing import List, Optional
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams, TokensPrompt
import uuid

class vLLMAsyncGenerator:
    def __init__(self, model_path: str, tensor_parallel_size: int = 1, gpu_memory_utilization: float = 0.95):
        engine_args = AsyncEngineArgs(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        self.tokenizer = self.engine.engine.tokenizer.tokenizer  # Expose tokenizer for apply_chat_template
        self._closed = False

    async def generate(
        self,
        prompt_tokens: List[int],
        stop_tokens: Optional[List[int]] = None,
        stop_strings: Optional[List[str]] = None,
        temperature: float = 1.0,
        max_tokens: int = 0,
        return_logprobs: bool = False,
        seed: Optional[int] = None,
    ):
        if max_tokens == 0:
            max_tokens = None

        sp = SamplingParams(
            max_tokens=max_tokens,
            stop_token_ids=stop_tokens,
            stop=stop_strings,
            temperature=temperature,
            logprobs=None if not return_logprobs else 1,
            seed=seed,
        )
        prompt = TokensPrompt(prompt_token_ids=prompt_tokens)

        seen = 0
        rid = uuid.uuid4().hex
        async for req_out in self.engine.generate(
            prompt=prompt,
            sampling_params=sp,
            request_id=rid,
        ):
            assert req_out.request_id == rid
            out = req_out.outputs[0]
            token_ids = out.token_ids
            for tid in token_ids[seen:]:
                yield tid
            seen = len(token_ids)
            if getattr(req_out, "finished", False):
                break

    async def complete_with_logprobs(
        self,
        prompt_tokens: List[int],
        max_tokens: int = 256,
        temperature: float = 0.0,
        stop_strings: Optional[List[str]] = None,
    ) -> dict:
        """Generate a short ASAG answer probe while preserving logprobs."""
        sp = SamplingParams(
            max_tokens=max_tokens,
            stop=stop_strings,
            temperature=temperature,
            logprobs=1,
        )
        prompt = TokensPrompt(prompt_token_ids=prompt_tokens)
        rid = uuid.uuid4().hex
        final_output = None
        async for req_out in self.engine.generate(
            prompt=prompt,
            sampling_params=sp,
            request_id=rid,
        ):
            final_output = req_out.outputs[0]
            if getattr(req_out, "finished", False):
                break

        if final_output is None:
            raise RuntimeError("ASAG answer probe returned no output")

        selected_logprobs = []
        for token_id, candidates in zip(final_output.token_ids, final_output.logprobs or []):
            selected = candidates.get(token_id) if candidates else None
            value = getattr(selected, "logprob", None)
            selected_logprobs.append(float(value) if value is not None else None)

        return {
            "text": self.tokenizer.decode(final_output.token_ids, skip_special_tokens=True),
            "token_logprobs": selected_logprobs,
            "tokens": [
                self.tokenizer.decode([token_id], skip_special_tokens=False)
                for token_id in final_output.token_ids
            ],
            "finish_reason": getattr(final_output, "finish_reason", None),
        }

    def shutdown(self) -> None:
        if self._closed:
            return
        engine = getattr(self, "engine", None)
        try:
            if engine is not None and hasattr(engine, "shutdown"):
                engine.shutdown()
        finally:
            self.engine = None
            self._closed = True

    def __del__(self):
        try:
            if not getattr(self, "_closed", True):
                self.shutdown()
        except Exception:
            pass
