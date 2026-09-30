"""AgentLoopManager that updates the external plugin from outcome rewards."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

import numpy as np
import ray

from verl import DataProto
from verl.experimental.agent_loop.agent_loop import AgentLoopManager, AgentLoopWorkerBase

def _ensure_plugin_agent_registered() -> None:
    """Register the custom loop in the current process.

    Ray actor workers have their own Python interpreter and registry.  Importing
    this module on the driver is therefore not sufficient for the worker-side
    ``_agent_loop_registry``.
    """
    from verl.experimental.agent_loop.agent_loop import register
    from verl.experimental.plugin_grpo.deepresearch_agent import DeepResearchPluginAgentLoop

    register("deepresearch_plugin_agent")(DeepResearchPluginAgentLoop)


# Register in the driver process as well.
_ensure_plugin_agent_registered()


def _post_json(url: str, path: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{url.rstrip('/')}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Plugin service HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Cannot reach plugin service: {exc.reason}") from exc
    if not isinstance(result, dict):
        raise TypeError("Plugin service returned non-object JSON")
    return result


@ray.remote
class PluginAgentLoopWorker(AgentLoopWorkerBase):
    """Worker importing the custom loop before consulting the registry."""

    def __init__(self, config, server_handles, reward_router_address=None):
        # Each Ray actor owns an independent registry, so register inside the
        # actor process before AgentLoopWorkerBase starts handling requests.
        _ensure_plugin_agent_registered()
        super().__init__(config, server_handles, reward_router_address)


class PluginAgentLoopManager(AgentLoopManager):
    """Run frozen-Host trajectories, then perform one grouped plugin update."""

    def __init__(self, *args, **kwargs) -> None:
        self.agent_loop_workers_class = PluginAgentLoopWorker
        super().__init__(*args, **kwargs)
        plugin_config = self.config.get("plugin_grpo", {})
        self.plugin_url = str(plugin_config.get("url", "http://127.0.0.1:8010"))
        self.plugin_timeout = float(plugin_config.get("timeout", 600.0))
        self.plugin_agent_name = str(plugin_config.get("agent_name", "deepresearch_plugin_agent"))

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        validate = bool(prompts.meta_info.get("validate", False))
        prompts.non_tensor_batch["agent_name"] = np.array([self.plugin_agent_name] * len(prompts), dtype=object)
        prompts.non_tensor_batch["plugin_validate"] = np.array([validate] * len(prompts), dtype=object)
        output = super().generate_sequences(prompts)
        if validate:
            return output

        required = {"plugin_trace_id", "plugin_group_id"}
        missing = required - set(output.non_tensor_batch)
        if missing:
            raise RuntimeError(f"Plugin rollout did not return required fields: {sorted(missing)}")
        if "rm_scores" not in output.batch:
            raise RuntimeError(
                "Plugin GRPO requires reward_model.use_reward_loop=true so rewards are available after rollout"
            )
        rewards = output.batch["rm_scores"].sum(dim=-1).float().cpu().tolist()
        trace_ids = output.non_tensor_batch["plugin_trace_id"].tolist()
        group_ids = output.non_tensor_batch["plugin_group_id"].tolist()
        items = [
            {"trace_id": str(trace_id), "group_id": str(group_id), "reward": float(reward)}
            for trace_id, group_id, reward in zip(trace_ids, group_ids, rewards, strict=True)
        ]
        started = time.perf_counter()
        metrics = _post_json(
            self.plugin_url,
            "/update",
            {"items": items},
            self.plugin_timeout,
        )
        output.meta_info.setdefault("timing", {})["plugin_grpo/update"] = time.perf_counter() - started
        values = np.empty(len(output), dtype=object)
        values[:] = [metrics] * len(output)
        output.non_tensor_batch["plugin_update_metrics"] = values
        return output

    def save_plugin_checkpoint(self, step: int) -> str:
        result = _post_json(
            self.plugin_url,
            "/save",
            {"step": int(step)},
            self.plugin_timeout,
        )
        return str(result["path"])
