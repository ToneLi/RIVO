#!/usr/bin/env bash
# GRPO for a frozen LiteResearcher-4B Host and a trainable 0.6B control/correction plugin.

set -euo pipefail
set -x
ulimit -n 65535

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config"
HOST_MODEL="${HOST_MODEL:-simplex-ai-inc/LiteResearcher-4B}"
BASELINES_ROOT="${BASELINES_ROOT:-}"
export CLASSIFIER_ADAPTER="${CLASSIFIER_ADAPTER:-$PROJECT_DIR/intial}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-/global_step_50}"
RESUME_ROOT="$(dirname "$RESUME_CHECKPOINT")"

PLUGIN_GPU="${PLUGIN_GPU:-2}"
HOST_GPUS="${HOST_GPUS:-6,7}"
PLUGIN_GRPO_PORT="${PLUGIN_GRPO_PORT:-8011}"
PLUGIN_GRPO_URL="http://127.0.0.1:$PLUGIN_GRPO_PORT"
PLUGIN_OUTPUT_DIR="${PLUGIN_OUTPUT_DIR:-$RESUME_ROOT}"
PLUGIN_LOG="${PLUGIN_LOG:-$PROJECT_DIR/deepresearch_hint_e2e_grpo_service.log}"
DEEPRESEARCH_SEARCH_URL="${DEEPRESEARCH_SEARCH_URL:-http://127.0.0.1:8001}"
OPENAI_ENV_FILE="${OPENAI_ENV_FILE:-}"

export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PLUGIN_GRPO_URL DEEPRESEARCH_SEARCH_URL
export PLUGIN_MIN_HINT_ROUND="${MIN_HINT_ROUND:-8}"
export PLUGIN_MAX_HINTS_PER_TRAJECTORY="${MAX_HINTS_PER_TRAJECTORY:-1}"
export PLUGIN_HINT_COOLDOWN_ROUNDS="${HINT_COOLDOWN_ROUNDS:-4}"
export DEEPRESEARCH_EVAL_PATH="${DEEPRESEARCH_EVAL_PATH:-}"
export LLM_JUDGE_MODEL="${LLM_JUDGE_MODEL:-gpt-4.1-2025-04-14}"
export LLM_JUDGE_QPS="${LLM_JUDGE_QPS:-50}"
export LLM_JUDGE_MAX_RETRIES="${LLM_JUDGE_MAX_RETRIES:-5}"
export VERL_MAX_CONCURRENT_PER_WORKER="${VERL_MAX_CONCURRENT_PER_WORKER:-16}"
export RAY_PORT="${RAY_PORT:-6393}"
export RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8272}"
export RAY_worker_register_timeout_seconds="${RAY_worker_register_timeout_seconds:-600}"
export RAY_max_pending_calls_per_actor="${RAY_max_pending_calls_per_actor:-10}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [ ! -d "$CLASSIFIER_ADAPTER" ] || [ ! -f "$CLASSIFIER_ADAPTER/adapter_config.json" ]; then
    echo "Error: 0.6B classifier adapter is missing: $CLASSIFIER_ADAPTER" >&2
    exit 1
fi
RESUME_STEP="${RESUME_CHECKPOINT##*/global_step_}"
if ! [[ "$RESUME_STEP" =~ ^[0-9]+$ ]]; then
    echo "Error: RESUME_CHECKPOINT must end with global_step_<number>: $RESUME_CHECKPOINT" >&2
    exit 1
fi
PLUGIN_RESUME_CHECKPOINT="$RESUME_CHECKPOINT/plugin"
for required_file in \
    "$RESUME_CHECKPOINT/data.pt" \
    "$PLUGIN_RESUME_CHECKPOINT/plugin_heads.pt" \
    "$PLUGIN_RESUME_CHECKPOINT/optimizer.pt" \
    "$PLUGIN_RESUME_CHECKPOINT/lora_adapter/adapter_config.json"; do
    if [ ! -f "$required_file" ]; then
        echo "Error: resume checkpoint is incomplete; missing: $required_file" >&2
        exit 1
    fi
done
if [ "$PLUGIN_OUTPUT_DIR" != "$RESUME_ROOT" ]; then
    echo "Error: PLUGIN_OUTPUT_DIR must equal RESUME_ROOT for in-place resume." >&2
    exit 1
fi
echo "Resuming plugin GRPO from step $RESUME_STEP: $RESUME_CHECKPOINT"
if [[ ",$HOST_GPUS," == *",$PLUGIN_GPU,"* ]]; then
    echo "Error: PLUGIN_GPU=$PLUGIN_GPU overlaps HOST_GPUS=$HOST_GPUS" >&2
    exit 1
fi

set +x
if [ -f "$OPENAI_ENV_FILE" ]; then
    set -a
    source "$OPENAI_ENV_FILE"
    set +a
fi
if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "Error: OPENAI_API_KEY is unset and was not found in $OPENAI_ENV_FILE" >&2
    exit 1
fi
export OPENAI_API_KEY
set -x

if ! curl -fsS --max-time 5 "$DEEPRESEARCH_SEARCH_URL/" >/dev/null; then
    echo "Error: retrieval service is not ready at $DEEPRESEARCH_SEARCH_URL" >&2
    exit 1
fi

"$PYTHON_BIN" -c '
from transformers import AutoConfig
import sys
config = AutoConfig.from_pretrained(sys.argv[1], trust_remote_code=True)
if config.model_type != "qwen3":
    raise RuntimeError(f"Expected Qwen3 Host, got {config.model_type}")
print(f"Host preflight passed: {sys.argv[1]} vocab={config.vocab_size}")
' "$HOST_MODEL"

TOOL_CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config/tool_config/deepresearch_browser_tool_config.yaml"
"$PYTHON_BIN" -c '
import sys
from verl.tools.utils.tool_registry import initialize_tools_from_config
names = [tool.name for tool in initialize_tools_from_config(sys.argv[1])]
expected = ["browser.search", "browser.open", "browser.find"]
if names != expected:
    raise RuntimeError(f"Unexpected browser tools: {names}; expected {expected}")
print(f"Browser tool preflight passed: {names}")
' "$TOOL_CONFIG_PATH"

"$PYTHON_BIN" scripts/prepare_deepresearch_grpo_data.py
TRAIN_FILE="$PROJECT_DIR/data/deepresearch/train_dev.parquet"
VAL_FILE="$PROJECT_DIR/data/deepresearch/test.parquet"

PLUGIN_PID=""
cleanup() {
    if [ -n "$PLUGIN_PID" ] && kill -0 "$PLUGIN_PID" 2>/dev/null; then
        kill "$PLUGIN_PID"
        wait "$PLUGIN_PID" || true
    fi
}
trap cleanup EXIT INT TERM

CUDA_VISIBLE_DEVICES="$PLUGIN_GPU" "$PYTHON_BIN" -m verl.experimental.plugin_grpo.service     --host-model "$HOST_MODEL"     --adapter-path "$CLASSIFIER_ADAPTER"     --output-dir "$PLUGIN_OUTPUT_DIR"     --port "$PLUGIN_GRPO_PORT"     --correction-rank "${CORRECTION_RANK:-64}"     --alpha "${CORRECTION_ALPHA:-20}"     --learning-rate "${PLUGIN_LEARNING_RATE:-1e-5}"     --correction-learning-rate "${CORRECTION_LEARNING_RATE:-1e-4}"     --clip-ratio "${GRPO_CLIP_RATIO:-0.2}"     --controller-temperature "${CONTROLLER_TEMPERATURE:-1.0}"     --reroute-temperature "${REROUTE_TEMPERATURE:-1.0}"     --reroute-top-p "${REROUTE_TOP_P:-0.95}"     --reroute-max-new-tokens "${REROUTE_MAX_NEW_TOKENS:-64}"     --hint-policy     --resume "$PLUGIN_RESUME_CHECKPOINT"     --hint-slot-token-budgets "${HINT_SLOT_BUDGETS:-6,6,6}"     --hint-temperature "${HINT_TEMPERATURE:-1.0}"     --hint-top-p "${HINT_TOP_P:-0.95}"     --hint-correction-topk "${HINT_CORRECTION_TOPK:-128}"     --hint-query-max-words "${HINT_QUERY_MAX_WORDS:-24}"     --hint-quality-reward-weight "${HINT_QUALITY_REWARD_WEIGHT:-0.5}"     --hint-quality-max-field-words "${HINT_QUALITY_MAX_FIELD_WORDS:-8}"     --min-stop-rounds "${MIN_STOP_ROUNDS:-15}"     --min-reroute-rounds "${MIN_REROUTE_ROUNDS:-8}"     --reroute-cooldown-rounds "${REROUTE_COOLDOWN_ROUNDS:-4}"     >"$PLUGIN_LOG" 2>&1 &
PLUGIN_PID=$!

for _ in $(seq 1 120); do
    if curl -fsS --max-time 5 "$PLUGIN_GRPO_URL/health" >/dev/null; then break; fi
    if ! kill -0 "$PLUGIN_PID" 2>/dev/null; then
        echo "Error: plugin GRPO service exited; see $PLUGIN_LOG" >&2
        exit 1
    fi
    sleep 5
done
if ! curl -fsS --max-time 5 "$PLUGIN_GRPO_URL/health" >/dev/null; then
    echo "Error: plugin GRPO service did not become ready; see $PLUGIN_LOG" >&2
    exit 1
fi

IFS=',' read -r -a HOST_GPU_ARRAY <<<"$HOST_GPUS"
NUM_HOST_GPUS="${#HOST_GPU_ARRAY[@]}"

CUDA_VISIBLE_DEVICES="$HOST_GPUS" "$PYTHON_BIN" -m verl.trainer.main_ppo     --config-path="$CONFIG_PATH"     --config-name=deepresearch_multiturn_grpo     +plugin_grpo.enabled=true     +plugin_grpo.url="$PLUGIN_GRPO_URL"     +plugin_grpo.timeout=1200     +plugin_grpo.agent_name=deepresearch_plugin_agent     +plugin_grpo.hint_policy=true     +plugin_grpo.min_hint_round="${MIN_HINT_ROUND:-8}"     +plugin_grpo.max_hints_per_trajectory="${MAX_HINTS_PER_TRAJECTORY:-1}"     +plugin_grpo.hint_cooldown_rounds="${HINT_COOLDOWN_ROUNDS:-4}"     algorithm.adv_estimator=grpo     algorithm.norm_adv_by_std_in_grpo=True     algorithm.use_kl_in_reward=False     data.train_files="$TRAIN_FILE"     data.val_files="$VAL_FILE"     data.train_batch_size="${TRAIN_BATCH_SIZE:-8}"     data.max_prompt_length=4096     data.max_response_length=32768     data.dataloader_num_workers=0     data.filter_overlong_prompts=True     data.truncation=error     data.return_raw_chat=True     actor_rollout_ref.model.path="$HOST_MODEL"     actor_rollout_ref.model.use_remove_padding=True     actor_rollout_ref.model.enable_gradient_checkpointing=False     actor_rollout_ref.actor.optim.lr=0     actor_rollout_ref.actor.ppo_mini_batch_size="${TRAIN_BATCH_SIZE:-8}"     actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1     actor_rollout_ref.actor.use_dynamic_bsz=True     actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24576     actor_rollout_ref.actor.ulysses_sequence_parallel_size="$NUM_HOST_GPUS"     actor_rollout_ref.actor.use_torch_compile=False     actor_rollout_ref.actor.use_kl_loss=False     actor_rollout_ref.actor.entropy_coeff=0     actor_rollout_ref.actor.fsdp_config.param_offload=True     actor_rollout_ref.actor.fsdp_config.optimizer_offload=False     actor_rollout_ref.rollout.tensor_model_parallel_size=1     actor_rollout_ref.rollout.name=sglang     actor_rollout_ref.rollout.mode=async     actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.6}"     actor_rollout_ref.rollout.enforce_eager=True     actor_rollout_ref.rollout.multi_stage_wake_up=True     actor_rollout_ref.rollout.calculate_log_probs=False     actor_rollout_ref.rollout.n="${GRPO_GROUP_SIZE:-5}"     actor_rollout_ref.rollout.free_cache_engine=True     actor_rollout_ref.rollout.trace.backend=none     actor_rollout_ref.rollout.trace.token2text=True     actor_rollout_ref.rollout.agent.default_agent_loop=deepresearch_plugin_agent     +actor_rollout_ref.rollout.agent.agent_loop_manager_class=verl.experimental.plugin_grpo.manager.PluginAgentLoopManager     actor_rollout_ref.rollout.multi_turn.enable=True     actor_rollout_ref.rollout.multi_turn.format=hermes     actor_rollout_ref.rollout.multi_turn.max_assistant_turns=60     actor_rollout_ref.rollout.multi_turn.max_tokens_per_assistant_turn=4096     actor_rollout_ref.rollout.multi_turn.max_user_turns=60     actor_rollout_ref.rollout.multi_turn.max_tool_response_length=65536     actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG_PATH"     trainer.critic_warmup=0     trainer.logger='["wandb"]'     trainer.project_name=DeepResearch     trainer.experiment_name=LiteResearcher4B_frozen_Qwen06B_hint_E2E_GRPO     trainer.rollout_data_dir="$PROJECT_DIR/deepresearch_hint_e2e_grpo_rollouts"     trainer.default_local_dir="$PLUGIN_OUTPUT_DIR"     trainer.resume_mode=resume_path     trainer.resume_from_path="$RESUME_CHECKPOINT"     trainer.n_gpus_per_node="$NUM_HOST_GPUS"     trainer.nnodes=1     trainer.save_freq="${SAVE_FREQ:-10}"     trainer.test_freq="${TEST_FREQ:-10}"     trainer.total_training_steps="${TOTAL_TRAINING_STEPS:-403}"     trainer.total_epochs=150     trainer.val_before_train=False
