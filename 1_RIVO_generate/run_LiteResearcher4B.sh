#!/usr/bin/env bash
# ASAG keeps Stop/Verify authority. Its Reroute decision and the trained 0.6B
# controller jointly trigger a corrected three-slot hint; the Host generates the query.
#
# Single-worker topology. The sidecar's frozen Host supplies ASAG attention:
#   GPU 2: one dense-search service on port 8034
#   GPU 3: one TP=1 vLLM Host on 8033 + GRPO/ASAG sidecar on 8032

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

CONFIG_FILE="${CONFIG_FILE:-$PROJECT_ROOT/config.env}"
if [ -f "$CONFIG_FILE" ]; then
    # shellcheck disable=SC1090
    set -a
    source "$CONFIG_FILE"
    set +a
fi

DEFAULT_MODEL="simplex-ai-inc/LiteResearcher-4B"
MODEL="${1:-${MODEL:-$DEFAULT_MODEL}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
PROMPT_FILE="${PROMPT_FILE:-$PROJECT_ROOT/prompts/liter_research_prompt.py}"

VLLM_GPU_1="${VLLM_GPU_1:-5}"
SEARCH_GPU="${SEARCH_GPU:-4}"

PLUGIN_PORT_1="${PLUGIN_PORT_1:-8092}"
VLLM_PORT_1="${VLLM_PORT_1:-8023}"
SEARCH_PORT_1="${SEARCH_PORT_1:-8014}"
VLLM_PORT_2="${VLLM_PORT_2:-8028}"
PLUGIN_PORT_2="${PLUGIN_PORT_2:-8029}"
SEARCH_PORT_2="${SEARCH_PORT_2:-8003}"

PLUGIN_GRPO_SOURCE_ROOT="${PLUGIN_GRPO_SOURCE_ROOT:-$PROJECT_ROOT/../0_GRPO_plug_end_to_end}"
PLUGIN_CHECKPOINT="${PLUGIN_CHECKPOINT:-$PLUGIN_GRPO_SOURCE_ROOT/RIVO_generate/Plugin}"
PLUGIN_ALPHA="${PLUGIN_ALPHA:-20.0}"
PLUGIN_REROUTE_SEED="${PLUGIN_REROUTE_SEED:-20260907}"
PLUGIN_REROUTE_FORMAT_ATTEMPTS="${PLUGIN_REROUTE_FORMAT_ATTEMPTS:-3}"
PLUGIN_CONTROLLER_TEMPERATURE="${PLUGIN_CONTROLLER_TEMPERATURE:-1.0}"
PLUGIN_HINT_SLOT_BUDGETS="${PLUGIN_HINT_SLOT_BUDGETS:-16,12,12}"
PLUGIN_HINT_TEMPERATURE="${PLUGIN_HINT_TEMPERATURE:-1.0}"
PLUGIN_HINT_TOP_P="${PLUGIN_HINT_TOP_P:-0.95}"
PLUGIN_HINT_CORRECTION_TOPK="${PLUGIN_HINT_CORRECTION_TOPK:-128}"
PLUGIN_HINT_QUERY_MAX_WORDS="${PLUGIN_HINT_QUERY_MAX_WORDS:-24}"
GRPO_HOST_QUERY_TEMPERATURE="${GRPO_HOST_QUERY_TEMPERATURE:-0.7}"
GRPO_HOST_QUERY_MAX_NEW_TOKENS="${GRPO_HOST_QUERY_MAX_NEW_TOKENS:-32}"
GRPO_HOST_QUERY_MAX_WORDS="${GRPO_HOST_QUERY_MAX_WORDS:-16}"
PLUGIN_MAX_LENGTH="${PLUGIN_MAX_LENGTH:-4096}"
JOINT_CONTROLLER_ENABLED="${JOINT_CONTROLLER_ENABLED:-1}"
JOINT_CONTROLLER_REROUTE_THRESHOLD="${JOINT_CONTROLLER_REROUTE_THRESHOLD:-0.25}"
JOINT_CONTROLLER_MAX_REROUTES="${JOINT_CONTROLLER_MAX_REROUTES:-2}"
JOINT_CONTROLLER_MIN_REROUTE_ROUNDS="${JOINT_CONTROLLER_MIN_REROUTE_ROUNDS:-8}"
JOINT_CONTROLLER_REROUTE_COOLDOWN_ROUNDS="${JOINT_CONTROLLER_REROUTE_COOLDOWN_ROUNDS:-4}"

VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.65}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-3600}"
MAX_ROUNDS="${MAX_ROUNDS:-80}"
GRPO_REROUTE_SEARCH_ATTEMPTS="${GRPO_REROUTE_SEARCH_ATTEMPTS:-2}"
GRPO_MIN_POST_REROUTE_ROUNDS="${GRPO_MIN_POST_REROUTE_ROUNDS:-20}"
GRPO_MAX_EXTRA_REROUTE_ROUNDS="${GRPO_MAX_EXTRA_REROUTE_ROUNDS:-20}"
# One worker uses the shared Host, sidecar, and search service.
MAX_CONCURRENCY_PER_WORKER="${MAX_CONCURRENCY_PER_WORKER:-3}"

ASAG_ENABLED="${ASAG_ENABLED:-1}"
ASAG_ATTENTION_DTYPE="${ASAG_ATTENTION_DTYPE:-bfloat16}"
ASAG_ANSWER_PROBE_MAX_TOKENS="${ASAG_ANSWER_PROBE_MAX_TOKENS:-32}"
ASAG_CONFIDENCE_THRESHOLD="${ASAG_CONFIDENCE_THRESHOLD:-0.95}"
ASAG_ENTROPY_DELTA_THRESHOLD="${ASAG_ENTROPY_DELTA_THRESHOLD:--0.10}"
ASAG_MAX_REROUTES="${ASAG_MAX_REROUTES:-2}"
ASAG_MAX_VERIFICATIONS="${ASAG_MAX_VERIFICATIONS:-1}"
ASAG_DECODING_WINDOW_TOKENS="${ASAG_DECODING_WINDOW_TOKENS:-32}"
ASAG_MAX_PROBE_TOKENS="${ASAG_MAX_PROBE_TOKENS:-64}"
ASAG_MAX_CACHED_SESSIONS="${ASAG_MAX_CACHED_SESSIONS:-3}"
ASAG_KV_OFFLOAD="${ASAG_KV_OFFLOAD:-0}"
ASAG_SELECTIVE_ATTENTION="${ASAG_SELECTIVE_ATTENTION:-1}"
ASAG_MONITORED_LAYERS="${ASAG_MONITORED_LAYERS:-4}"

OPENRESEARCHER_MAX_INPUT_TOKENS="${OPENRESEARCHER_MAX_INPUT_TOKENS:-23000}"
OPENRESEARCHER_TOOL_RESPONSE_TOKENS="${OPENRESEARCHER_TOOL_RESPONSE_TOKENS:-2048}"
OPENRESEARCHER_CONTEXT_REBUILD_TOKENS="${OPENRESEARCHER_CONTEXT_REBUILD_TOKENS:-16000}"

DATA_PATH="${DATA_PATH:-$PROJECT_ROOT/../BrowseCore/ourdata/data/test-*.parquet}"
CORPUS_PARQUET_PATH="${CORPUS_PARQUET_PATH:-$PROJECT_ROOT/../BrowseCore/corpus/data/*.parquet}"
DENSE_INDEX_PATH="${DENSE_INDEX_PATH:-$PROJECT_ROOT/../BrowseCore/training-indexes/qwen3-embedding-8b/*.pkl}"
DENSE_MODEL_NAME="${DENSE_MODEL_NAME:-Qwen/Qwen3-Embedding-8B}"
BROWSER_BACKEND="${BROWSER_BACKEND:-local}"
REASONING_EFFORT="${REASONING_EFFORT:-high}"
QID_FILE="${QID_FILE:-}"
CONTINUATION_SOURCE="${CONTINUATION_SOURCE:-}"
FRESH_RUN="${FRESH_RUN:-1}"

OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/results/ourdata/LiteResearcher4B_ASAG_06B_hint_E2E_GRPO_step70_alpha20_host_rewrite_reroute2_thr025}"
RUN_NAME="${RUN_NAME:-LiteResearcher4B_ASAG_06B_hint_E2E_GRPO_step70_alpha20_host_rewrite_reroute2_thr025}"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/logs/run_code/$RUN_NAME}"

PLUGIN_URLS="http://127.0.0.1:$PLUGIN_PORT_1"
ASAG_ATTENTION_URLS="$PLUGIN_URLS"
SERVER_URLS="http://127.0.0.1:$VLLM_PORT_1/v1"
SEARCH_URLS="http://127.0.0.1:$SEARCH_PORT_1"

export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/vendor/tevatron/src${PYTHONPATH:+:$PYTHONPATH}"
export LUCENE_EXTRA_DIR="${LUCENE_EXTRA_DIR:-$PROJECT_ROOT/vendor/tevatron}"
export CORPUS_PARQUET_PATH DENSE_INDEX_PATH DENSE_MODEL_NAME
export ASAG_ENABLED ASAG_ANSWER_PROBE_MAX_TOKENS
export ASAG_CONFIDENCE_THRESHOLD ASAG_ENTROPY_DELTA_THRESHOLD
export ASAG_MAX_REROUTES ASAG_MAX_VERIFICATIONS
export ASAG_DECODING_WINDOW_TOKENS ASAG_MAX_PROBE_TOKENS
export ASAG_MAX_CACHED_SESSIONS ASAG_KV_OFFLOAD
export ASAG_SELECTIVE_ATTENTION ASAG_MONITORED_LAYERS
export JOINT_CONTROLLER_ENABLED JOINT_CONTROLLER_REROUTE_THRESHOLD
export JOINT_CONTROLLER_MAX_REROUTES
export OPENRESEARCHER_MAX_INPUT_TOKENS OPENRESEARCHER_TOOL_RESPONSE_TOKENS
export OPENRESEARCHER_CONTEXT_REBUILD_TOKENS
export GRPO_REROUTE_SEARCH_ATTEMPTS GRPO_MIN_POST_REROUTE_ROUNDS
export GRPO_MAX_EXTRA_REROUTE_ROUNDS
export GRPO_HOST_QUERY_TEMPERATURE GRPO_HOST_QUERY_MAX_NEW_TOKENS
export GRPO_HOST_QUERY_MAX_WORDS

if [ "$ASAG_ENABLED" = "0" ]; then
    echo "This hybrid launcher requires ASAG_ENABLED=1." >&2
    exit 1
fi
if [ "$FRESH_RUN" != "0" ] && [ "$FRESH_RUN" != "1" ]; then
    echo "FRESH_RUN must be 0 or 1." >&2
    exit 1
fi
if [ "$JOINT_CONTROLLER_ENABLED" != "0" ] && [ "$JOINT_CONTROLLER_ENABLED" != "1" ]; then
    echo "JOINT_CONTROLLER_ENABLED must be 0 or 1." >&2
    exit 1
fi
if ! [[ "$JOINT_CONTROLLER_MAX_REROUTES" =~ ^[0-9]+$ ]]; then
    echo "JOINT_CONTROLLER_MAX_REROUTES must be non-negative." >&2
    exit 1
fi
for value_name in GRPO_REROUTE_SEARCH_ATTEMPTS GRPO_MIN_POST_REROUTE_ROUNDS GRPO_MAX_EXTRA_REROUTE_ROUNDS; do
    if ! [[ "${!value_name}" =~ ^[0-9]+$ ]]; then
        echo "$value_name must be a non-negative integer." >&2
        exit 1
    fi
done
if [ "$GRPO_REROUTE_SEARCH_ATTEMPTS" -lt 1 ]; then
    echo "GRPO_REROUTE_SEARCH_ATTEMPTS must be at least 1." >&2
    exit 1
fi
if [ ! -x "$PYTHON_BIN" ]; then
    echo "Python is not executable: $PYTHON_BIN" >&2
    exit 1
fi
for required in \
    "$PROMPT_FILE" \
    "$PLUGIN_GRPO_SOURCE_ROOT/verl/experimental/plugin_grpo/policy.py" \
    "$PLUGIN_CHECKPOINT/lora_adapter" \
    "$PLUGIN_CHECKPOINT/plugin_heads.pt" \
    "$PLUGIN_CHECKPOINT/metadata.json"; do
    if [ ! -e "$required" ]; then
        echo "Required path not found: $required" >&2
        exit 1
    fi
done
if [ -n "$QID_FILE" ] && [ ! -f "$QID_FILE" ]; then
    echo "QID file not found: $QID_FILE" >&2
    exit 1
fi
if ! compgen -G "$DATA_PATH" >/dev/null \
    || ! compgen -G "$CORPUS_PARQUET_PATH" >/dev/null \
    || ! compgen -G "$DENSE_INDEX_PATH" >/dev/null; then
    echo "Dataset, corpus, or dense index path is missing." >&2
    exit 1
fi

for gpu in "$SEARCH_GPU" "$VLLM_GPU_1"; do
    if ! [[ "$gpu" =~ ^[0-9]+$ ]]; then
        echo "Invalid GPU ID: $gpu" >&2
        exit 1
    fi
done
if [ "$VLLM_GPU_1" = "$SEARCH_GPU" ]; then
    echo "Host/sidecar GPU must differ from the Search GPU." >&2
    exit 1
fi

ports=(
    "$PLUGIN_PORT_1" "$VLLM_PORT_1" "$SEARCH_PORT_1"
)
for port in "${ports[@]}"; do
    if ! [[ "$port" =~ ^[0-9]+$ ]] || [ "$port" -lt 1 ] || [ "$port" -gt 65535 ]; then
        echo "Invalid service port: $port" >&2
        exit 1
    fi
done
if [ "$(printf '%s\n' "${ports[@]}" | sort -u | wc -l)" -ne "${#ports[@]}" ]; then
    echo "All vLLM, sidecar, and Search ports must differ." >&2
    exit 1
fi
if ! [[ "$MAX_CONCURRENCY_PER_WORKER" =~ ^[1-9][0-9]*$ ]]; then
    echo "MAX_CONCURRENCY_PER_WORKER must be positive." >&2
    exit 1
fi
total_active_questions=$MAX_CONCURRENCY_PER_WORKER
if [ "$ASAG_KV_OFFLOAD" = "0" ] && [ "$ASAG_MAX_CACHED_SESSIONS" -lt "$MAX_CONCURRENCY_PER_WORKER" ]; then
    echo "ASAG_MAX_CACHED_SESSIONS=$ASAG_MAX_CACHED_SESSIONS is below the $MAX_CONCURRENCY_PER_WORKER questions routed to each sidecar." >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
if [ "$FRESH_RUN" = "1" ] && compgen -G "$OUTPUT_DIR/node_*_shard_*.jsonl" >/dev/null; then
    echo "Fresh run requested, but result shards already exist in $OUTPUT_DIR." >&2
    echo "Choose a new OUTPUT_DIR or set FRESH_RUN=0 to resume successes." >&2
    exit 1
fi
exec 9>"$LOG_DIR/run.lock"
if ! flock -n 9; then
    echo "Another run is already using $LOG_DIR." >&2
    exit 1
fi
if ! flock -n "$OUTPUT_DIR/.deploy_agent.lock" -c true; then
    echo "Another process is writing to $OUTPUT_DIR." >&2
    exit 1
fi

port_is_listening() {
    ss -H -ltn "sport = :$1" | grep -q .
}

wait_for_endpoint() {
    local name=$1
    local url=$2
    local process_id=$3
    local log_file=$4
    local started_at
    started_at=$(date +%s)
    echo "Waiting for $name: $url"
    while true; do
        if curl -fsS --max-time 10 "$url" >/dev/null 2>&1; then
            echo "$name is ready."
            return 0
        fi
        if ! kill -0 "$process_id" 2>/dev/null; then
            echo "$name exited before becoming ready. Last log lines:" >&2
            tail -n 60 "$log_file" >&2 || true
            return 1
        fi
        if [ $(( $(date +%s) - started_at )) -ge "$STARTUP_TIMEOUT" ]; then
            echo "Timed out after $STARTUP_TIMEOUT seconds waiting for $name." >&2
            tail -n 60 "$log_file" >&2 || true
            return 1
        fi
        sleep 10
    done
}

for port in "${ports[@]}"; do
    if port_is_listening "$port"; then
        echo "Port $port is occupied; refusing to replace the existing service." >&2
        exit 1
    fi
done

SERVICE_PIDS=()
cleanup() {
    local exit_code=$?
    trap - EXIT INT TERM
    if [ "${#SERVICE_PIDS[@]}" -gt 0 ]; then
        echo "Stopping services started by this script..."
    fi
    for process_id in "${SERVICE_PIDS[@]}"; do
        if kill -0 "$process_id" 2>/dev/null; then
            kill -TERM -- "-$process_id" 2>/dev/null || true
        fi
    done
    for process_id in "${SERVICE_PIDS[@]}"; do
        wait "$process_id" 2>/dev/null || true
    done
    exit "$exit_code"
}
trap cleanup EXIT INT TERM

for port in "$SEARCH_PORT_1"; do
    search_log="$LOG_DIR/search_gpu${SEARCH_GPU}_port${port}.log"
    echo "Starting dense search on GPU $SEARCH_GPU, port $port..."
    setsid bash scripts/start_search_service.sh \
        dense "$port" "$SEARCH_GPU" >"$search_log" 2>&1 &
    process_id=$!
    SERVICE_PIDS+=("$process_id")
    wait_for_endpoint \
        "dense search on GPU $SEARCH_GPU" \
        "http://127.0.0.1:$port/" \
        "$process_id" \
        "$search_log"
done

for pair in "$VLLM_GPU_1:$VLLM_PORT_1"; do
    gpu="${pair%%:*}"
    port="${pair##*:}"
    vllm_log="$LOG_DIR/vllm_gpu${gpu}_port${port}.log"
    echo "Starting TP=1 LiteResearcher Host on GPU $gpu, port $port..."
    setsid env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$gpu" \
        "$PYTHON_BIN" scripts/deploy_vllm_service.py \
        --model "$MODEL" \
        --port "$port" \
        --tensor_parallel_size 1 \
        --gpu_memory_utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
        --max_model_len "$VLLM_MAX_MODEL_LEN" \
        >"$vllm_log" 2>&1 &
    process_id=$!
    SERVICE_PIDS+=("$process_id")
    wait_for_endpoint \
        "LiteResearcher Host on GPU $gpu" \
        "http://127.0.0.1:$port/v1/models" \
        "$process_id" \
        "$vllm_log"
done

for pair in "$VLLM_GPU_1:$PLUGIN_PORT_1"; do
    gpu="${pair%%:*}"
    port="${pair##*:}"
    plugin_log="$LOG_DIR/plugin_gpu${gpu}_port${port}.log"
    echo "Starting GRPO logits sidecar on GPU $gpu, port $port..."
    setsid env \
        CUDA_DEVICE_ORDER=PCI_BUS_ID \
        CUDA_VISIBLE_DEVICES="$gpu" \
        PLUGIN_GRPO_SOURCE_ROOT="$PLUGIN_GRPO_SOURCE_ROOT" \
        "$PYTHON_BIN" grpo_plugin_service.py \
        --checkpoint "$PLUGIN_CHECKPOINT" \
        --host-model "$MODEL" \
        --device cuda:0 \
        --port "$port" \
        --alpha "$PLUGIN_ALPHA" \
        --reroute-seed "$PLUGIN_REROUTE_SEED" \
        --reroute-format-attempts "$PLUGIN_REROUTE_FORMAT_ATTEMPTS" \
        --controller-temperature "$PLUGIN_CONTROLLER_TEMPERATURE" \
        --hint-slot-token-budgets "$PLUGIN_HINT_SLOT_BUDGETS" \
        --hint-temperature "$PLUGIN_HINT_TEMPERATURE" \
        --hint-top-p "$PLUGIN_HINT_TOP_P" \
        --hint-correction-topk "$PLUGIN_HINT_CORRECTION_TOPK" \
        --hint-query-max-words "$PLUGIN_HINT_QUERY_MAX_WORDS" \
        --max-plugin-length "$PLUGIN_MAX_LENGTH" \
        --min-reroute-rounds "$JOINT_CONTROLLER_MIN_REROUTE_ROUNDS" \
        --reroute-cooldown-rounds "$JOINT_CONTROLLER_REROUTE_COOLDOWN_ROUNDS" \
        --asag-attention-dtype "$ASAG_ATTENTION_DTYPE" \
        >"$plugin_log" 2>&1 &
    process_id=$!
    SERVICE_PIDS+=("$process_id")
    wait_for_endpoint \
        "GRPO logits sidecar on GPU $gpu" \
        "http://127.0.0.1:$port/health" \
        "$process_id" \
        "$plugin_log"
done

echo "================================================="
echo "LiteResearcher + ASAG + GRPO corrected logits"
echo "================================================="
echo "Host GPU/port: $VLLM_GPU_1:$VLLM_PORT_1"
echo "GRPO/ASAG sidecar GPU/port: $VLLM_GPU_1:$PLUGIN_PORT_1"
echo "Search GPU/port: $SEARCH_GPU:$SEARCH_PORT_1"
echo "vLLM URLs: $SERVER_URLS"
echo "Plugin URLs: $PLUGIN_URLS"
echo "ASAG attention URLs: $ASAG_ATTENTION_URLS"
echo "Search URLs: $SEARCH_URLS"
echo "Plugin checkpoint: $PLUGIN_CHECKPOINT"
echo "Plugin alpha: $PLUGIN_ALPHA"
echo "Reroute sampling seed: $PLUGIN_REROUTE_SEED"
echo "Reroute format attempts: $PLUGIN_REROUTE_FORMAT_ATTEMPTS"
echo "Hint slot token budgets: $PLUGIN_HINT_SLOT_BUDGETS"
echo "Hint temperature/top-p: $PLUGIN_HINT_TEMPERATURE/$PLUGIN_HINT_TOP_P"
echo "Hint correction Top-K: $PLUGIN_HINT_CORRECTION_TOPK"
echo "Hint query max words: $PLUGIN_HINT_QUERY_MAX_WORDS"
echo "Host rewrite temperature/max tokens/max words: $GRPO_HOST_QUERY_TEMPERATURE/$GRPO_HOST_QUERY_MAX_NEW_TOKENS/$GRPO_HOST_QUERY_MAX_WORDS"
echo "Control: step-70 corrected logits -> three-field hint -> Host English query rewrite -> browser.search"
echo "Joint controller enabled: $JOINT_CONTROLLER_ENABLED"
echo "Joint controller reroute threshold: $JOINT_CONTROLLER_REROUTE_THRESHOLD"
echo "Reroute search attempts: $GRPO_REROUTE_SEARCH_ATTEMPTS"
echo "Minimum post-reroute rounds / maximum extra rounds: $GRPO_MIN_POST_REROUTE_ROUNDS/$GRPO_MAX_EXTRA_REROUTE_ROUNDS"
echo "Max rounds: $MAX_ROUNDS"
echo "Concurrency per worker / total: $MAX_CONCURRENCY_PER_WORKER/$total_active_questions"
echo "ASAG cached sessions per sidecar: $ASAG_MAX_CACHED_SESSIONS"
echo "Run mode: $([ "$FRESH_RUN" = "1" ] && echo fresh-all-QIDs || echo resume-success-only)"
echo "Output: $OUTPUT_DIR"
echo "Logs: $LOG_DIR"
echo "================================================="

DEPLOY_EXTRA_ARGS=()
if [ -n "$QID_FILE" ]; then
    DEPLOY_EXTRA_ARGS+=(--qid_file "$QID_FILE")
fi
if [ -n "$CONTINUATION_SOURCE" ]; then
    if [ ! -e "$CONTINUATION_SOURCE" ]; then
        echo "Continuation source not found: $CONTINUATION_SOURCE" >&2
        exit 1
    fi
    DEPLOY_EXTRA_ARGS+=(--continuation_source "$CONTINUATION_SOURCE")
fi
if [ "$FRESH_RUN" = "1" ]; then
    DEPLOY_EXTRA_ARGS+=(--fresh_run)
else
    DEPLOY_EXTRA_ARGS+=(--resume_success_only)
fi

/usr/bin/time -v "$PYTHON_BIN" deploy_agent.py \
    --output_dir "$OUTPUT_DIR" \
    --model_name_or_path "$MODEL" \
    --system_prompt_file "$PROMPT_FILE" \
    --search_url "$SEARCH_URLS" \
    --asag_attention_url "$ASAG_ATTENTION_URLS" \
    --grpo_plugin_url "$PLUGIN_URLS" \
    --dataset_name browsecomp-plus \
    --data_path "$DATA_PATH" \
    --browser_backend "$BROWSER_BACKEND" \
    --reasoning_effort "$REASONING_EFFORT" \
    --vllm_server_url "$SERVER_URLS" \
    "${DEPLOY_EXTRA_ARGS[@]}" \
    --max_concurrency_per_worker "$MAX_CONCURRENCY_PER_WORKER" \
    --max_rounds "$MAX_ROUNDS"

echo "ASAG + GRPO logits run completed: $OUTPUT_DIR"
