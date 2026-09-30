#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$ROOT"
cd "$ROOT"

CONFIG_FILE="${CONFIG_FILE:-$ROOT/config.env}"
if [ -f "$CONFIG_FILE" ]; then
    # shellcheck disable=SC1090
    set -a
    source "$CONFIG_FILE"
    set +a
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_PATH="${DATA_PATH:-$PROJECT_ROOT/../BrowseCore/ourdata/data/test-*.parquet}"
CORPUS_PARQUET_PATH="${CORPUS_PARQUET_PATH:-$PROJECT_ROOT/../BrowseCore/corpus/data/*.parquet}"
DENSE_INDEX_PATH="${DENSE_INDEX_PATH:-$PROJECT_ROOT/../BrowseCore/training-indexes/qwen3-embedding-8b/*.pkl}"
PLUGIN_GRPO_SOURCE_ROOT="${PLUGIN_GRPO_SOURCE_ROOT:-$PROJECT_ROOT/../0_GRPO_plug_end_to_end}"
PLUGIN_CHECKPOINT="${PLUGIN_CHECKPOINT:-$PLUGIN_GRPO_SOURCE_ROOT/verl_checkpoints_deepresearch_hint_e2e_grpo_not_good/global_step_70/plugin}"

failed=0
check_glob() {
    local label=$1
    local pattern=$2
    if [ -n "$pattern" ] && compgen -G "$pattern" >/dev/null; then
        echo "OK: $label -> $pattern"
    else
        echo "MISSING: $label -> ${pattern:-<not configured>}" >&2
        failed=1
    fi
}

check_file() {
    local label=$1
    local path=$2
    if [ -f "$path" ]; then
        echo "OK: $label -> $path"
    else
        echo "MISSING: $label -> $path" >&2
        failed=1
    fi
}

check_contains() {
    local label=$1
    local path=$2
    local pattern=$3
    if grep -Fq -- "$pattern" "$path"; then
        echo "OK: $label"
    else
        echo "MISSING: $label ($pattern in $path)" >&2
        failed=1
    fi
}

check_file "agent" "$ROOT/deploy_agent.py"
check_file "algorithm/formula document" "$ROOT/ALGORITHM_ASAG_GRPO_LOGITS.md"
check_file "ASAG controller" "$ROOT/asag_controller.py"
check_file "ASAG attention implementation" "$ROOT/asag_attention.py"
check_file "ASAG attention service" "$ROOT/scripts/deploy_asag_attention_service.py"
check_file "GRPO sidecar" "$ROOT/grpo_plugin_service.py"
check_file "GRPO client" "$ROOT/utils/grpo_plugin_client.py"
check_file "hybrid launcher" "$ROOT/run_LiteResearcher4B_ASAG_GRPO_try2.sh"
check_contains "sidecar exposes ASAG analyze" "$ROOT/grpo_plugin_service.py" '@app.post("/analyze")'
check_contains "sidecar exposes ASAG session cleanup" "$ROOT/grpo_plugin_service.py" '@app.delete("/sessions/{session_id:path}")'
check_contains "hybrid reuses sidecar URLs for ASAG" "$ROOT/run_LiteResearcher4B_ASAG_GRPO_try2.sh" 'ASAG_ATTENTION_URLS="$PLUGIN_URLS"'
if grep -Fq -- 'scripts/deploy_asag_attention_service.py' "$ROOT/run_LiteResearcher4B_ASAG_GRPO_try2.sh"; then
    echo "UNEXPECTED: hybrid launcher still starts a standalone ASAG model" >&2
    failed=1
else
    echo "OK: hybrid launcher has no standalone ASAG model"
fi
check_file "GRPO policy source" "$PLUGIN_GRPO_SOURCE_ROOT/verl/experimental/plugin_grpo/policy.py"
check_file "GRPO hint policy source" "$PLUGIN_GRPO_SOURCE_ROOT/verl/experimental/plugin_grpo/hint_policy.py"
check_file "GRPO hint prompts" "$PLUGIN_GRPO_SOURCE_ROOT/verl/experimental/plugin_grpo/hint_prompts.py"
check_file "GRPO plugin heads" "$PLUGIN_CHECKPOINT/plugin_heads.pt"
check_file "GRPO checkpoint metadata" "$PLUGIN_CHECKPOINT/metadata.json"
if [ -d "$PLUGIN_CHECKPOINT/lora_adapter" ]; then
    echo "OK: GRPO LoRA adapter -> $PLUGIN_CHECKPOINT/lora_adapter"
else
    echo "MISSING: GRPO LoRA adapter -> $PLUGIN_CHECKPOINT/lora_adapter" >&2
    failed=1
fi
check_file "prompt" "$ROOT/prompts/liter_research_prompt.py"
check_file "Lucene highlighter" "$ROOT/vendor/tevatron/lucene-highlighter-9.9.1.jar"
check_glob "test data" "$DATA_PATH"
check_glob "corpus" "$CORPUS_PARQUET_PATH"
check_glob "external dense index" "$DENSE_INDEX_PATH"

if find "$ROOT" -type f \( -name "*.parquet" -o -name "*.pkl" \) -print -quit | grep -q .; then
    echo "UNEXPECTED: data files found inside bundle" >&2
    failed=1
else
    echo "OK: no parquet or dense-index pkl data bundled"
fi

if [ -x "$PYTHON_BIN" ]; then
    PYTHONPATH="$ROOT:$ROOT/vendor/tevatron/src${PYTHONPATH:+:$PYTHONPATH}" \
        "$PYTHON_BIN" -c 'import duckdb, faiss, fastapi, torch, transformers, vllm, tevatron; print("OK: Python imports")'
else
    echo "MISSING: executable Python -> $PYTHON_BIN" >&2
    failed=1
fi

exit "$failed"
