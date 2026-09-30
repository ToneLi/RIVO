#!/usr/bin/env bash
# Start the retrieval service used by run_deepresearch_plugin_grpo.sh.
#
# Defaults:
#   corpus/index: BrowseComp-Plus training corpus and dense index
#   URL:          http://127.0.0.1:8001
#   GPU:          physical GPU 0
#
# Usage:
#   bash start_deepresearch_search_service.sh
#   bash start_deepresearch_search_service.sh [dense|bm25] [port] [gpu]
#
# Environment overrides:
#   SEARCHER_TYPE=dense SEARCH_PORT=8001 SEARCH_GPU=0 \
#     bash start_deepresearch_search_service.sh

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEARCH_HOST_DIR="${SEARCH_HOST_DIR:-$(dirname "$PROJECT_DIR")/search_host}"
BACKEND_LAUNCHER="$SEARCH_HOST_DIR/start_training_search_service.sh"

SEARCHER_TYPE="${1:-${SEARCHER_TYPE:-dense}}"
SEARCH_PORT="${2:-${SEARCH_PORT:-8001}}"
SEARCH_GPU="${3:-${SEARCH_GPU:-0}}"
SEARCH_URL="http://127.0.0.1:${SEARCH_PORT}"

if [[ "$SEARCHER_TYPE" != "dense" && "$SEARCHER_TYPE" != "bm25" ]]; then
    echo "Error: searcher type must be 'dense' or 'bm25', got: $SEARCHER_TYPE" >&2
    exit 2
fi
if [[ ! "$SEARCH_PORT" =~ ^[0-9]+$ ]] || (( SEARCH_PORT < 1 || SEARCH_PORT > 65535 )); then
    echo "Error: invalid port: $SEARCH_PORT" >&2
    exit 2
fi
if [ ! -f "$BACKEND_LAUNCHER" ]; then
    echo "Error: retrieval backend launcher not found: $BACKEND_LAUNCHER" >&2
    exit 1
fi

# Loading Qwen3-Embedding-8B twice wastes substantial GPU memory. Treat an
# already healthy service on the requested port as success.
if curl -fsS --max-time 5 "$SEARCH_URL/" >/dev/null 2>&1; then
    echo "Retrieval service is already ready at $SEARCH_URL"
    exit 0
fi

echo "Starting DeepResearch retrieval service"
echo "  type: $SEARCHER_TYPE"
echo "  URL:  $SEARCH_URL"
echo "  GPU:  $SEARCH_GPU"
echo "  data: BrowseComp-Plus training corpus/indexes"
echo
echo "Keep this process running while run_deepresearch_plugin_grpo.sh is running."

export OPENRESEARCHER_ROOT="${OPENRESEARCHER_ROOT:-}"
export OPD_ENV="${OPD_ENV:-}"

exec bash "$BACKEND_LAUNCHER" "$SEARCHER_TYPE" "$SEARCH_PORT" "$SEARCH_GPU"
