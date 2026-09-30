#!/usr/bin/env bash

# Start search service with proper environment setup
# Usage: ./scripts/0_browcom_base_start_search_service.sh [bm25|dense] [port] [gpu]

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

export CUDA_VISIBLE_DEVICES="${3:-0}"
# Set GPU_IDS to 0 because CUDA_VISIBLE_DEVICES restricts the view to only the selected GPUs, 
# so the application sees them starting from index 0.
export GPU_IDS=0
if [ -n "${JAVA_HOME:-}" ]; then
    export PATH="${JAVA_HOME}/bin:${PATH}"
fi
# pyserini imports openai at module level even when not used; set a dummy key to bypass
export OPENAI_API_KEY=${OPENAI_API_KEY:-"dummy-not-used"}
# Parameters
SEARCHER_TYPE="${1:-dense}"
PORT="${2:-8000}"

echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}Starting Search Service${NC}"
echo -e "${GREEN}================================${NC}"
echo "Searcher Type: ${SEARCHER_TYPE}"
echo "Port: ${PORT}"
echo ""

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Use the same configurable Python environment as the main runner.
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
    echo -e "${RED}Error: Python is not executable: $PYTHON_BIN${NC}"
    echo "Run bash setup_env.sh or set PYTHON_BIN in config.env."
    exit 1
fi

# Verify the selected Python.
PYTHON_VERSION=$("$PYTHON_BIN" --version)
echo "Using: $PYTHON_VERSION"
echo "Python path: $PYTHON_BIN"
echo ""

# Set common environment variables
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/vendor/tevatron/src${PYTHONPATH:+:$PYTHONPATH}"
export LUCENE_EXTRA_DIR="${LUCENE_EXTRA_DIR:-$PROJECT_ROOT/vendor/tevatron}"
export CORPUS_PARQUET_PATH="${CORPUS_PARQUET_PATH:-$PROJECT_ROOT/../BrowseCore/corpus/data/*.parquet}"

echo "LUCENE_EXTRA_DIR: ${LUCENE_EXTRA_DIR}"
echo "CORPUS_PARQUET_PATH: ${CORPUS_PARQUET_PATH}"

# Check if Lucene JARs exist
if [ ! -f "${LUCENE_EXTRA_DIR}/lucene-highlighter-9.9.1.jar" ]; then
    echo -e "${RED}Error: Lucene JARs not found in ${LUCENE_EXTRA_DIR}${NC}"
    echo "Please run ./setup.sh to download them"
    exit 1
fi

# Check if corpus exists
if ! compgen -G "$CORPUS_PARQUET_PATH" >/dev/null; then
    echo -e "${RED}Error: Corpus not found${NC}"
    echo "Please run ./setup.sh to download the corpus"
    exit 1
fi

# Configure searcher-specific settings
if [ "$SEARCHER_TYPE" = "bm25" ]; then
    export LUCENE_INDEX_DIR="${PROJECT_ROOT}/Tevatron/browsecomp-plus-indexes/bm25"
    export SEARCHER_TYPE="bm25"

    if [ ! -d "$LUCENE_INDEX_DIR" ]; then
        echo -e "${RED}Error: BM25 index not found at $LUCENE_INDEX_DIR${NC}"
        echo "Please run ./setup.sh to download the index"
        exit 1
    fi

    echo "LUCENE_INDEX_DIR: ${LUCENE_INDEX_DIR}"
    echo "SEARCHER_TYPE: ${SEARCHER_TYPE}"

elif [ "$SEARCHER_TYPE" = "dense" ]; then
    export DENSE_INDEX_PATH="${DENSE_INDEX_PATH:-$PROJECT_ROOT/../BrowseCore/training-indexes/qwen3-embedding-8b/*.pkl}"
    export DENSE_MODEL_NAME="${DENSE_MODEL_NAME:-Qwen/Qwen3-Embedding-8B}"
    export SEARCHER_TYPE="dense"

    # Check if index files exist
    if [ -z "$DENSE_INDEX_PATH" ] || ! compgen -G "$DENSE_INDEX_PATH" >/dev/null; then
        echo -e "${RED}Error: Dense index not found${NC}"
        echo "Set DENSE_INDEX_PATH to the external qwen3-embedding-8b/*.pkl files."
        exit 1
    fi

    echo "DENSE_INDEX_PATH: ${DENSE_INDEX_PATH}"
    echo "DENSE_MODEL_NAME: ${DENSE_MODEL_NAME}"
    echo "SEARCHER_TYPE: ${SEARCHER_TYPE}"

else
    echo -e "${RED}Error: Invalid searcher type. Use 'bm25' or 'dense'${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}Starting uvicorn server...${NC}"
echo "Press Ctrl+C to stop"
echo ""

# Start uvicorn
exec "$PYTHON_BIN" -m uvicorn scripts.deploy_search_service:app --host 127.0.0.1 --port "$PORT"
