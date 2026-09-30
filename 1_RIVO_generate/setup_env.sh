#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE:-python3.12}"
if ! command -v "$PYTHON_EXECUTABLE" >/dev/null 2>&1; then
    echo "Python executable not found: $PYTHON_EXECUTABLE" >&2
    exit 1
fi

if [ ! -d .venv ]; then
    "$PYTHON_EXECUTABLE" -m venv .venv
fi

.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -e vendor/tevatron

if ! command -v java >/dev/null 2>&1 && [ -z "${JAVA_HOME:-}" ]; then
    echo "Warning: Java was not found. Install OpenJDK 21 or set JAVA_HOME before running."
fi

echo "Environment ready: $ROOT/.venv"
