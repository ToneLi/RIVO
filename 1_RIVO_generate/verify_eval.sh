#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_EVAL=0
INPUT_DIR=""

usage() {
    cat <<'USAGE'
Usage:
  bash verify_eval.sh [--run] <input_dir>

Checks the input files expected by eval.py. Add --run to invoke eval.py
after validation; that makes OpenAI API calls and writes evaluation outputs.
USAGE
}

while (($#)); do
    case "$1" in
        --run) RUN_EVAL=1 ;;
        -h|--help) usage; exit 0 ;;
        --*) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
        *)
            if [[ -n "$INPUT_DIR" ]]; then
                echo "Only one input directory may be provided." >&2
                usage >&2
                exit 2
            fi
            INPUT_DIR="$1"
            ;;
    esac
    shift
done

if [[ -z "$INPUT_DIR" ]]; then
    echo "Missing evaluation input directory." >&2
    usage >&2
    exit 2
fi
if [[ ! -d "$INPUT_DIR" ]]; then
    echo "Input directory does not exist: $INPUT_DIR" >&2
    exit 1
fi
INPUT_DIR="$(cd "$INPUT_DIR" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Python executable not found: $PYTHON_BIN" >&2
    echo "Set PYTHON_BIN to the Python environment used by eval.py." >&2
    exit 1
fi

cd "$SCRIPT_DIR"
"$PYTHON_BIN" - "$INPUT_DIR" "$SCRIPT_DIR/eval.py" <<'PYTHON'
import ast
import glob
import importlib.util
import json
import os
import sys

input_dir, eval_path = sys.argv[1:3]
missing = [
    name for name in ("openai", "tqdm", "prettytable", "dotenv")
    if importlib.util.find_spec(name) is None
]
if missing:
    print("MISSING Python packages: " + ", ".join(missing), file=sys.stderr)
    sys.exit(1)

with open(eval_path, encoding="utf-8") as f:
    ast.parse(f.read(), filename=eval_path)
print("OK: eval.py parses and required Python packages are available")

files = sorted(
    f for f in glob.glob(os.path.join(input_dir, "*.jsonl"))
    if not f.endswith("evaluated.jsonl")
)
if not files:
    print(f"No input .jsonl files found in: {input_dir}", file=sys.stderr)
    sys.exit(1)

latest_by_qid = {}
duplicates = 0
successes = 0
errors = []
for path in files:
    with open(path, encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"{os.path.basename(path)}:{line_number}: invalid JSON ({exc.msg})")
                continue
            if not isinstance(record, dict):
                errors.append(f"{os.path.basename(path)}:{line_number}: record must be a JSON object")
                continue
            if "qid" not in record:
                errors.append(f"{os.path.basename(path)}:{line_number}: missing qid")
                continue
            if record.get("status") == "success":
                messages = record.get("messages")
                if not isinstance(record.get("question"), str):
                    errors.append(f"{os.path.basename(path)}:{line_number}: success record needs string question")
                if "answer" not in record:
                    errors.append(f"{os.path.basename(path)}:{line_number}: success record needs answer")
                if not isinstance(messages, list) or not messages or not isinstance(messages[-1], dict) or "content" not in messages[-1]:
                    errors.append(f"{os.path.basename(path)}:{line_number}: success record needs messages[-1].content")
                else:
                    successes += 1
            qid = str(record["qid"])
            if qid in latest_by_qid:
                duplicates += 1
            latest_by_qid[qid] = record

if errors:
    print("Input validation failed:", file=sys.stderr)
    for error in errors[:20]:
        print(f"  - {error}", file=sys.stderr)
    if len(errors) > 20:
        print(f"  ... and {len(errors) - 20} more error(s)", file=sys.stderr)
    sys.exit(1)

print(f"OK: {len(files)} JSONL file(s), {len(latest_by_qid)} unique QID(s), {successes} successful record(s)")
if duplicates:
    print(f"NOTE: {duplicates} duplicate QID record(s); eval.py will keep the latest attempt")
if successes == 0:
    print("NOTE: no successful records; eval.py will not send samples to the LLM judge")
PYTHON

if ((RUN_EVAL)); then
    if ! "$PYTHON_BIN" - "$SCRIPT_DIR/.env" <<'PYTHON'
import os
import sys
from dotenv import dotenv_values
env_file = sys.argv[1]
key = os.environ.get("OPENAI_API_KEY") or dotenv_values(env_file).get("OPENAI_API_KEY")
sys.exit(0 if key else 1)
PYTHON
    then
        echo "OPENAI_API_KEY is missing from the environment and $SCRIPT_DIR/.env" >&2
        exit 1
    fi
    echo "Running eval.py (this sends successful records to the configured OpenAI model)..."
    "$PYTHON_BIN" "$SCRIPT_DIR/eval.py" --input_dir "$INPUT_DIR"
else
    echo "Preflight complete. Pass --run to invoke eval.py."
fi
