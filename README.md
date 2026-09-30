# BrowseCore, Plugin GRPO, and RIVO

This directory contains the framework diagram, the BrowseCore retrieval data, the Plugin GRPO training code, and the RIVO inference and evaluation code.

## Main Framework

![Main framework](picture/main_frame.png)

The framework combines a frozen DeepResearch Host with a trainable 0.6B Plugin. A state-guided controller uses confidence, evidence attention, and reasoning convergence signals to decide whether to continue, stop, or reroute. When rerouting is needed, the Plugin produces a structured hint and a low-rank logit correction. The Host uses that information to rewrite the search query and searches the local corpus through BrowseCore. GRPO optimizes the Plugin using task rewards.

```text
Question → Host reasoning and search → State decision → Continue / Stop / Reroute
                                                ↓ Reroute
                              0.6B Plugin hint and logit correction
                                                ↓
                              Host rewrites query → BrowseCore
```

## Plugin GRPO Training

Run the commands below from the `github_load/` directory. The training scripts require a Search service at `http://127.0.0.1:8001` before training starts. Use two terminals.

**Terminal 1: Start the Search service**

```bash
cd 0_GRPO_plug_end_to_end
bash start_deepresearch_search_service.sh dense 8001 0
```

This launcher calls `../search_host/start_training_search_service.sh`. The `search_host/` directory is not included in the current `github_load` checkout. Add that backend, set `SEARCH_HOST_DIR` to its location, or start a Search service separately and keep it available on port 8001.

**Terminal 2: Start GRPO training**

```bash
cd 0_GRPO_plug_end_to_end
export DEEPRESEARCH_SEARCH_URL=http://127.0.0.1:8001
export PYTHON_BIN=/path/to/your/env/bin/python
export CLASSIFIER_ADAPTER="$PWD/intial"
export OPENAI_ENV_FILE=/path/to/OpenResearcher/.env
export DEEPRESEARCH_EVAL_PATH=/path/to/OpenResearcher/eval.py
export RESUME_CHECKPOINT=/path/to/checkpoints/global_step_50
bash run_deepresearch_plugin_grpo.sh
```

`run_deepresearch_plugin_grpo.sh` checks the Search service, starts the Plugin service, and then starts VERL/GRPO training. To run the right-hint variant, use `run_deepresearch_plugin_grpo_right_hint.sh` and set its `RESUME_CHECKPOINT` and `PLUGIN_OUTPUT_DIR` as needed. Configure the Python and OpenResearcher paths for your environment. The OpenAI environment file must provide `OPENAI_API_KEY`.

## RIVO Inference

`1_RIVO_generate/run_LiteResearcher4B.sh` starts dense Search, the LiteResearcher Host, and the GRPO/ASAG Plugin service before running inference. It reads data, index, model, and GPU settings from `config.env`. Edit that file to point to the current checkout, for example:

```bash
cd 1_RIVO_generate
# Set local paths in config.env:
# PLUGIN_GRPO_SOURCE_ROOT="$PROJECT_ROOT/../0_GRPO_plug_end_to_end"
# PLUGIN_CHECKPOINT="$PROJECT_ROOT/Plugin"
# DATA_PATH="$PROJECT_ROOT/../BrowseCore/ourdata/data/test-*.parquet"
# CORPUS_PARQUET_PATH="$PROJECT_ROOT/../BrowseCore/corpus/data/*.parquet"
# DENSE_INDEX_PATH="$PROJECT_ROOT/../BrowseCore/training-indexes/qwen3-embedding-8b/*.pkl"
# Also set PYTHON_BIN, the model path, and GPU IDs
bash run_LiteResearcher4B.sh
```

Inference outputs and run logs are written to `1_RIVO_generate/results/` and `1_RIVO_generate/logs/` by default.

### Validate and Run Evaluation

Use verify_eval.sh to check an inference output directory against the JSONL fields consumed by eval.py. The preflight checks Python dependencies, parses eval.py, validates the JSONL records, and reports duplicate QIDs (the evaluator keeps the latest attempt).

~~~bash
cd 1_RIVO_generate
bash verify_eval.sh results/ourdata/my_run
~~~

The command above only validates inputs. To run the evaluator after validation, add --run:

~~~bash
bash verify_eval.sh --run results/ourdata/my_run
~~~

Running with --run sends successful records to the OpenAI judge and writes evaluated.jsonl and any generated plots into the input directory. Set OPENAI_API_KEY in the environment or in an untracked 1_RIVO_generate/.env file. Set PYTHON_BIN if the evaluator dependencies are installed in a specific Python environment.

## Downloading Model and Data Files

Binary model and data assets are stored with Git LFS. Install Git LFS before cloning this repository. After cloning, run the following commands from the repository root to download the actual assets:

~~~bash
git lfs install
git lfs pull
~~~
