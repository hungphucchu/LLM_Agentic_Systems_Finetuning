# Assignment 3: Sequential Instruction Tuning Pipeline

**Course write-up (Option 1 — Markdown report):** **[REPORT.md](REPORT.md)** — main submission narrative, tables, and appendix pointers.

This repository implements a two-stage post-training pipeline for a small LLM:

1. Stage 1: Alpaca-style instruction tuning (QLoRA)
2. Stage 2: Teacher-generated JSON imitation tuning (QLoRA continuation)

The system evaluates three checkpoints (`ckpt0`, `ckpt1`, `ckpt2`) on both general instruction and structured JSON tasks, then performs forgetting analysis.

## Project Layout

- `config/`: model, training, and path configuration
- `prompts/`: editable teacher/judge prompt templates
- `src/data_prep/`: Alpaca prep + teacher JSON dataset construction
- `src/training/`: Stage 1 and Stage 2 QLoRA training
- `src/inference/`: checkpoint output generation
- `src/evaluation/`: automatic + judge-based evaluation and aggregation
- `hpc/`: slurm scripts for UTSA HPC execution
- `artifacts/`: `predictions/` (checkpoint generations), `judge/` (judge JSONL), `logs/` (Slurm), `tables/` (CSVs), `metrics/` (JSON), `archives/` (optional backups), `checkpoints/` (QLoRA weight directories — see below)

### What belongs in `artifacts/checkpoints/`

Training scripts **write here** relative to the repo root (same paths on HPC after `cd` into the project). **Checkpoint 0 (base)** is not saved locally — it is `microsoft/Phi-3.5-mini-instruct` from Hugging Face.

| Directory | Produced by | Role |
|-----------|-------------|------|
| `stage1_alpaca_adapter/` | `src/training/train_stage1_alpaca.py` | **ckpt1** — LoRA after Stage 1 (Alpaca) |
| `stage2_json_adapter/` | `src/training/train_stage2_json.py` (default `STAGE2_OUT_DIR`) | **ckpt2** — LoRA after Stage 2 (JSON), baseline run |
| `stage2_json_adapter_ep2_ablate/` | Stage 2 with `STAGE2_OUT_DIR=artifacts/checkpoints/stage2_json_adapter_ep2_ablate` (and e.g. `STAGE2_EPOCHS=2`) | Optional ablation adapter |

Each adapter directory should contain a **PEFT** export from `model.save_pretrained(...)` plus `tokenizer.save_pretrained(...)`, typically including **`adapter_config.json`**, **`adapter_model.safetensors`** (or `adapter_model.bin`), and tokenizer files (**`tokenizer.json`**, **`tokenizer_config.json`**, **`special_tokens_map.json`**, etc.). Hugging Face `Trainer` may also leave **`checkpoint-1/`**, **`checkpoint-2/`**, … subfolders (per-epoch saves) and **`trainer_state.json`** in the same output dir; you can keep or delete the intermediate `checkpoint-*` folders after a successful final save — inference uses the top-level adapter files.

To reproduce eval locally, **rsync or copy** those folders from the machine where you trained into the same paths under `artifacts/checkpoints/`, then set `STAGE1_ADAPTER_PATH` / `STAGE2_ADAPTER_PATH` if you use non-default names (see `src/inference/generate_checkpoint_outputs.py`).

## Dataset splits (reproducibility)

- **Alpaca:** `prepare_alpaca.py` loads `tatsu-lab/alpaca`, shuffles with **seed 42**, uses **95% train** / **5% eval** → `data/processed/alpaca_train.jsonl` and `alpaca_eval.jsonl`. Eval is never used in Stage 1 training.
- **Teacher JSON:** After teacher generation, rows are **shuffled** (`random.shuffle`). The **first** `JSON_EVAL_SIZE` rows (default **100**) become `json_eval.jsonl`; the next up to `JSON_TRAIN_CAP` rows (default **80**) become `json_train_teacher.jsonl` for Stage 2. Sets are **disjoint** by construction. Override with env vars `JSON_EVAL_SIZE`, `JSON_TRAIN_CAP` in `generate_teacher_json.py`.

## Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
```

Add your token to `.env`:

```bash
HUGGING_FACE_TOKEN=your_token_here
```

**Prompt templates (Assignment §6):** Teacher and judge prompts load from `prompts/*.md` via `src/utils/prompt_loader.py`. Optional overrides: `TEACHER_PROMPT_TEMPLATE`, `JUDGE_SYSTEM_PROMPT`, `JUDGE_PAIRWISE_PROMPT`, `JUDGE_JSON_PROMPT` (paths relative to repo root).

## UTSA HPC (GPU, Primary Workflow)

Run the full pipeline on HPC (including data prep, Stage 1, Stage 2, and evaluation):
- `hpc/stage1_train.slurm`
- `hpc/stage2_train.slurm`
- `hpc/eval_all.slurm`

Submit jobs with:

```bash
sbatch hpc/stage1_train.slurm
sbatch hpc/stage2_train.slurm
sbatch hpc/eval_all.slurm
```

### Direct commands on HPC (interactive or inside job scripts)

```bash
export PYTHONPATH="."
python src/data_prep/prepare_alpaca.py
python src/data_prep/build_json_prompts.py
python src/data_prep/generate_teacher_json.py
python src/data_prep/validate_json_dataset.py

python src/training/train_stage1_alpaca.py
python src/training/train_stage2_json.py

python src/inference/generate_checkpoint_outputs.py
python src/evaluation/eval_alpaca_auto.py
python src/evaluation/eval_alpaca_judge.py
python src/evaluation/eval_json_auto.py
python src/evaluation/eval_json_judge.py
python src/evaluation/forgetting_analysis.py
python src/evaluation/aggregate_results.py
```

### Optional local usage

Local execution is optional and intended only for quick syntax/debug checks. Assignment-grade runs should be executed on UTSA HPC GPU.
