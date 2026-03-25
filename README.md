# Assignment 3: Sequential Instruction Tuning Pipeline

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
- `artifacts/`: checkpoints, predictions, logs, metrics, figures

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
