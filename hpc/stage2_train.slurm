#!/bin/bash
#SBATCH --job-name=a3_stage2
#SBATCH --partition=gpu1v100
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=artifacts/logs/stage2_%j.out

set -euo pipefail
source ~/.bashrc
cd "$SLURM_SUBMIT_DIR"

export PYTHONPATH="$SLURM_SUBMIT_DIR"

python3 -m pip install -r requirements.txt
python3 src/data_prep/build_json_prompts.py
python3 src/data_prep/generate_teacher_json.py
python3 src/data_prep/validate_json_dataset.py
python3 src/training/train_stage2_json.py
