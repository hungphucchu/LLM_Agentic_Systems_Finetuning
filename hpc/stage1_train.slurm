#!/bin/bash
#SBATCH --job-name=a3_stage1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=artifacts/logs/stage1_%j.out

set -euo pipefail
source ~/.bashrc
cd "$SLURM_SUBMIT_DIR"

export PYTHONPATH="$SLURM_SUBMIT_DIR"

python3 -m pip install -r requirements.txt
python3 src/data_prep/prepare_alpaca.py
python3 src/training/train_stage1_alpaca.py
