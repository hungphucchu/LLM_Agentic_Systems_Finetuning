#!/bin/bash
#SBATCH --job-name=a3_stage1
#SBATCH --partition=gpu1v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=artifacts/logs/stage1_%j.out

set -euo pipefail
source ~/.bashrc
cd "$SLURM_SUBMIT_DIR"

# Load environment variables from .env once per job.
# This avoids manual exports in every interactive session.
if [ -f ".env" ]; then
  set -a
  source .env
  set +a
else
  echo "ERROR: .env not found in $SLURM_SUBMIT_DIR"
  exit 1
fi

# Map project variable names to standard HF names when needed.
if [ -n "${HUGGING_FACE_TOKEN:-}" ] && [ -z "${HF_TOKEN:-}" ]; then
  export HF_TOKEN="$HUGGING_FACE_TOKEN"
fi

export PYTHONPATH="$SLURM_SUBMIT_DIR"

python3 -m pip install -r requirements.txt
python3 src/data_prep/prepare_alpaca.py
python3 src/training/train_stage1_alpaca.py
