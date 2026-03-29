#!/bin/bash
#SBATCH --job-name=a3_stage1
#SBATCH --partition=gpu1v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1            
#SBATCH --time=12:00:00
#SBATCH --output=stage1_%j.out
#SBATCH --error=stage1_%j.err

set -eo pipefail
# Some cluster bashrc files reference unset vars; source safely.
set +u
source ~/.bashrc || true
set -u

# Prefer SLURM_SUBMIT_DIR (real submission cwd). Some clusters copy scripts to
# /var/spool/slurm, which makes BASH_SOURCE unusable for project root detection.
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-}"
if [ -z "$PROJECT_ROOT" ] || [ ! -f "$PROJECT_ROOT/requirements.txt" ]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi
cd "$PROJECT_ROOT"

echo "[stage1] started at $(date) on $(hostname)"
echo "[stage1] SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR"
echo "[stage1] PROJECT_ROOT=$PROJECT_ROOT"
pwd

# Load environment variables from .env once per job.
# This avoids manual exports in every interactive session.
if [ -f ".env" ]; then
  set -a
  source .env
  set +a
else
  echo "ERROR: .env not found in $PROJECT_ROOT"
  exit 1
fi

# Map project variable names to standard HF names when needed.
if [ -n "${HUGGING_FACE_TOKEN:-}" ] && [ -z "${HF_TOKEN:-}" ]; then
  export HF_TOKEN="$HUGGING_FACE_TOKEN"
fi

export PYTHONPATH="$PROJECT_ROOT"

python3 -m pip install -r requirements.txt
# PyPI torch often needs a newer NVIDIA driver than the cluster provides; cu121 wheels work on typical V100 nodes.
python3 -m pip install --upgrade "torch>=2.2.0" --index-url https://download.pytorch.org/whl/cu121

python3 src/data_prep/prepare_alpaca.py
python3 src/training/train_stage1_alpaca.py
