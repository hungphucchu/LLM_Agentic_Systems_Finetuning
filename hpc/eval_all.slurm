#!/bin/bash
#SBATCH --job-name=a3_eval
#SBATCH --partition=gpu1v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=eval_%j.out
#SBATCH --error=eval_%j.err

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

echo "[eval] started at $(date) on $(hostname)"
echo "[eval] SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR"
echo "[eval] PROJECT_ROOT=$PROJECT_ROOT"
pwd

# Load environment variables from .env once per job.
if [ -f ".env" ]; then
  set -a
  source .env
  set +a
else
  echo "ERROR: .env not found in $PROJECT_ROOT"
  exit 1
fi

# Map project variable names to standard names used by scripts/libraries.
if [ -n "${HUGGING_FACE_TOKEN:-}" ] && [ -z "${HF_TOKEN:-}" ]; then
  export HF_TOKEN="$HUGGING_FACE_TOKEN"
fi
if [ -n "${UTSA_API_KEY:-}" ] && [ -z "${API_KEY:-}" ]; then
  export API_KEY="$UTSA_API_KEY"
fi
if [ -n "${UTSA_BASE_URL:-}" ] && [ -z "${BASE_URL:-}" ]; then
  export BASE_URL="$UTSA_BASE_URL"
fi
if [ -n "${UTSA_MODEL:-}" ] && [ -z "${TEACHER_MODEL:-}" ]; then
  export TEACHER_MODEL="$UTSA_MODEL"
fi

export PYTHONPATH="$PROJECT_ROOT"

python3 -m pip install -r requirements.txt
bash hpc/install_torch_cu121.sh

python3 src/inference/generate_checkpoint_outputs.py
python3 src/evaluation/eval_alpaca_auto.py
python3 src/evaluation/eval_json_auto.py
python3 src/evaluation/eval_alpaca_judge.py
python3 src/evaluation/eval_json_judge.py
python3 src/evaluation/forgetting_analysis.py
python3 src/evaluation/aggregate_results.py
