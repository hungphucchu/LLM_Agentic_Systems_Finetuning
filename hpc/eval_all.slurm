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
source ~/.bashrc
set -u
cd "$SLURM_SUBMIT_DIR"

# Load environment variables from .env once per job.
if [ -f ".env" ]; then
  set -a
  source .env
  set +a
else
  echo "ERROR: .env not found in $SLURM_SUBMIT_DIR"
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

export PYTHONPATH="$SLURM_SUBMIT_DIR"

python3 -m pip install -r requirements.txt
python3 src/inference/generate_checkpoint_outputs.py
python3 src/evaluation/eval_alpaca_auto.py
python3 src/evaluation/eval_json_auto.py
python3 src/evaluation/eval_alpaca_judge.py
python3 src/evaluation/eval_json_judge.py
python3 src/evaluation/forgetting_analysis.py
python3 src/evaluation/aggregate_results.py
