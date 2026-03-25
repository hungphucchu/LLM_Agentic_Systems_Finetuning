#!/bin/bash
#SBATCH --job-name=a3_eval
#SBATCH --partition=gpu1v100
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=artifacts/logs/eval_%j.out

set -euo pipefail
source ~/.bashrc
cd "$SLURM_SUBMIT_DIR"

export PYTHONPATH="$SLURM_SUBMIT_DIR"

python3 -m pip install -r requirements.txt
python3 src/inference/generate_checkpoint_outputs.py
python3 src/evaluation/eval_alpaca_auto.py
python3 src/evaluation/eval_json_auto.py
python3 src/evaluation/eval_alpaca_judge.py
python3 src/evaluation/eval_json_judge.py
python3 src/evaluation/forgetting_analysis.py
python3 src/evaluation/aggregate_results.py
