#!/bin/bash
#SBATCH --job-name=ts-followups
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-47:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --array=0-5%2
#SBATCH --output=logs/followups-%A_%a.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

ulimit -v unlimited || true
export TIKTOKEN_CACHE_DIR="$PWD/data/tiktoken-cache"

CONFIGS=(
  configs/followups/xl_full_1h_seed42.yaml
  configs/followups/xxl_full_1h_seed42.yaml
  configs/followups/xxl_full_1h_seed43.yaml
  configs/followups/xl_full_1h_seed43.yaml
  configs/followups/xl_full_1h_seed44.yaml
  configs/followups/xxl_full_1h_seed44.yaml
)

CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"

echo "Node: $(hostname)"
echo "Config: $CONFIG"
nvidia-smi

"$HOME/Project/.venv/bin/python" -u run.py train --config "$CONFIG"
