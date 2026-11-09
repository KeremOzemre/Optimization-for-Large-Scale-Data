#!/bin/bash
#SBATCH --job-name=ts-eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-47:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --array=0-3
#SBATCH --output=logs/ts-eval-%A_%a.out

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

ulimit -v unlimited || true

PYTHON="$HOME/Project/.venv/bin/python"

CONFIGS=(
    "configs/experiments/large_t1800.yaml"
    "configs/experiments/xl_t1800.yaml"
    "configs/experiments/xxl_t1800.yaml"
    "configs/experiments/xl_ctx512.yaml"
)

CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"

echo "Node: $(hostname)"
echo "Evaluating: $CONFIG"

nvidia-smi

"$PYTHON" -u run.py evaluate --config "$CONFIG"
