#!/bin/bash
#SBATCH --job-name=ts-three
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-47:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --array=0-2
#SBATCH --output=logs/ts-three-%A_%a.out

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

ulimit -v unlimited || true

PYTHON="$HOME/Project/.venv/bin/python"

CONFIGS=(
    "configs/experiments/xl_ctx512.yaml"
    "configs/experiments/xl_t1800.yaml"
    "configs/experiments/xxl_t1800.yaml"
)

CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"

echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Experiment: $CONFIG"

nvidia-smi

"$PYTHON" -u run.py train --config "$CONFIG"
