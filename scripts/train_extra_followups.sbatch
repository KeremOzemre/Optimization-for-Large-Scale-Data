#!/bin/bash
#SBATCH --job-name=ts-extra
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-47:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/extra-%A_%a.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

PYTHON="$HOME/Project/.venv/bin/python"

case "$SLURM_ARRAY_TASK_ID" in
  0)
    CONFIG="configs/followups/xl_full_equal_tokens_seed42.yaml"
    ;;
  1)
    CONFIG="configs/followups/xxl_full_equal_tokens_seed42.yaml"
    ;;
  2)
    CONFIG="configs/followups/xxl_full_1h_lr15e-5_seed42.yaml"
    ;;
esac

echo "Training $CONFIG"

"$PYTHON" -u run.py train --config "$CONFIG"

echo "Finished $CONFIG"
