#!/bin/bash
#SBATCH --job-name=tinystories
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-47:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=logs/tinystories-%j.out

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
ulimit -v unlimited || true

PYTHON="$HOME/Project/.venv/bin/python"
CONFIG="$1"

echo "Node: $(hostname)"
nvidia-smi

"$PYTHON" -u run.py train --config "$CONFIG"
