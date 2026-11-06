#!/bin/bash
#SBATCH --job-name=ts-prepare
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-47:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/prepare-%j.out

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

ulimit -v unlimited || true

PYTHON="$HOME/Project/.venv/bin/python"

echo "Node: $(hostname)"
echo "Virtual memory limit: $(ulimit -v)"
nvidia-smi

for FRACTION in 25 50 75 100; do
    echo "Preparing TinyStories ${FRACTION}%"

    "$PYTHON" run.py prepare \
        --config "configs/experiments/small_data${FRACTION}.yaml"
done

echo "All four datasets prepared."
