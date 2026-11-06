#!/bin/bash
#SBATCH --job-name=hf-upload
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-47:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/hf-upload-%j.out

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

ulimit -v unlimited || true

echo "Node: $(hostname)"
echo "Virtual memory limit: $(ulimit -v)"

HF="$HOME/Project/.venv/bin/hf"

"$HF" upload \
    dsa4212-tinystories/tinystories-nanogpt \
    runs/large_data100 \
    large_data100 \
    --repo-type model

echo "Upload completed."

