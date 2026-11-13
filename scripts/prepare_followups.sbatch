#!/bin/bash
#SBATCH --job-name=prepare-full
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-47:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/prepare-full-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

ulimit -v unlimited || true

export TIKTOKEN_CACHE_DIR="$PWD/data/tiktoken-cache"

"$HOME/Project/.venv/bin/python" -u run.py prepare --config configs/followups/xl_full_1h_seed42.yaml
