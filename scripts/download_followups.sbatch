#!/bin/bash
#SBATCH --job-name=download-ckpts
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-47:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/download-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

ulimit -v unlimited || true

"$HOME/Project/.venv/bin/python" -u download_checkpoints.py \
  --revision fa18888b1b9b599e9deb455ca740896e7d0642f3
