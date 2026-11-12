#!/bin/bash
#SBATCH --job-name=hf-all-followups
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-47:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/hf-all-followups-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

"$HOME/Project/.venv/bin/python" -u upload_all_followups.py
