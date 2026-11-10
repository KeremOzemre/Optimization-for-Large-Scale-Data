#!/bin/bash
#SBATCH --job-name=score-archived
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-47:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/score-archived-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

ulimit -v unlimited || true

"$HOME/Project/.venv/bin/python" -u score_checkpoints.py \
  --checkpoints \
    hf_cache/large_data100/checkpoint.pt \
    hf_cache/xl_data50/checkpoint.pt \
    hf_cache/xl_data100/checkpoint.pt \
    hf_cache/xxl_data100/checkpoint.pt \
  --data-dir data/tokens-full \
  --device cuda \
  --precision float32 \
  --output-dir analysis/followups/archived_validation
