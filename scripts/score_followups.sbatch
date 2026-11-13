#!/bin/bash
#SBATCH --job-name=score-followups
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-47:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/score-followups-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

ulimit -v unlimited || true

PYTHON="$HOME/Project/.venv/bin/python"

"$PYTHON" -u score_checkpoints.py \
  --checkpoints \
    runs/followups_1h/xl_full_1h_seed42/checkpoint.pt \
    runs/followups_1h/xxl_full_1h_seed42/checkpoint.pt \
    runs/followups_1h/xl_full_1h_seed43/checkpoint.pt \
    runs/followups_1h/xxl_full_1h_seed43/checkpoint.pt \
    runs/followups_1h/xl_full_1h_seed44/checkpoint.pt \
    runs/followups_1h/xxl_full_1h_seed44/checkpoint.pt \
  --data-dir data/tokens-full \
  --device cuda \
  --precision float32 \
  --output-dir analysis/followups/seed_validation

"$PYTHON" -u summarize_followups.py \
  --scores analysis/followups/seed_validation/scores.csv \
  --output analysis/followups/seed_summary.md
