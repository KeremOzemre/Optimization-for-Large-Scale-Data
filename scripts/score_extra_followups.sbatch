#!/bin/bash
#SBATCH --job-name=score-extra
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-47:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/score-extra-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

PYTHON="$HOME/Project/.venv/bin/python"

# Equal-token evaluation
"$PYTHON" -u score_checkpoints.py \
  --checkpoints \
    runs/followups_equal_tokens_143m/xl_full_equal_tokens_seed42/checkpoint.pt \
    runs/followups_equal_tokens_143m/xxl_full_equal_tokens_seed42/checkpoint.pt \
  --data-dir data/tokens-full \
  --device cuda \
  --precision float32 \
  --output-dir analysis/followups/equal_token_143m_validation

# Lower-learning-rate XXL evaluation
"$PYTHON" -u score_checkpoints.py \
  --checkpoints \
    runs/followups_lr/xxl_full_1h_lr15e-5_seed42/checkpoint.pt \
  --data-dir data/tokens-full \
  --device cuda \
  --precision float32 \
  --output-dir analysis/followups/lr_validation

echo "All evaluations completed."
