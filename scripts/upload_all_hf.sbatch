#!/bin/bash
#SBATCH --job-name=hf-upload
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-47:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/hf-upload-%j.out

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
ulimit -v unlimited || true

HF="$HOME/Project/.venv/bin/hf"
REPO="dsa4212-tinystories/tinystories-nanogpt"

echo "Node: $(hostname)"
echo "Uploading checkpoints to $REPO"

"$HF" auth whoami

for MODEL in small medium large xl xxl; do
    for FRACTION in 25 50 75 100; do
        NAME="${MODEL}_data${FRACTION}"
        FOLDER="runs/$NAME"

        if [ ! -f "$FOLDER/checkpoint.pt" ]; then
            echo "ERROR: Missing checkpoint for $NAME"
            exit 1
        fi
    done
done

for MODEL in small medium large xl xxl; do
    for FRACTION in 25 50 75 100; do
        NAME="${MODEL}_data${FRACTION}"

        echo "================================"
        echo "Uploading $NAME"
        echo "================================"

        "$HF" upload \
            "$REPO" \
            "runs/$NAME" \
            "$NAME" \
            --repo-type model

        echo "Successfully uploaded $NAME"
    done
done

echo "All 20 checkpoints uploaded successfully."
