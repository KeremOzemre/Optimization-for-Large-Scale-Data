# Small language model on TinyStories

This project is a starting point for training a small language model on the [TinyStories dataset](https://huggingface.co/datasets/roneneldan/TinyStories) with limited compute. The current code prepares the dataset and defines training subsets; it does not train a model yet.

## Setup

From the project root, create a Python environment and install the current dependency:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install datasets
```

## Prepare the data

```bash
python prepare_data.py
```

`prepare_data.py` downloads `roneneldan/TinyStories` through Hugging Face Datasets and saves a processed `DatasetDict` to `data/tinystories`. It:

1. Shuffles the original training split with seed `42`.
2. Splits the original validation split into a new validation split (80%) and a held-out test split (20%), also using seed `42`.
3. Saves the three splits to disk for reuse.

The percentages in step 2 apply **only to the original validation split**; no stories are taken from the training split for validation or testing. Run this script before `train.py`. It needs an internet connection on the first download and enough disk space for the dataset.

The scripts use the Hugging Face dataset and the saved `data/tinystories` directory. They do not read any separate TinyStories `.txt` files in `data/`.

## Training subsets

```bash
python train.py
```

`train.py` currently loads `data/tinystories` and selects the first 25%, 50%, 75%, and 100% of the shuffled training split as `train_1` through `train_4`. These subsets are nested, so each smaller subset is contained in the larger ones. The script currently produces no output and does not train a model.
