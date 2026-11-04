"""Load prepared TinyStories data and define nested training subsets.

This file does not tokenize stories or train a model yet.
"""

from datasets import load_from_disk

# Run prepare_data.py first to create this local dataset.
dataset = load_from_disk("data/tinystories")


def get_train_subset(fraction):
    """Return the first `fraction` of the shuffled training split."""
    train = dataset["train"]
    cutoff = int(len(train) * fraction)
    return train.select(range(cutoff))


# Prefix selection makes the 25%, 50%, 75%, and 100% subsets nested.
train_1 = get_train_subset(0.25)
train_2 = get_train_subset(0.50)
train_3 = get_train_subset(0.75)
train_4 = get_train_subset(1.0)
