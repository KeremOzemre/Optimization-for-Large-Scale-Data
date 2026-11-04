"""Download TinyStories and save reproducible train, validation, and test splits."""

from datasets import load_dataset, DatasetDict

seed = 42
test_ratio = 0.2

dataset = load_dataset("roneneldan/TinyStories")

# Keep the source training split separate; reserve 20% of source validation for test.
split = dataset["validation"].train_test_split(
    test_size=test_ratio,
    seed=seed
)

dataset = DatasetDict({
    # Shuffling once makes the prefix subsets in train.py less dependent on source order.
    "train": dataset["train"].shuffle(seed=seed),
    "validation": split["train"],
    "test": split["test"]
})

# train.py reads this saved copy rather than downloading and splitting again.
dataset.save_to_disk("data/tinystories")
