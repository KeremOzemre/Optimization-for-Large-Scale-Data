from datasets import load_dataset, DatasetDict

seed = 42
test_ratio = 0.2

dataset = load_dataset("roneneldan/TinyStories")

# Split original validation into validation and test
split = dataset["validation"].train_test_split(
    test_size=test_ratio,
    seed=seed
)

dataset = DatasetDict({
    "train": dataset["train"].shuffle(seed=seed),
    "validation": split["train"],
    "test": split["test"]
})

# Save once
dataset.save_to_disk("data/tinystories")