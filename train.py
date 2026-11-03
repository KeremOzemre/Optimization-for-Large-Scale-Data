from datasets import load_from_disk

dataset = load_from_disk("data/tinystories")

def get_train_subset(fraction):
    train = dataset["train"]
    cutoff = int(len(train) * fraction)
    return train.select(range(cutoff))

train_1 = get_train_subset(0.25)
train_2 = get_train_subset(0.50)
train_3 = get_train_subset(0.75)
train_4 = get_train_subset(1.0)



