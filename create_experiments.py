
import copy
import json
from pathlib import Path

import yaml

BASE = Path("configs/default.yaml")
OUT = Path("configs/experiments")
OUT.mkdir(parents=True, exist_ok=True)

with BASE.open() as f:
    default = yaml.safe_load(f)

# Approximate model sizes: 16M, 30M, 51M
MODELS = {
    "small":  {"layers": 4, "width": 256, "heads": 4},
    "medium": {"layers": 6, "width": 384, "heads": 6},
    "large":  {"layers": 8, "width": 512, "heads": 8},
}

# Requires a prepared full dataset with meta.json
metadata_path = Path("data/tinystories/full/meta.json")
with metadata_path.open() as f:
    metadata = json.load(f)

total_stories = metadata["story_counts"]["train"]

for fraction in [25, 50, 75, 100]:
    n_stories = round(total_stories * fraction / 100)

    for model_name, architecture in MODELS.items():
        config = copy.deepcopy(default)

        name = f"{model_name}_data{fraction}"

        config["experiment"]["name"] = name
        config["experiment"]["seed"] = 42
        config["experiment"]["output_dir"] = f"runs/{name}"

        config["data"]["train_stories"] = (
            None if fraction == 100 else n_stories
        )
        config["data"]["cache_dir"] = (
            f"data/tinystories/data{fraction}"
        )

        config["model"].update(architecture)
        config["model"]["context_length"] = 256
        config["model"]["dropout"] = 0.1

        config["training"]["device"] = "cuda"
        config["training"]["precision"] = "bfloat16"
        config["training"]["time_limit_seconds"] = 3600
        config["training"]["microbatch_size"] = 16
        config["training"]["gradient_accumulation_steps"] = 1

        config["optimizer"]["learning_rate"] = 3e-4
        config["optimizer"]["weight_decay"] = 0.1
        config["optimizer"]["betas"] = [0.9, 0.95]
        config["optimizer"]["gradient_clip"] = 1.0

        config["schedule"]["warmup_fraction"] = 0.02
        config["schedule"]["min_lr_ratio"] = 0.1

        config["evaluation"]["interval_seconds"] = 300

        path = OUT / f"{name}.yaml"
        with path.open("w") as f:
            yaml.safe_dump(config, f, sort_keys=False)

        print(f"Created {path}")
