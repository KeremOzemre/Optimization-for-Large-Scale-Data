"""One entry point for TinyStories preparation, training, evaluation, and generation."""

import argparse
import math
from pathlib import Path

import yaml


CONFIG_KEYS = {
    "experiment": {"name", "seed", "output_dir"},
    "data": {"dataset", "train_split", "heldout_split", "test_ratio",
             "train_stories", "validation_stories", "test_stories", "cache_dir"},
    "tokenizer": {"name"},
    "model": {"layers", "width", "heads", "context_length", "dropout"},
    "training": {"device", "precision", "time_limit_seconds", "microbatch_size",
                 "gradient_accumulation_steps", "log_interval_steps"},
    "optimizer": {"learning_rate", "weight_decay", "betas", "gradient_clip"},
    "schedule": {"warmup_fraction", "min_lr_ratio"},
    "evaluation": {"interval_seconds", "batches"},
    "generation": {"prompt", "max_new_tokens", "temperature", "top_k"},
}


def positive_number(value, name):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value <= 0):
        raise ValueError(f"{name} must be positive")


def positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def load_config(path):
    with Path(path).open() as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping")
    if set(config) != set(CONFIG_KEYS):
        raise ValueError(f"Configuration sections must be: {', '.join(CONFIG_KEYS)}")
    for section, fields in CONFIG_KEYS.items():
        if not isinstance(config[section], dict) or set(config[section]) != fields:
            raise ValueError(f"{section} needs exactly these fields: {', '.join(sorted(fields))}")

    experiment, data, tokenizer = (config[key] for key in ("experiment", "data", "tokenizer"))
    model, training, optimizer = (config[key] for key in ("model", "training", "optimizer"))
    schedule, evaluation, generation = (config[key] for key in ("schedule", "evaluation", "generation"))
    if not isinstance(experiment["name"], str) or not experiment["name"].strip():
        raise ValueError("experiment.name must be nonempty")
    if not isinstance(experiment["output_dir"], str) or not experiment["output_dir"].strip():
        raise ValueError("experiment.output_dir must be nonempty")
    if type(experiment["seed"]) is not int or experiment["seed"] < 0:
        raise ValueError("experiment.seed must be a nonnegative integer")
    if data["dataset"] != "roneneldan/TinyStories":
        raise ValueError("Only roneneldan/TinyStories is supported")
    if (not isinstance(data["train_split"], str)
            or not isinstance(data["heldout_split"], str)
            or data["train_split"] == data["heldout_split"]):
        raise ValueError("data.train_split and data.heldout_split must be different split names")
    if not isinstance(data["cache_dir"], str) or not data["cache_dir"].strip():
        raise ValueError("data.cache_dir must be nonempty")
    if not isinstance(data["test_ratio"], (int, float)) or not 0 < data["test_ratio"] < 1:
        raise ValueError("data.test_ratio must be between 0 and 1")
    for field in ("train_stories", "validation_stories", "test_stories"):
        if data[field] is not None:
            positive_integer(data[field], f"data.{field}")
    if tokenizer["name"] != "gpt2":
        raise ValueError("Only the tiktoken gpt2 tokenizer is supported")
    for field in ("layers", "width", "heads", "context_length"):
        positive_integer(model[field], f"model.{field}")
    if model["width"] % model["heads"]:
        raise ValueError("model.width must be divisible by model.heads")
    if not isinstance(model["dropout"], (int, float)) or not 0 <= model["dropout"] < 1:
        raise ValueError("model.dropout must be in [0, 1)")
    if training["device"] not in ("auto", "cuda", "mps", "cpu"):
        raise ValueError("training.device must be auto, cuda, mps, or cpu")
    if training["precision"] not in ("auto", "float32", "bfloat16"):
        raise ValueError("training.precision must be auto, float32, or bfloat16")
    positive_number(training["time_limit_seconds"], "training.time_limit_seconds")
    for field in ("microbatch_size", "gradient_accumulation_steps", "log_interval_steps"):
        positive_integer(training[field], f"training.{field}")
    positive_number(optimizer["learning_rate"], "optimizer.learning_rate")
    if not isinstance(optimizer["weight_decay"], (int, float)) or optimizer["weight_decay"] < 0:
        raise ValueError("optimizer.weight_decay cannot be negative")
    if not isinstance(optimizer["gradient_clip"], (int, float)) or optimizer["gradient_clip"] < 0:
        raise ValueError("optimizer.gradient_clip cannot be negative")
    if (not isinstance(optimizer["betas"], list) or len(optimizer["betas"]) != 2
            or any(not isinstance(beta, (int, float)) or not 0 < beta < 1
                   for beta in optimizer["betas"])):
        raise ValueError("optimizer.betas must contain two numbers between 0 and 1")
    if (not isinstance(schedule["warmup_fraction"], (int, float))
            or not 0 <= schedule["warmup_fraction"] < 1):
        raise ValueError("schedule.warmup_fraction must be in [0, 1)")
    if (not isinstance(schedule["min_lr_ratio"], (int, float))
            or not 0 < schedule["min_lr_ratio"] <= 1):
        raise ValueError("schedule.min_lr_ratio must be in (0, 1]")
    positive_number(evaluation["interval_seconds"], "evaluation.interval_seconds")
    positive_integer(evaluation["batches"], "evaluation.batches")
    if not isinstance(generation["prompt"], str) or not generation["prompt"]:
        raise ValueError("generation.prompt must be nonempty")
    if type(generation["max_new_tokens"]) is not int or generation["max_new_tokens"] < 0:
        raise ValueError("generation.max_new_tokens must be a nonnegative integer")
    positive_number(generation["temperature"], "generation.temperature")
    positive_integer(generation["top_k"], "generation.top_k")
    return config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "train", "evaluate", "generate", "run"))
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", help="Checkpoint for evaluate or generate")
    parser.add_argument("--prompt", help="Override generation.prompt for this request")
    parser.add_argument("--force", action="store_true", help="Rebuild an existing data cache")
    args = parser.parse_args(argv)
    if args.force and args.action not in ("prepare", "run"):
        parser.error("--force applies only to prepare and run")
    if args.checkpoint and args.action not in ("evaluate", "generate"):
        parser.error("--checkpoint applies only to evaluate and generate")
    if args.prompt and args.action != "generate":
        parser.error("--prompt applies only to generate")
    try:
        config = load_config(args.config)
        if args.action in ("prepare", "run"):
            from prepare import prepare_dataset
            if args.action == "run" and Path(config["experiment"]["output_dir"]).exists():
                raise FileExistsError("Training output already exists; choose a new experiment.output_dir")
            prepare_dataset(config, force=args.force)
        if args.action in ("train", "run"):
            from train import train_model
            train_model(config)
        if args.action in ("evaluate", "run"):
            from evaluate import evaluate_checkpoint
            checkpoint = args.checkpoint or str(Path(config["experiment"]["output_dir"]) / "checkpoint.pt")
            evaluate_checkpoint(config, checkpoint)
        if args.action == "generate":
            from generate import generate_text
            checkpoint = args.checkpoint or str(Path(config["experiment"]["output_dir"]) / "checkpoint.pt")
            generate_text(config, checkpoint, prompt=args.prompt)
    except (ValueError, FileExistsError, FileNotFoundError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
