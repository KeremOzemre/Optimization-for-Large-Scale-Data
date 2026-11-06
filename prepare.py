"""Tokenize TinyStories into nanoGPT-style uint16 token streams."""

import json
import os
from pathlib import Path

from datasets import load_dataset
import numpy as np
import tiktoken


def cache_is_valid(out_dir, settings):
    metadata_path = out_dir / "meta.json"
    if not metadata_path.exists():
        return False
    try:
        metadata = json.loads(metadata_path.read_text())
        if any(metadata.get(key) != value for key, value in settings.items()):
            # The previous pipeline's small cache had equivalent split settings
            # but recorded "mode" instead of explicit story limits.
            legacy_small = (settings["train_stories"] == 2000
                            and settings["validation_stories"] == 200
                            and settings["test_stories"] == 200)
            legacy_full = all(settings[key] is None for key in
                              ("train_stories", "validation_stories", "test_stories"))
            if not ((legacy_small and metadata.get("mode") == "small")
                    or (legacy_full and metadata.get("mode") == "full")):
                return False
            if (metadata.get("source") != settings["source"]
                    or metadata.get("tokenizer") != settings["tokenizer"]
                    or metadata.get("vocab_size") != settings["vocab_size"]
                    or metadata.get("seed") != settings["seed"]
                    or metadata.get("test_ratio") != settings["test_ratio"]
                    or settings["train_split"] != "train"
                    or settings["heldout_split"] != "validation"):
                return False
        if metadata.get("eot_token") != 50256:
            return False
        return all(
            (out_dir / f"{split}.bin").stat().st_size
            == metadata["token_counts"][split] * np.dtype(np.uint16).itemsize
            and metadata["token_counts"][split] > 0
            for split in ("train", "val", "test")
        )
    except (OSError, KeyError, TypeError, ValueError):
        return False


def write_split(stories, path, tokenizer):
    token_count = 0
    story_count = 0
    buffer = []
    try:
        with path.open("wb") as output:
            for story in stories:
                # encode_ordinary treats any special-looking text as ordinary story text.
                buffer.extend(tokenizer.encode_ordinary(story["text"]))
                buffer.append(tokenizer.eot_token)
                story_count += 1
                if len(buffer) >= 100_000:
                    np.asarray(buffer, dtype=np.uint16).tofile(output)
                    token_count += len(buffer)
                    buffer.clear()
            if buffer:
                np.asarray(buffer, dtype=np.uint16).tofile(output)
                token_count += len(buffer)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    print(f"{path.name}: {story_count:,} stories, {token_count:,} tokens", flush=True)
    return story_count, token_count


def preparation_settings(config):
    data = config["data"]
    tokenizer = tiktoken.get_encoding(config["tokenizer"]["name"])
    return {
        "source": data["dataset"], "train_split": data["train_split"],
        "heldout_split": data["heldout_split"], "test_ratio": data["test_ratio"],
        "train_stories": data["train_stories"],
        "validation_stories": data["validation_stories"],
        "test_stories": data["test_stories"],
        "tokenizer": config["tokenizer"]["name"],
        "vocab_size": tokenizer.n_vocab,
        "seed": config["experiment"]["seed"],
    }


def prepared_cache_is_valid(config):
    return cache_is_valid(Path(config["data"]["cache_dir"]), preparation_settings(config))


def prepare_dataset(config, force=False):
    out_dir = Path(config["data"]["cache_dir"])
    settings = preparation_settings(config)
    if cache_is_valid(out_dir, settings) and not force:
        print(f"Using existing token cache: {out_dir}")
        return
    if (out_dir / "meta.json").exists() and not force:
        raise FileExistsError(f"{out_dir} has incompatible settings; choose a new data.cache_dir or use --force")

    tokenizer = tiktoken.get_encoding(settings["tokenizer"])
    if tokenizer.eot_token > np.iinfo(np.uint16).max or tokenizer.n_vocab > np.iinfo(np.uint16).max + 1:
        raise RuntimeError("Unexpected GPT-2 tokenizer vocabulary or end-of-text token")

    print(f"Loading {settings['source']} via Hugging Face Datasets...", flush=True)
    dataset = load_dataset(settings["source"])
    if settings["train_split"] not in dataset or settings["heldout_split"] not in dataset:
        raise ValueError("Configured dataset split was not found")
    held_out = dataset[settings["heldout_split"]].train_test_split(
        test_size=settings["test_ratio"], seed=settings["seed"])
    splits = {
        "train": dataset[settings["train_split"]].shuffle(
    seed=settings["seed"]
),
        "val": held_out["train"],
        "test": held_out["test"],
    }
    limits = {"train": settings["train_stories"],
              "val": settings["validation_stories"],
              "test": settings["test_stories"]}
    splits = {name: stories.select(range(min(len(stories), limits[name])))
              if limits[name] is not None else stories
              for name, stories in splits.items()}

    out_dir.mkdir(parents=True, exist_ok=True)
    # The metadata file is the completion marker. Remove it before replacing bins.
    (out_dir / "meta.json").unlink(missing_ok=True)
    story_counts = {}
    token_counts = {}
    for name, stories in splits.items():
        temporary = out_dir / f"{name}.bin.tmp"
        story_counts[name], token_counts[name] = write_split(stories, temporary, tokenizer)
        os.replace(temporary, out_dir / f"{name}.bin")

    metadata = {**settings, "eot_token": tokenizer.eot_token,
                "story_counts": story_counts, "token_counts": token_counts}
    temporary = out_dir / "meta.json.tmp"
    temporary.write_text(json.dumps(metadata, indent=2) + "\n")
    os.replace(temporary, out_dir / "meta.json")
    print(f"Saved token cache to {out_dir}")


if __name__ == "__main__":
    from run import main
    import sys
    main(["prepare", *sys.argv[1:]])
