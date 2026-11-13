"""Score checkpoints on identical validation targets and save paired story losses.

Full mode scores every target except token zero, resetting context every 256
tokens (or --context-length). Sampled mode reproduces fixed in-loop windows.
No training or generation occurs here; test scoring requires an explicit flag.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from train import (autocast_context, batch_at, file_sha256,
                   hardware_name, load_checkpoint_model, load_tokens,
                   resolve_precision, select_device)


def evaluation_windows(token_count, context_length, batch_size, sample_windows, seed):
    """Yield offsets and lengths; include the short final window in a full pass."""
    if sample_windows:
        starts = np.random.default_rng(seed).integers(
            0, token_count - context_length, size=sample_windows)
        for i in range(0, len(starts), batch_size):
            yield starts[i:i + batch_size], context_length
    else:
        full = (token_count - 1) // context_length
        for i in range(0, full, batch_size):
            yield np.arange(i, min(i + batch_size, full)) * context_length, context_length
        remaining = token_count - 1 - full * context_length
        if remaining:
            yield np.array([full * context_length]), remaining


def score_stories(model, tokens, windows, device, precision, eot_token=50256):
    """Aggregate per-token NLL by the story containing its target, including EOT.

    Context may cross EOT, exactly as in training. Story bootstrap intervals are
    conditional on these checkpoints; they do not measure training-seed variance.
    """
    ends = np.flatnonzero(tokens == eot_token)
    if not len(ends) or ends[-1] != len(tokens) - 1:
        raise ValueError("Expected an end-of-text marker after every story")
    nll = np.zeros(len(ends), dtype=np.float64)
    counts = np.zeros(len(ends), dtype=np.int64)
    model.eval()
    with torch.inference_mode():
        for starts, length in windows:
            x, y = batch_at(tokens, starts, length, device)
            with autocast_context(device, precision):
                logits, _ = model(x, y)
                losses = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)),
                                         y.reshape(-1), reduction="none")
            positions = (np.asarray(starts)[:, None] + np.arange(1, length + 1)).reshape(-1)
            story_ids = np.searchsorted(ends, positions, side="left")
            nll += np.bincount(story_ids, weights=losses.cpu().numpy(), minlength=len(ends))
            counts += np.bincount(story_ids, minlength=len(ends))
    if not np.isfinite(nll).all() or counts.sum() == 0:
        raise RuntimeError("Evaluation produced invalid losses or zero targets")
    return nll, counts


def paired_bootstrap(a, b, counts, repeats=1000, seed=123):
    """Positive delta means checkpoint B has higher (worse) CE than A."""
    keep = counts > 0
    a, b, counts = a[keep], b[keep], counts[keep]
    rng = np.random.default_rng(seed)
    deltas = np.empty(repeats)
    for i in range(repeats):
        draw = rng.integers(0, len(counts), size=len(counts))
        deltas[i] = (b[draw].sum() - a[draw].sum()) / counts[draw].sum()
    return {"delta_b_minus_a": float((b.sum() - a.sum()) / counts.sum()),
            "ci95_low": float(np.quantile(deltas, 0.025)),
            "ci95_high": float(np.quantile(deltas, 0.975)), "bootstrap_repeats": repeats}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--data-dir", default="data/tokens-full")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sample-windows", type=int, default=0, help="0: full pass; 256: 65,536 targets at context 256")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "mps", "cpu"))
    parser.add_argument("--precision", default="float32", choices=("auto", "float32", "bfloat16"))
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if min(args.context_length, args.batch_size, args.bootstrap_repeats) <= 0 or args.sample_windows < 0:
        parser.error("Lengths, batch size, and bootstrap repeats must be positive; sample windows may be zero")
    out = Path(args.output_dir)
    if out.exists():
        parser.error("Choose a new output directory; previous scores are never overwritten")
    device = select_device(args.device)
    precision = resolve_precision(device, args.precision)
    data_dir = Path(args.data_dir)
    meta = json.loads((data_dir / "meta.json").read_text())
    if meta["tokenizer"] != "gpt2" or meta["vocab_size"] != 50257:
        parser.error("Evaluation requires the project's GPT-2 token cache")
    tokens = load_tokens(data_dir, args.split, args.context_length)
    if len(tokens) != meta["token_counts"][args.split]:
        parser.error("Token count does not match cache metadata")
    paths = [Path(path) for path in args.checkpoints]
    labels = [path.parent.name for path in paths]
    if len(set(labels)) != len(labels):
        parser.error("Each checkpoint must have a distinct parent-directory name")
    if any(not path.is_file() for path in paths):
        parser.error("A checkpoint is missing; run download_checkpoints.py or finish training first")
    protocol = {"split": args.split, "tokenizer": "gpt2", "context_length": args.context_length,
                "sample_windows": args.sample_windows, "seed": args.seed,
                "batch_size": args.batch_size, "precision": precision,
                "data_sha256": file_sha256(data_dir / f"{args.split}.bin"),
                "metadata_sha256": file_sha256(data_dir / "meta.json"),
                "preparation_seed": meta["seed"], "source": meta["source"],
                "test_ratio": meta["test_ratio"], "cache_tokens": len(tokens),
                "context_policy": "reset per window; include short tail in full mode; score EOT",
                "hardware": hardware_name(device), "torch": str(torch.__version__)}
    out.mkdir(parents=True)
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    scores, rows, reference_counts = [], [], None
    for label, path in zip(labels, paths):
        print(f"Scoring {label} on common {args.split} targets...", flush=True)
        model, checkpoint = load_checkpoint_model(path, device)
        if model.config.block_size < args.context_length or model.config.vocab_size != 50257:
            raise ValueError(f"Checkpoint is incompatible with the common scoring protocol: {path}")
        saved = checkpoint.get("run_config", {})
        if "data" in saved:
            expected_seed = saved["data"].get("preparation_seed", saved["experiment"]["seed"])
            if (expected_seed != meta["seed"] or saved["data"]["test_ratio"] != meta["test_ratio"]
                    or saved["data"]["dataset"] != meta["source"]):
                raise ValueError("Checkpoint and scoring cache use different source/split settings")
        row = {"run": label, "checkpoint_sha256": file_sha256(path),
               "training_seed": checkpoint["seed"], "tokens_processed": checkpoint["tokens_processed"],
               "training_seconds": checkpoint["elapsed_seconds"],
               "training_hardware": saved.get("runtime", {}).get("hardware"),
               "training_data_sha256": saved.get("runtime", {}).get("data_sha256", {}).get("train.bin"),
               "parameters": sum(p.numel() for p in model.parameters())}
        (out / f"{label}_training_config.json").write_text(json.dumps(saved, indent=2) + "\n")
        del checkpoint  # Release AdamW state before scoring the next model.
        windows = evaluation_windows(len(tokens), args.context_length, args.batch_size,
                                     args.sample_windows, args.seed)
        nll, counts = score_stories(model, tokens, windows, device, precision, meta["eot_token"])
        if reference_counts is not None and not np.array_equal(counts, reference_counts):
            raise RuntimeError("Checkpoints were evaluated on different targets")
        reference_counts = counts
        loss = float(nll.sum() / counts.sum())
        row.update(loss=loss, perplexity=math.exp(loss), scored_targets=int(counts.sum()))
        np.savez_compressed(out / f"{label}_stories.npz", nll=nll, counts=counts)
        rows.append(row)
        scores.append(nll)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            torch.mps.empty_cache()
        print(f"{label}: CE={loss:.6f}, PPL={math.exp(loss):.4f}, targets={counts.sum():,}", flush=True)
    with (out / "scores.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    comparisons = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            comparisons.append({"a": labels[i], "b": labels[j],
                                **paired_bootstrap(scores[i], scores[j], reference_counts,
                                                   args.bootstrap_repeats)})
    (out / "paired_comparisons.json").write_text(json.dumps({
        "interpretation": "Positive B-A favours A. Story-resampling intervals condition on fixed checkpoints; cross-story context and duplicate stories may induce dependence. They do not measure training-seed uncertainty. Intervals are unadjusted for multiple comparisons.",
        "comparisons": comparisons}, indent=2) + "\n")


if __name__ == "__main__":
    main()
