"""Test-set evaluation and story generation for every checkpoint on Hugging Face.

Why this exists: evaluate.py needs the local token cache (data/.../test.bin) that
only the training node had. This script rebuilds the exact same test split from
the TinyStories validation parquet (train_test_split, test_size=0.2, seed=42),
downloads each checkpoint from the Hub one at a time, evaluates it on the FULL
test split (not 16 sampled batches), reports loss / perplexity / bits-per-byte,
generates a few stories from fixed prompts, and deletes the checkpoint again so
the disk never holds more than one at a time.

Usage (from the repo root, with a venv that has requirements.txt + huggingface_hub):
    python eval_hf.py                 # all 20 runs
    python eval_hf.py --runs small_data100 xl_data50
    python eval_hf.py --device cpu    # if MPS misbehaves

Outputs:
    results/<run>/run_config.json, metrics.csv   (copied from the Hub, so analyze.py works)
    results/<run>/test_results.json              (loss, ppl, bpb, token/byte counts)
    results/<run>/generations.md
    analysis/test_summary.csv
    analysis/generations_all.md
"""

import argparse
import csv
import json
import math
import os
from pathlib import Path

import numpy as np
import tiktoken
import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download

from model import GPT, GPTConfig

REPO_ID = "dsa4212-tinystories/tinystories-nanogpt"
SIZES = ["small", "medium", "large", "xl", "xxl"]
FRACTIONS = [25, 50, 75, 100]
ALL_RUNS = [f"{s}_data{f}" for s in SIZES for f in FRACTIONS]

PROMPTS = [
    "Once upon a time",
    "One day, a little girl named Lily",
    "Tom had a big red ball.",
    "The sun was shining and the birds were singing.",
    "Max the dog was very hungry, so",
]
GEN_TOKENS = 150
GEN_TEMPERATURE = 0.8
GEN_TOP_K = 40
GEN_SEED = 42
BATCH_SIZE = 32


def pick_device(requested):
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_test_tokens(cache_path, expected=957_274):
    """Same recipe as prepare.py: validation split -> train_test_split(0.2, seed 42) -> 'test'."""
    if cache_path.exists():
        tokens = np.fromfile(cache_path, dtype=np.uint16)
        print(f"Using cached test tokens: {cache_path} ({len(tokens):,} tokens)")
        return tokens
    enc = tiktoken.get_encoding("gpt2")
    print("Downloading TinyStories validation split and rebuilding the test split...")
    held_out = load_dataset("roneneldan/TinyStories", split="validation")
    test = held_out.train_test_split(test_size=0.2, seed=42)["test"]
    buf = []
    for story in test:
        buf.extend(enc.encode_ordinary(story["text"]))
        buf.append(enc.eot_token)
    tokens = np.asarray(buf, dtype=np.uint16)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tokens.tofile(cache_path)
    print(f"Test split: {len(test):,} stories, {len(tokens):,} tokens")
    if len(tokens) != expected:
        print(f"WARNING: expected {expected:,} tokens (from run_config.json), got {len(tokens):,}. "
              "The split differs from the training node; results are still valid but say so in the report.")
    return tokens


def token_bytes_table(enc):
    return np.array([len(enc.decode_single_token_bytes(i)) for i in range(enc.n_vocab)], dtype=np.int64)


@torch.no_grad()
def evaluate_full(model, tokens, block_size, device, bytes_table):
    """Non-overlapping windows over the whole test split. Every model sees identical targets."""
    model.eval()
    n_windows = (len(tokens) - 1) // block_size
    total_nll = 0.0
    total_tokens = 0
    total_bytes = 0
    for start in range(0, n_windows, BATCH_SIZE):
        idx = np.arange(start, min(start + BATCH_SIZE, n_windows)) * block_size
        x = np.stack([tokens[i:i + block_size] for i in idx]).astype(np.int64)
        y = np.stack([tokens[i + 1:i + block_size + 1] for i in idx]).astype(np.int64)
        xt = torch.from_numpy(x).to(device)
        yt = torch.from_numpy(y).to(device)
        _, loss = model(xt, yt)
        total_nll += loss.item() * yt.numel()
        total_tokens += yt.numel()
        total_bytes += int(bytes_table[y].sum())
    mean_loss = total_nll / total_tokens
    bpb = (mean_loss / math.log(2)) * (total_tokens / total_bytes)
    return mean_loss, math.exp(mean_loss), bpb, total_tokens, total_bytes


@torch.no_grad()
def generate_stories(model, enc, device):
    out = []
    for prompt in PROMPTS:
        torch.manual_seed(GEN_SEED)
        ids = torch.tensor([enc.encode_ordinary(prompt)], dtype=torch.long, device=device)
        gen = model.generate(ids, GEN_TOKENS, temperature=GEN_TEMPERATURE, top_k=GEN_TOP_K)[0].tolist()
        if enc.eot_token in gen[len(ids[0]):]:
            gen = gen[:len(ids[0]) + gen[len(ids[0]):].index(enc.eot_token)]
        out.append((prompt, enc.decode(gen)))
    return out


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model = GPT(GPTConfig(**ckpt["model_config"]))
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    meta = {"model_config": ckpt["model_config"], "seed": ckpt["seed"],
            "tokens_processed": ckpt.get("tokens_processed"),
            "elapsed_seconds": ckpt.get("elapsed_seconds")}
    del ckpt
    return model, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=ALL_RUNS)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--keep-checkpoints", action="store_true")
    ap.add_argument("--skip-generation", action="store_true")
    args = ap.parse_args()

    device = pick_device(args.device)
    print(f"Device: {device}")
    enc = tiktoken.get_encoding("gpt2")
    bytes_table = token_bytes_table(enc)
    test_tokens = build_test_tokens(Path("data/test_tokens.bin"))

    results_dir = Path("results")
    analysis_dir = Path("analysis")
    analysis_dir.mkdir(exist_ok=True)
    hf_dir = Path("hf_cache")
    rows = []

    for run in args.runs:
        out = results_dir / run
        out.mkdir(parents=True, exist_ok=True)
        res_path = out / "test_results.json"
        for name in ("run_config.json", "metrics.csv"):
            if not (out / name).exists():
                p = hf_hub_download(REPO_ID, f"{run}/{name}", local_dir=hf_dir)
                os.replace(p, out / name)
        if res_path.exists() and (out / "generations.md").exists():
            print(f"[{run}] already done, skipping")
            rows.append(json.loads(res_path.read_text()))
            continue

        print(f"[{run}] downloading checkpoint...")
        ckpt_path = hf_hub_download(REPO_ID, f"{run}/checkpoint.pt", local_dir=hf_dir)
        model, meta = load_model(ckpt_path, device)
        block = meta["model_config"]["block_size"]
        n_params = sum(p.numel() for p in model.parameters())

        print(f"[{run}] evaluating on full test split ({n_params/1e6:.1f}M params)...")
        loss, ppl, bpb, n_tok, n_bytes = evaluate_full(model, test_tokens, block, device, bytes_table)
        result = {"run": run, "test_loss": round(loss, 6), "test_perplexity": round(ppl, 4),
                  "test_bits_per_byte": round(bpb, 6), "evaluated_tokens": n_tok,
                  "evaluated_bytes": n_bytes, "parameter_count": n_params,
                  "tokens_processed": meta["tokens_processed"], "seed": meta["seed"],
                  "protocol": "non-overlapping 256-token windows over the whole test split"}
        res_path.write_text(json.dumps(result, indent=2) + "\n")
        print(f"[{run}] test loss {loss:.4f}  ppl {ppl:.2f}  bpb {bpb:.4f}  ({n_tok:,} tokens)")

        if not args.skip_generation:
            stories = generate_stories(model, enc, device)
            with (out / "generations.md").open("w") as f:
                f.write(f"# {run}\n\ntemperature {GEN_TEMPERATURE}, top-k {GEN_TOP_K}, "
                        f"max {GEN_TOKENS} new tokens, seed {GEN_SEED}\n\n")
                for prompt, text in stories:
                    f.write(f"## Prompt: {prompt}\n\n{text}\n\n")
        rows.append(result)

        del model
        if device.type == "mps":
            torch.mps.empty_cache()
        if not args.keep_checkpoints:
            os.remove(ckpt_path)

    with (analysis_dir / "test_summary.csv").open("w", newline="") as f:
        fields = ["run", "parameter_count", "tokens_processed", "test_loss",
                  "test_perplexity", "test_bits_per_byte", "evaluated_tokens", "evaluated_bytes"]
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda r: r["test_loss"]):
            w.writerow(r)
    with (analysis_dir / "generations_all.md").open("w") as f:
        for r in rows:
            g = results_dir / r["run"] / "generations.md"
            if g.exists():
                f.write(g.read_text() + "\n---\n\n")
    print("\nDone. See analysis/test_summary.csv and analysis/generations_all.md")


if __name__ == "__main__":
    main()
