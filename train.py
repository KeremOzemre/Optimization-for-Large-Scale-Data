"""Train a small GPT from scratch on TinyStories.

The model, batch sampling, and optimizer setup adapt nanoGPT's model.py and
train.py at commit 3adf61e154c3fe3fca428ad6bc3818b27a3b8291 (MIT).
"""

from contextlib import nullcontext
import csv
from dataclasses import asdict
import json
import math
import os
import platform
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

import numpy as np
import tiktoken
import torch

from model import GPT, GPTConfig


CSV_FIELDS = [
    "step", "elapsed_seconds", "train_loss", "val_loss", "val_perplexity",
    "learning_rate", "tokens_processed", "tokens_per_second", "phase",
]


def select_device(requested="auto"):
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("Apple MPS is not available")
    return torch.device(requested)


def resolve_precision(device, requested):
    if requested == "auto":
        return "bfloat16" if device.type == "cuda" and torch.cuda.is_bf16_supported() else "float32"
    if requested == "bfloat16" and not (device.type == "cuda" and torch.cuda.is_bf16_supported()):
        raise ValueError("bfloat16 precision requires a CUDA device that supports it")
    return requested


def hardware_name(device):
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    if device.type == "mps":
        try:
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    return platform.processor() or platform.machine()


def autocast_context(device, precision="auto"):
    if resolve_precision(device, precision) == "bfloat16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()  # float32 on MPS, CPU, and older CUDA GPUs


def load_tokens(data_dir, split, block_size):
    path = Path(data_dir) / f"{split}.bin"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run prepare.py first")
    tokens = np.memmap(path, dtype=np.uint16, mode="r")
    if len(tokens) <= block_size:
        raise ValueError(f"{path} needs at least {block_size + 1} tokens")
    return tokens


def batch_at(tokens, starts, block_size, device):
    """Input positions [i:i+T] predict targets [i+1:i+T+1]."""
    x = np.stack([tokens[i:i + block_size] for i in starts]).astype(np.int64)
    y = np.stack([tokens[i + 1:i + block_size + 1] for i in starts]).astype(np.int64)
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


def random_batch(tokens, batch_size, block_size, device, rng):
    starts = rng.integers(0, len(tokens) - block_size, size=batch_size)
    return batch_at(tokens, starts, block_size, device)


def fixed_batches(tokens, batch_size, block_size, count, seed):
    """Choose held-out windows once; every validation uses these same offsets."""
    rng = np.random.default_rng(seed)
    return [rng.integers(0, len(tokens) - block_size, size=batch_size)
            for _ in range(count)]


def evaluate_model(model, tokens, batches, block_size, device, precision="auto"):
    was_training = model.training
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    with torch.inference_mode():
        for starts in batches:
            x, y = batch_at(tokens, starts, block_size, device)
            with autocast_context(device, precision):
                _, loss = model(x, y)
            total_nll += loss.item() * y.numel()
            total_tokens += y.numel()
    model.train(was_training)
    mean_loss = total_nll / total_tokens
    return mean_loss, math.exp(mean_loss), total_tokens


def learning_rate(elapsed, time_limit, peak_lr, warmup_fraction, min_lr_ratio):
    """Time-based warmup followed by cosine decay to a fraction of peak LR."""
    minimum = peak_lr * min_lr_ratio
    warmup = time_limit * warmup_fraction
    if elapsed < warmup:
        return minimum + (peak_lr - minimum) * elapsed / warmup
    progress = min(1.0, (elapsed - warmup) / (time_limit - warmup))
    return minimum + 0.5 * (peak_lr - minimum) * (1 + math.cos(math.pi * progress))


def load_checkpoint_model(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = GPT(GPTConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model, checkpoint


def train_model(settings):
    """Train one run using the validated YAML settings from run.py."""
    from prepare import prepared_cache_is_valid

    experiment = settings["experiment"]
    training = settings["training"]
    optimizer_settings = settings["optimizer"]
    schedule = settings["schedule"]
    evaluation = settings["evaluation"]
    model_settings = settings["model"]
    args = SimpleNamespace(
        data_dir=settings["data"]["cache_dir"], out_dir=experiment["output_dir"],
        n_layer=model_settings["layers"], n_embd=model_settings["width"],
        n_head=model_settings["heads"], block_size=model_settings["context_length"],
        batch_size=training["microbatch_size"],
        grad_accum_steps=training["gradient_accumulation_steps"],
        dropout=model_settings["dropout"], learning_rate=optimizer_settings["learning_rate"],
        weight_decay=optimizer_settings["weight_decay"],
        grad_clip=optimizer_settings["gradient_clip"], seed=experiment["seed"],
        time_limit_seconds=training["time_limit_seconds"],
        eval_interval_seconds=evaluation["interval_seconds"],
        eval_batches=evaluation["batches"],
        log_interval_steps=training["log_interval_steps"],
        device=training["device"], precision=training["precision"],
    )
    out_dir = Path(args.out_dir)
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} already exists; choose a new experiment.output_dir")
    if not prepared_cache_is_valid(settings):
        raise ValueError(f"{args.data_dir} is missing or incompatible; run prepare with this configuration")

    data_dir = Path(args.data_dir)
    metadata_path = data_dir / "meta.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing {metadata_path}; run prepare.py first")
    metadata = json.loads(metadata_path.read_text())
    tokenizer = tiktoken.get_encoding(settings["tokenizer"]["name"])
    if metadata.get("tokenizer") != tokenizer.name or metadata.get("vocab_size") != tokenizer.n_vocab:
        raise ValueError("Token cache does not match the configured tokenizer")

    device = select_device(args.device)
    precision = resolve_precision(device, args.precision)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    train_tokens = load_tokens(data_dir, "train", args.block_size)
    val_tokens = load_tokens(data_dir, "val", args.block_size)
    val_batches = fixed_batches(val_tokens, args.batch_size, args.block_size,
                                args.eval_batches, args.seed + 1)

    config = GPTConfig(block_size=args.block_size, vocab_size=tokenizer.n_vocab,
                       n_layer=args.n_layer, n_head=args.n_head,
                       n_embd=args.n_embd, dropout=args.dropout, bias=False)
    model = GPT(config).to(device)  # Fresh random weights; no pretrained load path.
    optimizer = model.configure_optimizers(args.weight_decay, args.learning_rate,
                                            tuple(optimizer_settings["betas"]), device.type)
    out_dir.mkdir(parents=True, exist_ok=False)
    parameter_count = sum(p.numel() for p in model.parameters())
    run_config = {**settings, "runtime": {
        "device": device.type,
        "hardware": hardware_name(device),
        "precision": precision, "vocab_size": tokenizer.n_vocab,
        "parameter_count": parameter_count,
        "prepared_token_counts": metadata["token_counts"],
    }}
    (out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2) + "\n")
    print(f"Training {parameter_count:,} parameters on {device} with {precision}; output: {out_dir}", flush=True)

    step = 0
    tokens_processed = 0
    next_eval = args.eval_interval_seconds
    last_train_loss = float("nan")
    current_lr = args.learning_rate
    # Preparation, model creation, and optimizer creation are outside this timer.
    loop_start = time.perf_counter()
    with (out_dir / "metrics.csv").open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        while True:
            elapsed = time.perf_counter() - loop_start
            if step and elapsed >= args.time_limit_seconds:
                break
            current_lr = learning_rate(
                elapsed, args.time_limit_seconds, args.learning_rate,
                schedule["warmup_fraction"], schedule["min_lr_ratio"])
            for group in optimizer.param_groups:
                group["lr"] = current_lr
            optimizer.zero_grad(set_to_none=True)
            loss_sum = None
            for _ in range(args.grad_accum_steps):
                x, y = random_batch(train_tokens, args.batch_size, args.block_size,
                                    device, rng)
                with autocast_context(device, precision):
                    _, loss = model(x, y)
                (loss / args.grad_accum_steps).backward()
                loss_sum = loss.detach() if loss_sum is None else loss_sum + loss.detach()
            if args.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            step += 1
            tokens_processed += args.batch_size * args.block_size * args.grad_accum_steps
            last_train_loss = (loss_sum / args.grad_accum_steps).item()
            elapsed = time.perf_counter() - loop_start
            if not math.isfinite(last_train_loss):
                raise RuntimeError(f"Non-finite training loss at step {step}")

            val_loss = val_ppl = None
            if elapsed < args.time_limit_seconds and elapsed >= next_eval:
                val_loss, val_ppl, _ = evaluate_model(
                    model, val_tokens, val_batches, args.block_size, device, precision)
                if not math.isfinite(val_loss):
                    raise RuntimeError("Non-finite validation loss")
                while next_eval <= time.perf_counter() - loop_start:
                    next_eval += args.eval_interval_seconds
            elapsed = time.perf_counter() - loop_start
            if (step % args.log_interval_steps == 0 or val_loss is not None
                    or elapsed >= args.time_limit_seconds):
                row = {
                    "step": step, "elapsed_seconds": round(elapsed, 3),
                    "train_loss": last_train_loss, "val_loss": val_loss,
                    "val_perplexity": val_ppl, "learning_rate": current_lr,
                    "tokens_processed": tokens_processed,
                    "tokens_per_second": tokens_processed / max(elapsed, 1e-9),
                    "phase": "validation" if val_loss is not None else "train",
                }
                writer.writerow(row)
                csv_file.flush()
                print(row, flush=True)
            if time.perf_counter() - loop_start >= args.time_limit_seconds:
                break
    training_elapsed = time.perf_counter() - loop_start

    eval_start = time.perf_counter()
    final_loss, final_ppl, _ = evaluate_model(model, val_tokens, val_batches,
                                               args.block_size, device, precision)
    final_eval_seconds = time.perf_counter() - eval_start
    if not math.isfinite(final_loss):
        raise RuntimeError("Non-finite final validation loss")
    with (out_dir / "metrics.csv").open("a", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writerow({
            "step": step, "elapsed_seconds": round(training_elapsed, 3),
            "train_loss": last_train_loss, "val_loss": final_loss,
            "val_perplexity": final_ppl, "learning_rate": current_lr,
            "tokens_processed": tokens_processed,
            "tokens_per_second": tokens_processed / max(training_elapsed, 1e-9),
            "phase": "final",
        })
    print(f"Final validation: loss={final_loss:.4f}, perplexity={final_ppl:.2f}; "
          f"evaluation took {final_eval_seconds:.2f}s", flush=True)

    save_start = time.perf_counter()
    checkpoint = {
        "model": model.state_dict(), "model_config": asdict(config),
        "optimizer": optimizer.state_dict(), "seed": args.seed,
        "elapsed_seconds": training_elapsed, "tokens_processed": tokens_processed,
        "run_config": run_config,
    }
    temporary = out_dir / "checkpoint.pt.tmp"
    torch.save(checkpoint, temporary)
    os.replace(temporary, out_dir / "checkpoint.pt")
    save_seconds = time.perf_counter() - save_start
    print(f"Training: {step} steps, {tokens_processed:,} tokens in {training_elapsed:.2f}s. "
          f"Checkpoint: {out_dir / 'checkpoint.pt'} (save took {save_seconds:.2f}s)", flush=True)


if __name__ == "__main__":
    from run import main
    import sys
    main(["train", *sys.argv[1:]])
