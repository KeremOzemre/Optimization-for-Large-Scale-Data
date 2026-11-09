"""Generate reproducible samples and a blind two-rater scoring sheet."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import tiktoken
import torch
import yaml

from train import file_sha256, load_checkpoint_model, select_device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--prompts", default="configs/followups/prompts.yaml")
    parser.add_argument("--seeds", nargs="+", type=int, default=[101, 102])
    parser.add_argument("--max-new-tokens", type=int, default=150)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if args.max_new_tokens < 1 or args.temperature <= 0 or args.top_k < 1:
        parser.error("Generation length, temperature, and top-k must be positive")
    prompts = yaml.safe_load(Path(args.prompts).read_text())
    if not isinstance(prompts, list) or not prompts or any(not isinstance(p, str) or not p for p in prompts):
        parser.error("Prompts must be a nonempty YAML list of nonempty strings")
    out = Path(args.output_dir)
    if out.exists():
        parser.error("Choose a new output directory")
    device = select_device(args.device)
    tokenizer = tiktoken.get_encoding("gpt2")
    records = []
    paths = [Path(path) for path in args.checkpoints]
    if len({str(path.resolve()) for path in paths}) != len(paths) or len(set(args.seeds)) != len(args.seeds):
        parser.error("Checkpoints and seeds must be distinct")
    out.mkdir(parents=True)
    for path in paths:
        model, checkpoint = load_checkpoint_model(path, device)
        model_hash = file_sha256(path)
        del checkpoint
        for prompt_index, prompt in enumerate(prompts):
            prompt_ids = tokenizer.encode_ordinary(prompt)
            for seed in args.seeds:
                torch.manual_seed(seed)
                x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
                ids = model.generate(x, args.max_new_tokens, args.temperature, args.top_k,
                                     eos_token_id=tokenizer.eot_token)[0].tolist()[len(prompt_ids):]
                stopped = ids[-1] == tokenizer.eot_token
                continuation = tokenizer.decode(ids[:-1] if stopped else ids)
                sample_id = hashlib.sha256(f"{model_hash}:{prompt_index}:{seed}".encode()).hexdigest()[:16]
                records.append({"sample_id": sample_id, "checkpoint": str(path),
                                "checkpoint_sha256": model_hash, "prompt_index": prompt_index,
                                "prompt": prompt, "seed": seed, "continuation": continuation,
                                "new_tokens": len(ids), "termination": "eot" if stopped else "length_limit",
                                "prompt_tokens": len(prompt_ids),
                                "prompt_truncated": len(prompt_ids) > model.config.block_size})
        del model
    (out / "samples.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
    (out / "protocol.json").write_text(json.dumps({
        "prompts_sha256": file_sha256(args.prompts), "seeds": args.seeds,
        "temperature": args.temperature, "top_k": args.top_k, "max_new_tokens": args.max_new_tokens,
        "torch": str(torch.__version__), "device": str(device)}, indent=2) + "\n")
    # Hash order hides architecture and keeps the display order reproducible.
    blind = sorted(records, key=lambda row: row["sample_id"])
    (out / "blind_samples.md").write_text("# Blind story samples\n\n" + "\n".join(
        f"## {row['sample_id']}\n\nPrompt: {row['prompt']}\n\n{row['continuation']}\n" for row in blind))
    with (out / "ratings.csv").open("w", newline="") as handle:
        fields = ["sample_id", "rater", "grammar_1_to_5", "plot_1_to_5", "consistency_1_to_5",
                  "repetition_1_to_5", "notes"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({"sample_id": row["sample_id"], "rater": rater}
                         for row in blind for rater in ("A", "B"))
    print(f"Saved {len(records)} samples. Give raters only blind_samples.md and ratings.csv.")


if __name__ == "__main__":
    main()
