"""Evaluate a trained checkpoint on the untouched TinyStories test split."""

import json
from pathlib import Path

from train import (evaluate_model, fixed_batches, load_checkpoint_model,
                   load_tokens, resolve_precision, select_device)


def evaluate_checkpoint(settings, checkpoint_path):
    device = select_device(settings["training"]["device"])
    precision = resolve_precision(device, settings["training"]["precision"])
    model, checkpoint = load_checkpoint_model(checkpoint_path, device)
    saved_config = checkpoint["run_config"]
    if "data" in saved_config:
        from prepare import prepared_cache_is_valid
        if not prepared_cache_is_valid(saved_config):
            raise ValueError("The checkpoint's test token cache is missing or incompatible")
        data_dir = saved_config["data"]["cache_dir"]
    else:
        data_dir = saved_config["data_dir"]  # Checkpoints from the earlier CLI pipeline.
    block_size = checkpoint["model_config"]["block_size"]
    test_tokens = load_tokens(Path(data_dir), "test", block_size)
    batches = fixed_batches(test_tokens, settings["training"]["microbatch_size"],
                            block_size, settings["evaluation"]["batches"],
                            checkpoint["seed"] + 2)
    loss, perplexity, count = evaluate_model(model, test_tokens, batches,
                                              block_size, device, precision)
    print(f"Test loss: {loss:.4f}; perplexity: {perplexity:.2f}; "
          f"evaluated tokens: {count:,}")
    # Persist next to the checkpoint so analyze.py can pick up test numbers.
    results = {"test_loss": round(float(loss), 6),
               "test_perplexity": round(float(perplexity), 6),
               "evaluated_tokens": int(count)}
    (Path(checkpoint_path).parent / "test_results.json").write_text(
        json.dumps(results, indent=2) + "\n")
    return loss, perplexity


if __name__ == "__main__":
    from run import main
    import sys
    main(["evaluate", *sys.argv[1:]])
