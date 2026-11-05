# Project context

## Purpose

Four students are training a small GPT language model from random weights on TinyStories. The initial research question is: **Which model size and training configuration gives the best language-modeling performance under approximately one hour on the same hardware?** The first priority is a reliable, reproducible pipeline.

## Confirmed decisions and defaults

- Start from nanoGPT's GPT architecture at commit `3adf61e154c3fe3fca428ad6bc3818b27a3b8291`; retain its MIT license, initialization, causal attention, and tied embeddings. No pretrained weights, distributed training, or compilation.
- Use `roneneldan/TinyStories`: official training split for training, with the official validation split divided 80/20 into validation and test at the story level using seed 42. Tokenize once with `tiktoken` GPT-2, append end-of-text to each story, and cache `uint16` token streams. The vocabulary size comes from the tokenizer.
- Default model: 4 layers, width 256, 4 heads, 256-token context, dropout 0. Width must divide evenly by head count.
- Default training: one device selected CUDA → MPS → CPU; bfloat16 on supported CUDA, otherwise float32. AdamW with learning rate 0.0003, weight decay 0.1, betas (0.9, 0.95), clipping 1.0, microbatch 16, accumulation 1. Warm up for 2% of the time budget, then cosine-decay to 10% of peak learning rate.
- Default budget: 3,600 seconds of wall-clock time inside the training loop, including periodic validation and logging. Finish the current optimizer step, then do final validation and checkpoint saving outside the budget. Validation uses fixed seeded windows about every 300 seconds. Test data is for post-training evaluation only.

## Workflow and code

See [CONFIGURATION.md](CONFIGURATION.md) for all configuration keys, allowed values, defaults, and explanations of their effects.

`configs/default.yaml` is the main configuration for an experiment; copy it and change its name and unused output directory for each new run. `configs/smoke.yaml` is a short, small-data example. `run.py` is the user-facing entry point:

```bash
python run.py prepare --config configs/default.yaml
python run.py train --config configs/default.yaml
python run.py evaluate --config configs/default.yaml
python run.py generate --config configs/default.yaml
```

`python run.py run --config <file>` prepares missing compatible data, trains, then evaluates on test data. Use the separate `prepare` and `train` actions while selecting models by validation; evaluate on test after selection. Preparation settings must match the token cache; an incompatible cache is rejected unless rebuilt explicitly with `--force`. A training run refuses an existing output directory. `prepare.py` owns data loading and tokenization; `model.py` owns the nanoGPT model; `train.py` owns the time-limited optimizer loop and checkpoint; `evaluate.py` evaluates test windows; `generate.py` samples text. `prepare_data.py` remains a compatibility entry point for preparation. Run outputs are `run_config.json`, `metrics.csv`, and `checkpoint.pt` in the configured output directory.

## Experiment rules

For the first model-size comparisons, use the same prepared data, tokenizer, seed, hardware, time budget, optimizer and schedule, batch settings, and fixed validation procedure. Change model layers, width, and heads, with a distinct output directory for each run. Record the actual parameter count and processed training tokens: equal time does not imply equal token exposure. Compare held-out cross-entropy and perplexity only when the tokenizer is the same. Reserve test results for after choosing a model using validation. Generated-story quality can supplement, but not replace, the initial held-out metrics.

## Current status and limitations

The central `run` path was smoke-tested on Apple MPS: fresh small-data preparation, compatible cache reuse, training with finite loss, periodic/final validation, checkpoint loading, test evaluation, and generation worked. Changing model width in the YAML changed the saved model configuration and parameter count; existing output directories and incompatible caches were rejected. Correct target shifting, causal masking, weight tying, and tiny-batch overfitting have focused tests. The full token cache and one-hour run have not been executed; CUDA and CPU end-to-end runs have not been checked here. Evaluation samples a fixed number of token windows rather than exhaustively scoring every held-out token. Training samples random windows with replacement, so one hour does not guarantee an epoch or unique coverage. The upstream dataset revision is not pinned; cached tokens are reproducible for the recorded preparation settings, but a future fresh download could change if the source changes.

## Possible later experiments and open questions

Potential extensions include context length, depth versus width, data selection, tokenizer/vocabulary changes, story quality, and controlled context-tracking tasks. These are **proposals**, not features in the current pipeline. Token-level losses and perplexities are not directly comparable across tokenizers. Open choices include which model sizes to compare, which common hardware to use, and whether a fixed shared optimizer recipe is sufficient for all sizes.

## Next steps

1. Review the baseline configuration and prepare the full token cache.
2. Copy the baseline configuration for each model size, changing its experiment name and output directory.
3. Run separate one-hour experiments on the same hardware, compare validation loss alongside token throughput, then evaluate the selected model on test data.
