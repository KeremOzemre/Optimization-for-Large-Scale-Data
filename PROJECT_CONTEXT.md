# Project context

## Purpose

Four students are training a small GPT language model from random weights on TinyStories. The initial research question is: **Which model size and training configuration gives the best language-modeling performance under approximately one hour on the same hardware?** The first priority is a reliable, reproducible pipeline.

## Confirmed decisions and defaults

- Start from nanoGPT's GPT architecture at commit `3adf61e154c3fe3fca428ad6bc3818b27a3b8291`; retain its MIT license, initialization, causal attention, and tied embeddings. No pretrained weights, distributed training, or compilation.
- Require PyTorch >=2.0 and use its scaled dot-product attention directly. Training reports and saves the total parameter count, including position embeddings, in one place.
- Use `roneneldan/TinyStories`: official training split for training, with the official validation split divided 80/20 into validation and test at the story level using seed 42. Tokenize once with `tiktoken` GPT-2, append end-of-text to each story, and cache `uint16` token streams. The vocabulary size comes from the tokenizer.
- Default model: 4 layers, width 256, 4 heads, 256-token context, dropout 0. Width must divide evenly by head count.
- Default training: one device selected CUDA → MPS → CPU; bfloat16 on supported CUDA, otherwise float32. AdamW with learning rate 0.0003, weight decay 0.1, betas (0.9, 0.95), clipping 1.0, microbatch 16, accumulation 1. Warm up for 2% of the time budget, then cosine-decay to 10% of peak learning rate.
- Default budget: 3,600 seconds of wall-clock time inside the training loop, including periodic validation and logging. Finish the current optimizer step, then do final validation and checkpoint saving outside the budget. Validation uses fixed seeded windows about every 300 seconds. Test data is for post-training evaluation only.

## Workflow and code

See [CONFIGURATION.md](CONFIGURATION.md) for all configuration keys, allowed values, defaults, and explanations of their effects.

See [analysis/token_exposure_and_epoch_estimates.md](analysis/token_exposure_and_epoch_estimates.md) for the saved-run token audit and measured-throughput estimates. Dataset-equivalent exposure counts repeated targets; it does not verify a full pass over unique positions.

See [NEXT_STEPS_GUIDE.md](NEXT_STEPS_GUIDE.md) for the proposed validation audit, repeat trials, longer-budget and equal-token comparisons, and capability checks. Validated one-/four-hour XL/XXL configs in `configs/followups/` have not been run; they equalize sampled validation to 65,536 targets while retaining the original effective training batch.

`configs/default.yaml` is the main configuration for an experiment; copy it and change its name and unused output directory for each new run. `configs/smoke.yaml` is a short, small-data example. `run.py` is the user-facing entry point:

```bash
python run.py prepare --config configs/default.yaml
python run.py train --config configs/default.yaml
python run.py evaluate --config configs/default.yaml
python run.py generate --config configs/default.yaml
```

`python run.py run --config <file>` prepares missing compatible data, trains, then evaluates on test data. Use the separate `prepare` and `train` actions while selecting models by validation; evaluate on test after selection. Preparation settings must match the token cache; an incompatible cache is rejected unless rebuilt explicitly with `--force`. A training run refuses an existing output directory. `prepare.py` owns data loading and tokenization; `model.py` owns the nanoGPT model; `train.py` owns the time-limited optimizer loop and checkpoint; `evaluate.py` evaluates test windows; `generate.py` samples text. Preparation uses `run.py prepare`; the redundant `prepare_data.py` entry point has been removed. Run outputs are `run_config.json`, `metrics.csv`, and `checkpoint.pt` in the configured output directory.

## Experiment rules

For the first model-size comparisons, use the same prepared data, tokenizer, seed, hardware, time budget, optimizer and schedule, batch settings, and fixed validation procedure. Change model layers, width, and heads, with a distinct output directory for each run. Record the actual parameter count and processed training tokens: equal time does not imply equal token exposure. Compare held-out cross-entropy and perplexity only when the tokenizer is the same. Reserve test results for after choosing a model using validation. Generated-story quality can supplement, but not replace, the initial held-out metrics.

## Current status and limitations

The central pipeline was smoke-tested on Apple MPS, including preparation, training, evaluation, checkpoint loading, and generation. Focused tests cover shifted targets, causal masking, weight tying, and tiny-batch overfitting. The full token cache is prepared. The `results/` archive contains logs, saved configurations, and test-result files for 20 one-hour CUDA runs: five model families (small through XXL) across 25/50/75/100% story subsets, on an NVIDIA H100 NVL MIG 3g.47gb partition using bfloat16. Checkpoints are not included locally, so these archived checkpoints cannot be independently re-evaluated here. All 262,432 logged token counters match the completed-step formula.

On the full cache of 473,992,236 tokens, one-hour exposure ranges from 2.077 dataset-equivalents for small to 0.303 for XXL. The measured-throughput estimate for XXL to process one full-cache-equivalent is about 3h 18m; this is not a guarantee of a complete epoch. Training still samples windows with replacement and has no epoch stop or resume interface. Validation samples fixed windows, and only one training seed is recorded per configuration. The shared seed also controls the data split. The dataset revision is not pinned.

The supported conclusion is the best observed configuration under the fixed-time shared recipe: XL has the lowest recorded validation loss among the tested families. A universal claim that larger models are better, or that all models were trained to convergence or equal exposure, is unsupported. A four-hour follow-up budget is a proposal only; the confirmed default remains one hour.

Follow-up audit findings: original sampled validation sizes differ by family (Large 65,536, XL 32,768, XXL 16,384 targets); compare saved finalists on common validation targets before interpreting close differences. Existing test exports use common non-overlapping target windows. The additional 30-minute/context runs under `runs/` record `NVIDIA H100 NVL`, while the main grid records an H100 MIG partition; hardware comparability needs reconciliation. Report claims about insufficient token volume, automatic XXL overtaking, and an isolated budget effect need qualification as described in the guide.

## Possible later experiments and open questions

Potential extensions include context length, depth versus width, data selection, tokenizer/vocabulary changes, story quality, and controlled context-tracking tasks. These are **proposals**, not features in the current pipeline. Token-level losses and perplexities are not directly comparable across tokenizers. Open choices include which model sizes to compare, which common hardware to use, and whether a fixed shared optimizer recipe is sufficient for all sizes.

## Next steps

1. Reconcile the report claims and hardware labels, then compare saved finalists on identical validation targets using the commands in `NEXT_STEPS_GUIDE.md`.
2. Check the one-hour ranking's reproducibility; separate preparation and evaluation seeds from training randomness before independent repeat trials.
3. Choose a longer shared-time or equal-token follow-up. Equal-token stopping and guaranteed epochs require explicit implementation; increasing a time limit alone cannot guarantee coverage.
