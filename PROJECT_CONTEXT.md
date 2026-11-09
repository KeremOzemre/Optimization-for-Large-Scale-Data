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

See [FOLLOWUP_COMMANDS.md](FOLLOWUP_COMMANDS.md) for runnable commands, budgets, and return files; [NEXT_STEPS_GUIDE.md](NEXT_STEPS_GUIDE.md) explains the reasoning. Ten validated XL/XXL YAMLs cover three one-hour seeds per family, a four-hour pair, and an equal-token pair. They share `data/tokens-full`, preparation seed 42, evaluation seed 43, 65,536 validation targets, and 4,096 training targets/update. None has been trained.

`configs/default.yaml` is the main configuration for an experiment; copy it and change its name and unused output directory for each new run. `configs/smoke.yaml` is a short, small-data example. `run.py` is the user-facing entry point:

```bash
python run.py prepare --config configs/default.yaml
python run.py train --config configs/default.yaml
python run.py evaluate --config configs/default.yaml
python run.py generate --config configs/default.yaml
```

`python run.py run --config <file>` prepares missing compatible data, trains, then evaluates on test data. Use the separate `prepare` and `train` actions while selecting models by validation; evaluate on test after selection. Preparation settings must match the token cache; an incompatible cache is rejected unless rebuilt explicitly with `--force`. A training run refuses an existing output directory. `prepare.py` owns data loading and tokenization; `model.py` owns the nanoGPT model; `train.py` owns the time-limited optimizer loop and checkpoint; `evaluate.py` evaluates test windows; `generate.py` samples text. Preparation uses `run.py prepare`; the redundant `prepare_data.py` entry point has been removed. Run outputs are `run_config.json`, `metrics.csv`, `checkpoint.pt`, and `training_summary.json` in the configured output directory.

## Experiment rules

The follow-up schema adds `data.preparation_seed`, `training.max_steps`, `schedule.progress` (`time`/`steps`), `evaluation.microbatch_size`, `evaluation.seed`, and `generation.seed`. Omitted fields preserve legacy YAML defaults. Equal-token runs request 115,721 updates (473,993,216 targets), step-based warmup/cosine, and a six-hour safety cap. A run stopped early by time is incomplete. `training_summary.json` records steps/targets, stop reason, and separate training/evaluation/save times. New training records code/cache SHA-256 hashes and environment versions before the timer. Generation has an independent seed and EOT stopping.

`score_checkpoints.py` scores identical validation targets (full stream including tail by default, sampled mode available), saving per-story losses and paired story-bootstrap diagnostics; test mode is explicit. `summarize_followups.py` reports all seeds, mean/SD, paired differences, and comparability warnings. `sample_stories.py` creates frozen 20-prompt/two-seed blind rating material; `context_checks.py` supplies ten paired color-swap scenarios. Story intervals condition on fixed checkpoints and do not measure training-seed variability. Human ratings and all full follow-up evaluations remain pending.

For the first model-size comparisons, use the same prepared data, tokenizer, seed, hardware, time budget, optimizer and schedule, batch settings, and fixed validation procedure. Change model layers, width, and heads, with a distinct output directory for each run. Record the actual parameter count and processed training tokens: equal time does not imply equal token exposure. Compare held-out cross-entropy and perplexity only when the tokenizer is the same. Reserve test results for after choosing a model using validation. Generated-story quality can supplement, but not replace, the initial held-out metrics.

## Current status and limitations

The original pipeline was smoke-tested on Apple MPS. Follow-up setup passed eleven CPU checks and synthetic end-to-end checks of all four new scoring/generation/summary CLIs; no long training or full research evaluation was launched. The full cache is prepared. The `results/` archive contains 20 one-hour CUDA runs (five families × four fractions) on an NVIDIA H100 NVL MIG 3g.47gb partition using bfloat16. All 262,432 logged token counters match the completed-step formula.

Four archived checkpoints plus logs/configs are downloaded into ignored `hf_cache/`: Large/full, XL/50%, XL/full, XXL/full. Hub revision `fa18888b1b9b599e9deb455ca740896e7d0642f3` and SHA-256 hashes are recorded in `analysis/followups/checkpoint_manifest.json`; `download_checkpoints.py` reproduces/reuses the downloads. All four passed CPU loading, finite-forward/generation checks, and an eight-target CLI scoring smoke test with temporary outputs deleted. They are evaluation baselines, not initializations for new training. The report PDF was rebuilt and visually checked; substantive follow-up scores remain pending.

On the full cache of 473,992,236 tokens, original one-hour exposure ranges from 2.077 equivalents for small to 0.303 for XXL. XXL's measured-throughput estimate for one equivalent is about 3h 18m; this does not guarantee an epoch. Training samples with replacement and has no without-replacement epoch or resume interface. Only one seed exists per original configuration. Follow-up seeds are now separated, but the dataset revision remains unpinned and old runs lack cache hashes.

The supported conclusion is the best observed configuration under the fixed-time shared recipe: XL has the lowest recorded validation loss among the tested families. A universal claim that larger models are better, or that all models were trained to convergence or equal exposure, is unsupported. A four-hour follow-up budget is a proposal only; the confirmed default remains one hour.

Original validation sizes differ by family (Large 65,536, XL 32,768, XXL 16,384 targets); common-target re-evaluation is pending. Existing test exports use common non-overlapping windows, dropping the tail. Additional 30-minute/context runs record `NVIDIA H100 NVL`, while the grid records an H100 MIG partition; hardware comparability remains unresolved. Report sources now correct the token-volume/overtaking claims, qualify budget/context causality, disclose test reuse, and label follow-ups pending. Updated fit/budget plots fix the data fraction at 100% and carry caveats.

## Possible later experiments and open questions

Broader extensions include controlled context-length comparisons, depth versus width, data selection, and tokenizer/vocabulary changes; these remain proposals. Narrow color-tracking probes and blind story-rating material are now implemented, with evaluations pending. Token-level losses and perplexities are not directly comparable across tokenizers. Hardware allocation and optimizer tuning per size remain open experimental choices.

## Next steps

1. User runs the common validation audit and six one-hour seed trials from `FOLLOWUP_COMMANDS.md`, then returns logs/configs/common scores. Confirm matching GPU allocation.
2. Interpret common scores and training-seed variability before choosing the prepared four-hour or equal-token pair.
3. Complete blind ratings/context probes and update the report from actual evidence. All 20 original test scores were inspected; reused test evaluation is not fresh confirmation.
