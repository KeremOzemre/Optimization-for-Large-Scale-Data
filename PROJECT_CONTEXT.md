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

See [FOLLOWUP_COMMANDS.md](FOLLOWUP_COMMANDS.md) for the focused execution checklist and [NEXT_STEPS_GUIDE.md](NEXT_STEPS_GUIDE.md) for the reasoning. Approved order: common full-validation audit, six fresh one-hour XL/XXL seed trials, interpret, then optionally a short equal-token pair. Main training costs six GPU-hours; the optional pair adds about 1.5 at original rates. Four-hour configs remain deferred templates. All use `data/tokens-full`, preparation seed 42, evaluation seed 43, 65,536 in-loop validation targets, and 4,096 training targets/update. No research follow-up has been run.

`configs/default.yaml` is the main configuration for an experiment; copy it and change its name and unused output directory for each new run. `configs/smoke.yaml` is a short, small-data example. `run.py` is the user-facing entry point:

```bash
python run.py prepare --config configs/default.yaml
python run.py train --config configs/default.yaml
python run.py evaluate --config configs/default.yaml
python run.py generate --config configs/default.yaml
```

`python run.py run --config <file>` prepares missing compatible data, trains, then evaluates on test data. Use the separate `prepare` and `train` actions while selecting models by validation; evaluate on test after selection. Preparation settings must match the token cache; an incompatible cache is rejected unless rebuilt explicitly with `--force`. A training run refuses an existing output directory. `prepare.py` owns data loading and tokenization; `model.py` owns the nanoGPT model; `train.py` owns the time-limited optimizer loop and checkpoint; `evaluate.py` evaluates test windows; `generate.py` samples text. Preparation uses `run.py prepare`; the redundant `prepare_data.py` entry point has been removed. Run outputs are `run_config.json`, `metrics.csv`, `checkpoint.pt`, and `training_summary.json` in the configured output directory.

## Experiment rules

The follow-up schema adds `data.preparation_seed`, `training.max_steps`, `schedule.progress` (`time`/`steps`), `evaluation.microbatch_size`, `evaluation.seed`, and `generation.seed`. Omitted fields preserve legacy defaults. The revised equal-token pair requests 35,000 updates (143,360,000 targets), a step-based schedule (700 warmup updates), and a 7,200-second safety cap per model. Fresh root `runs/followups_equal_tokens_143m/` separates it from the superseded 474M proposal. A time-capped run is incomplete. `training_summary.json` records actual steps/targets, stop reason, and separate timing; new runs save code/cache hashes and environment versions before the timer. Generation has an independent seed and EOT stopping.

`score_checkpoints.py` scores common validation targets (full stream including tail by default), saving per-story losses and paired bootstrap diagnostics; test mode is explicit. `summarize_followups.py` reports all seeds, mean/SD, paired differences, and comparability warnings. Story intervals condition on fixed checkpoints and do not measure retraining variability. `sample_stories.py` supplies supplementary blind two-rater material. The old color probe allowed first-color copying (20/20); v2 balances queried entity, mention order, and colors (80 prompts/ten scenarios) and records protocol `color_tracking_v2_balanced`. Simple color/position heuristics score 50%; this narrow diagnostic remains lower priority and its scores are incompatible with v1. All research evaluations/ratings remain pending.

For the first model-size comparisons, use the same prepared data, tokenizer, seed, hardware, time budget, optimizer and schedule, batch settings, and fixed validation procedure. Change model layers, width, and heads, with a distinct output directory for each run. Record the actual parameter count and processed training tokens: equal time does not imply equal token exposure. Compare held-out cross-entropy and perplexity only when the tokenizer is the same. Reserve test results for after choosing a model using validation. Generated-story quality can supplement, but not replace, the initial held-out metrics.

## Current status and limitations

The original pipeline was smoke-tested on Apple MPS. Follow-up setup passed 19 CPU checks, including the revised budget/completion warnings and balanced-probe design. Synthetic CLI checks also passed; no long training or full research evaluation was launched. The full cache is prepared. The `results/` archive contains 20 one-hour CUDA runs (five families × four fractions) on an NVIDIA H100 NVL MIG 3g.47gb partition using bfloat16. All 262,432 logged token counters match the completed-step formula.

Four archived checkpoints plus logs/configs are downloaded into ignored `hf_cache/`: Large/full, XL/50%, XL/full, XXL/full. Hub revision `fa18888b1b9b599e9deb455ca740896e7d0642f3` and SHA-256 hashes are recorded in `analysis/followups/checkpoint_manifest.json`; `download_checkpoints.py` reproduces/reuses the downloads. All four passed CPU loading, finite-forward/generation checks, and an eight-target CLI scoring smoke test with temporary outputs deleted. They are evaluation baselines, not initializations for new training. The report PDF was rebuilt and visually checked; substantive follow-up scores remain pending.

On the full cache of 473,992,236 tokens, original one-hour exposure ranges from 2.077 equivalents for small to 0.303 for XXL. XXL's measured-throughput estimate for one equivalent is about 3h 18m; this does not guarantee an epoch. Training samples with replacement and has no without-replacement epoch or resume interface. Only one seed exists per original configuration. Follow-up seeds are now separated, but the dataset revision remains unpinned and old runs lack cache hashes.

The supported conclusion is the best observed configuration under the fixed-time shared recipe: XL has the lowest recorded validation loss among the tested families. Repeatability remains untested. Three new seeds are an initial variability check; an inconclusive ranking is an acceptable result. No per-size optimizer search was done. Universal size, convergence, or equal-exposure claims are unsupported. Four-hour runs are deferred; the confirmed default remains one hour.

Original validation sizes differ by family (Large 65,536, XL 32,768, XXL 16,384 targets); common-target re-evaluation is pending. Existing test exports use common non-overlapping windows, dropping the tail. Additional 30-minute/context runs record `NVIDIA H100 NVL`, while the grid records an H100 MIG partition; hardware comparability remains unresolved. Report sources now correct the token-volume/overtaking claims, qualify budget/context causality, disclose test reuse, and label follow-ups pending. Updated fit/budget plots fix the data fraction at 100% and carry caveats.

## Possible later experiments and open questions

Broader extensions include context length, depth versus width, symmetric learning-rate sensitivity, and tokenizer changes. A separate data-fraction study could use one model with 1/5/25/100% data at fixed exposure and repeated subsets/seeds; no new subset YAMLs are prepared. Every original family scores better on 100% than 25% by a small margin; Small/25% processed 8.206 equivalents, so insufficient time to revisit data cannot explain the whole grid. Token-level losses are not directly comparable across tokenizers.

## Next steps

1. User runs the common validation audit and six one-hour seed trials from `FOLLOWUP_COMMANDS.md`, then returns logs/configs/common scores. Confirm matching GPU allocation.
2. Interpret common scores and training-seed variability before optionally adding the short equal-token pair. Keep four-hour runs and other new studies deferred.
3. Add modest blind ratings if useful; context probes are supplementary diagnostics. Update the report only from actual results. All 20 original test scores were inspected; reused test evaluation is not fresh confirmation.
