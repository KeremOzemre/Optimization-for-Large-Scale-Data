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

See [FOLLOWUP_COMMANDS.md](FOLLOWUP_COMMANDS.md) for commands and [NEXT_STEPS_GUIDE.md](NEXT_STEPS_GUIDE.md) for rationale. The archived common-validation audit and six fresh one-hour XL/XXL trials (seeds 42/43/44) have returned results; see [the analysis and plots](analysis/followups/review/ANALYSIS.md). The next optional explanatory pair is the preplanned 143.36M-target comparison, about 32 + 62 minutes at new mean throughputs. Four-hour runs remain deferred. New trials use the full cache, preparation seed 42, evaluation seed 43, 65,536 in-loop validation targets, and 4,096 training targets/update.

The separate scientific paper is [report/report2.tex](report/report2.tex), with an eight-page, two-column PDF, its own bibliography, source-derived tables, and four figures. It covers all 30 research runs (20-run screen, three 30-minute extensions, one 512-context extension, six repetitions), the four-checkpoint audit, all historical test exports, generated-story observations, exposure/compute analyses, and nine pilot/development records. Methods are concise; attention/loss/schedule derivations and duplicated summary tables were removed. Original size/loss and exposure graphs motivate the XL/XXL re-tests. Hardware/scoring changes, missing provenance, small samples, and pending work remain explicit. It supplements the original report; rebuild commands are in the README.

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

`score_checkpoints.py` scores common validation targets and saves per-story/bootstrap diagnostics; test mode is explicit. `summarize_followups.py` reports seed means/SD and paired differences. `analyze_followups.py` audits supplied configs/logs/scores and regenerates `analysis/followups/review/` (analysis Markdown, audit JSON, two plots), without loading weights or modifying raw results. Story intervals condition on fixed checkpoints and do not measure retraining variability. Blind story ratings and the v2 balanced color diagnostic remain supplementary and pending; v2 removes the original first-color shortcut, not every possible heuristic.

For the first model-size comparisons, use the same prepared data, tokenizer, seed, hardware, time budget, optimizer and schedule, batch settings, and fixed validation procedure. Change model layers, width, and heads, with a distinct output directory for each run. Record the actual parameter count and processed training tokens: equal time does not imply equal token exposure. Compare held-out cross-entropy and perplexity only when the tokenizer is the same. Reserve test results for after choosing a model using validation. Generated-story quality can supplement, but not replace, the initial held-out metrics.

## Current status and limitations

The original pipeline was smoke-tested on Apple MPS, and setup passed 19 CPU checks. The full cache is prepared. The original `results/` archive contains 20 one-hour H100 MIG bfloat16 runs; all 262,432 logged counters match the completed-step formula. Supplied `runs/followups_1h/` now contains six new configs/logs; all 29,961 rows have finite recorded losses and `tokens=step×4096`. All six finish at 3,600.008–3,600.101 seconds. Training controls, code/cache hashes, hardware label, precision, and software versions match across repetitions. This analysis did not launch training or re-evaluation.

New reported full-validation CE: XL mean **1.494657**, sample SD **0.003793**, geometric PPL **4.457805**; XXL mean **1.524387**, SD **0.009792**, PPL **4.592329**. Paired XXL-minus-XL CE gaps for seeds 42/43/44 are **0.039417 / 0.034309 / 0.015466**. Every XL run beats every XXL run. XL has about **2.93% lower PPL** and processes **268.939M vs 139.278M** targets on average (**1.931×**). These are 0.5674 and 0.2938 cache-equivalents. XXL seed 44 processed 9.8% more targets than seed 43, so observed variation includes throughput/exposure differences as well as training randomness.

Four archived baselines remain downloaded in ignored `hf_cache/`; pinned revision and hashes are in `analysis/followups/checkpoint_manifest.json`. Their supplied audit rows match those hashes. Common validation CE is Large/full **1.546940**, XL/50% **1.491074**, XL/full **1.489276**, XXL/full **1.523562**. The full-data XL–XXL gap grows from 0.003782 under unequal sampled scoring to **0.034285** on the supplied common scores. XL/full edges XL/50% by just 0.001798, reversing the old sampled ordering without establishing a general fraction winner.

Handoff limitations: neither scoring directory includes `protocol.json`, `paired_comparisons.json`, or per-story NPZ files; all six new checkpoints and `training_summary.json` files are missing locally. Score rows report 3,808,643 targets, matching the intended count, but offline precision/context policy/validation identity and story intervals cannot be independently verified here. Recover existing artifacts before final reproducibility claims. Source CSVs/logs are preserved; no missing interval or protocol was fabricated.

On the full cache of 473,992,236 tokens, original one-hour exposure ranges from 2.077 equivalents for small to 0.303 for XXL. XXL's measured-throughput estimate for one equivalent is about 3h 18m; this does not guarantee an epoch. Training samples with replacement and has no without-replacement epoch or resume interface. Only one seed exists per original configuration. Follow-up seeds are now separated, but the dataset revision remains unpinned and old runs lack cache hashes.

The supported conclusion is a consistent observed XL advantage in the supplied three-seed, one-hour shared-recipe comparison. Three trials are initial repeatability evidence, not a precise population-level or significance claim; no per-size optimizer search was done. Existing sampled curves favour XXL near equal target counts, but time-based schedule progress differs, so this is descriptive rather than controlled exposure evidence. The fresh equal-token pair is now the targeted next test. Universal size, convergence, and exposure-causality claims remain unsupported.

Original validation sizes differ by family (Large 65,536, XL 32,768, XXL 16,384 targets); common-score exports are now supplied with the provenance limits above. Historical test exports drop the short tail. Additional 30-minute/context runs record a full H100 label rather than MIG, so their hardware comparability remains unresolved. Reports distinguish completed supplied audit/repetition results from pending equal-token and capability checks; updated fit/budget plots retain their caveats.

## Possible later experiments and open questions

Broader extensions include context length, depth versus width, symmetric learning-rate sensitivity, and tokenizer changes. A separate data-fraction study could use one model with 1/5/25/100% data at fixed exposure and repeated subsets/seeds; no new subset YAMLs are prepared. Every original family scores better on 100% than 25% by a small margin; Small/25% processed 8.206 equivalents, so insufficient time to revisit data cannot explain the whole grid. Token-level losses are not directly comparable across tokenizers.

## Next steps

1. Recover scoring protocol, per-story/paired-comparison files, and new training summaries; preserve new checkpoints for independent re-evaluation.
2. If explaining the ranking is the next goal, run the preplanned fresh 35,000-update pair at seed 42 with matching step schedules. Keep four-hour runs and broad new studies deferred.
3. Add modest blind ratings if useful. All 20 original test scores were inspected; reused test evaluation is not fresh confirmation. Data-fraction conclusions still come from the original single-seed grid.
