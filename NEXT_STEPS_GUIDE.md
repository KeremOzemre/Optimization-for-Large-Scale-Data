# Next steps: make our TinyStories conclusions more reliable

This is a practical follow-up guide built from the code, saved results, and report currently in this repository. The follow-up tooling and ten training YAMLs are now implemented. **No new GPU training or full validation audit has been run.** Use [FOLLOWUP_COMMANDS.md](FOLLOWUP_COMMANDS.md) for the exact commands, execution order, budgets, and files to return; this guide explains the reasoning behind them.

The immediate objective is to strengthen this claim:

> Among the tested configurations, XL achieved the lowest observed validation loss under our one-hour training recipe on the recorded H100 MIG allocation.

We want to learn how repeatable this result is and why it happens. We do not yet know whether XXL would overtake XL with more training, or whether the same ranking would hold under another optimizer recipe.

## 1. What we already have and should build on

| Existing asset | Where it lives | How to use it next |
| --- | --- | --- |
| Twenty one-hour training runs | `results/<run>/run_config.json` and `metrics.csv` | Use saved settings as the historical source of truth; current defaults do not describe every old run. |
| Full-window test evaluation | `results/<run>/test_results.json`, [analysis/test_summary.csv](analysis/test_summary.csv) | All twenty have test results. These are useful descriptive evidence, but the test set has already been inspected. |
| Initial generated stories | `results/<run>/generations.md`, `analysis/generations_all.md` | Use the five prompts per model to pilot a scoring rubric; they are too few for a strong capability conclusion. |
| Model checkpoints on the Hub | `dsa4212-tinystories/tinystories-nanogpt` | Four baselines are available locally in `hf_cache/` after download, with a pinned revision and checksum manifest. They remain separate from `results/`. |
| Result aggregation | [analyze.py](analyze.py) | Regenerate tables and plots. For close comparisons, calculate from raw metrics rather than its rounded summary values. |
| Training-exposure audit | [analysis/token_exposure_and_epoch_estimates.md](analysis/token_exposure_and_epoch_estimates.md) and its CSV | Distinguish prepared data, repeated token exposure, and estimated runtime. |
| Exploratory model-size fit | [analyze_compute_optimal.py](analyze_compute_optimal.py) | Treat the quadratic minimum as descriptive, not a proven optimum or scaling law. |
| Additional budget/context runs | `runs/large_t1800`, `runs/xl_t1800`, `runs/xxl_t1800`, `runs/xl_ctx512` | Audit their configurations and hardware before combining them with the original grid. |
| Draft report | [REPORT.md](REPORT.md) and [report/report.tex](report/report.tex) | Update both versions when evidence or interpretation changes. |

The original full-data reference points are:

| Model | Parameters | Final validation loss | Recorded test loss | One-hour training targets | Full-cache-equivalent exposure |
| --- | ---: | ---: | ---: | ---: | ---: |
| Large | 51,037,184 | 1.546009 | 1.539026 | 562,429,952 | 1.187x |
| XL | 123,747,840 | 1.506811 | 1.482373 | 279,830,528 | 0.590x |
| XXL | 253,085,696 | 1.510593 | 1.517420 | 143,654,912 | 0.303x |

The full training stream contains 473,992,236 tokens. Exposure divides processed targets by cache size; random sampling can repeat text, so this is not exact unique coverage or a sequential epoch.

Use 100% data for both finalists in the main follow-ups so data availability stays controlled. Comparing XL on 50% with XXL on 100% would change data availability as well as architecture.

## 2. First correct the interpretation and evaluation protocol

### 2.1 Repair claims that are stronger than the evidence

The Markdown and LaTeX reports now correct these historical claims:

| Current claim or concern | What the saved results actually support |
| --- | --- |
| No run consumes even 25% of the split. | Incorrect as a token-volume claim: many runs process multiple cache-equivalents. Exact unique coverage is not logged. |
| Larger models always work better. | Performance improves through XL in the tested grid, while XXL is slightly worse under the one-hour recipe. |
| XXL is worse because it is undertrained and will eventually overtake XL. | Slower training is a plausible explanation; overtaking is an untested hypothesis. |
| The fitted minimum near 130M establishes the compute-optimal size. | It was an exploratory fit to five model families using their best fractions; the updated plot fixes the fraction at 100%. It does not provide a precise or universal optimum. |
| Equal wall-clock time means equal FLOPs or measured GPU utilization. | Time is controlled; theoretical `6ND` is an approximation. Actual utilization was not directly measured. |
| The 30-minute runs isolate the effect of reducing the budget. | Their saved hardware label is `NVIDIA H100 NVL`, while the original grid records `NVIDIA H100 NVL MIG 3g.47gb`. Reconcile allocation/conditions before making a controlled budget claim. |

`analyze_additions.py` now uses `data100` at both budgets and labels the hardware/evaluation caveat. Hardware comparability remains unresolved; fixing the fraction alone does not isolate time.

### 2.2 Check validation on identical targets

The historical validation protocol used the training microbatch size:

| Model family | Historical validation targets |
| --- | ---: |
| Small / Medium / Large | 65,536 |
| XL | 32,768 |
| XXL | 16,384 |

This matters for the close XL–XXL gap. The new configs explicitly set evaluation
microbatch 4, batches 64, and seed 43, independently of training settings.
`score_checkpoints.py` now re-scores multiple saved checkpoints on identical
validation targets. Its default is a full pass including the short tail; it also
supports the new 256-window sampled protocol. It saves token-weighted CE,
perplexity, cache/checkpoint hashes, per-story losses, and paired bootstrap intervals.
See step 2 of [FOLLOWUP_COMMANDS.md](FOLLOWUP_COMMANDS.md) for commands. The
`run.py evaluate` action still scores test data; use the new scorer for validation.

### 2.3 Use two different kinds of uncertainty correctly

- **Evaluation uncertainty:** would the difference change on another held-out sample? Use paired resampling of the same stories or suitable blocks scored by both models. Do not treat neighbouring tokens as independent observations. The current aggregate CSV cannot provide a reliable per-story confidence interval on its own.
- **Training uncertainty:** would the ranking change after another random initialization and sampled training sequence? This needs independent training trials with fixed assessment data. A confidence interval from held-out text alone does not answer it.

The current full-test pass already scores the same 957,184 targets for the original 256-context runs. Do not spend time repeatedly rerunning it unchanged. Since those test scores have already been seen, disclose this when discussing later adaptive choices; use validation for further selection. If a strictly untouched confirmation set is required, reserve genuinely unused examples before further tuning.

## 3. Reproduce the finalists before varying random seeds

### Ready-to-run confirmation configurations

The following files were created from the actual `results/xl_data100/run_config.json` and `results/xxl_data100/run_config.json`, retaining each model's architecture, microbatch, optimizer, full cache, seed, and effective training batch. They use fresh output directories and an explicit shared evaluation microbatch of 4, 64 batches, and seed 43 (65,536 targets). Additional seed-43/44 and equal-token YAMLs are listed in FOLLOWUP_COMMANDS.md.

| Config | Output directory | Budget |
| --- | --- | ---: |
| [configs/followups/xl_full_1h_seed42.yaml](configs/followups/xl_full_1h_seed42.yaml) | `runs/followups_1h/xl_full_1h_seed42` | 1 hour |
| [configs/followups/xxl_full_1h_seed42.yaml](configs/followups/xxl_full_1h_seed42.yaml) | `runs/followups_1h/xxl_full_1h_seed42` | 1 hour |
| [configs/followups/xl_full_4h_seed42.yaml](configs/followups/xl_full_4h_seed42.yaml) | `runs/followups_4h/xl_full_4h_seed42` | 4 hours |
| [configs/followups/xxl_full_4h_seed42.yaml](configs/followups/xxl_full_4h_seed42.yaml) | `runs/followups_4h/xxl_full_4h_seed42` | 4 hours |

Run these on the same H100 MIG allocation as the original experiment. The time estimates in this guide do not apply to a laptop or a different GPU allocation. Keep other GPU jobs, precision, power conditions, and software versions controlled as far as practical.

From the project root on the training machine:

```bash
source .venv/bin/activate
python run.py prepare --config configs/followups/xl_full_1h_seed42.yaml
```

The preparation command should reuse the existing compatible full cache. Preserve that original cache and its provenance. If compatibility fails, investigate the cause before rebuilding; do not use `--force` simply to bypass it. The current preparation code shuffles the training stories, and old metadata does not explicitly record that algorithm/version, so matching lengths alone cannot establish historical cache identity.

Then run the pair sequentially:

```bash
python run.py train --config configs/followups/xl_full_1h_seed42.yaml
python run.py train --config configs/followups/xxl_full_1h_seed42.yaml
```

Run the second command after the first succeeds. Each folder must be unused. These are fresh random-weight runs, not resumptions of the old checkpoints. Same-seed repetition helps check reproducibility and hardware conditions, but it is not an independent random-seed trial.

The CPU correctness checks have already passed. Score and summarize only this budget group:

```bash
python score_checkpoints.py --checkpoints runs/followups_1h/*/checkpoint.pt --device cuda --output-dir analysis/followups/seed_validation
python summarize_followups.py --scores analysis/followups/seed_validation/scores.csv --output analysis/followups/seed_summary.md
```

Save the common-protocol validation scores with the new results. The original seed-42 checkpoints remain useful baselines; because their validation work differed, compare re-evaluated scores and account for the small difference in validation overhead when interpreting time-limited runs.

## 4. Add independent repeat trials to test the one-hour ranking

This is the most direct way to add certainty to the existing one-hour conclusion.

**Implemented:** the follow-up configs separate `data.preparation_seed=42`, `experiment.seed=42/43/44`, `evaluation.seed=43`, and `generation.seed=101`. Changing the training seed now leaves the prepared cache and validation sample unchanged in these configs.

The minimum intended behaviour is:

| Randomness | Follow-up requirement |
| --- | --- |
| Data split and story shuffle | Keep a fixed preparation seed of 42 and reuse the exact cache. |
| Model initialization and training-window sampling | Vary the training seed, for example 42, 43, and 44. |
| Validation sample | Keep the same sample/seed across all training trials, or score the same complete held-out windows. |
| Generation sampling | Allow a separate generation seed; do not tie every sample to its training seed. |

The configuration validator, preparation, training, and generation now support these fields. Legacy YAMLs keep their earlier derived defaults when the new fields are omitted; the follow-up YAMLs always specify them.

Six ready YAMLs specify three one-hour trials each for XL and XXL on 100% data: six model-hours total, including a seed-42 baseline if it remains comparable. Three seeds give an initial estimate of variability, not a guarantee of a precise confidence interval. More repeats may be needed if the differences remain close to the observed variability.

For every seed, record `XXL loss - XL loss` on the same validation targets. Positive differences favour XL. Report individual results, the mean difference, and variability; preserve inconvenient outcomes. Avoid presenting one best seed as the model's representative score.

## 5. Test whether the preferred size changes with a larger time budget

Question:

> Under the same shared recipe and hardware, does XXL become preferable when both models receive four hours?

This can use the existing time-limited trainer. After the validation/hardware checks above, the prepared configs are runnable:

```bash
python run.py train --config configs/followups/xl_full_4h_seed42.yaml
python run.py train --config configs/followups/xxl_full_4h_seed42.yaml
python score_checkpoints.py --checkpoints runs/followups_4h/*/checkpoint.pt --device cuda --output-dir analysis/followups/budget_validation
```

Run sequentially and inspect each command's success. This pair costs approximately eight model-hours on the target allocation; budget for setup, final evaluation, and saving separately. It need not be run if the cheaper checks already change your interpretation or resources are limited.

At the original observed rates, XXL would process one full-cache-equivalent in about 3h 18m. Four hours provides a planning margin, but neither a guarantee of that rate nor complete unique coverage. Faster models would still receive more tokens.

The learning-rate schedule stretches with the budget: warmup is 72 seconds in a one-hour run and 288 seconds in a four-hour run. Therefore a four-hour run is not an exact continuation of the one-hour run, and its score at hour one is not the same experiment as a one-hour-scheduled run. Use fresh runs and describe the result as performance under each budget-specific schedule.

Possible outcomes:

- XXL overtakes XL: evidence that the preferred configuration depends on the time budget; it does not isolate token exposure from all schedule effects.
- XL remains better: the original explanation was incomplete; learning-rate suitability, architecture, or other constraints may matter.
- Results vary across seeds or differ only slightly: report uncertainty and avoid a firm ranking.

There is currently no resume command. Do not assume the optimizer state in a checkpoint makes continuation available through `run.py`.

## 6. Optional: isolate training exposure with an equal-token comparison

This is a different question from equal-time performance:

> If XL and XXL receive the same training targets under the same progress-based recipe, which predicts held-out text better?

Use the same full cache and sampled window sequence, the same effective batch of 4,096 targets per update, and a shared target/update budget. For example, **115,721 optimizer updates produce 473,993,216 targets**, close to one full-cache-equivalent.

| Model | Planning estimate for that exposure |
| --- | ---: |
| XL | About 1h 42m |
| XXL | About 3h 18m |

The pair is roughly five model-hours at the original rates. These estimates require remeasurement if hardware, code, or batch strategy changes.

**Implemented:** `training.max_steps=115721` with `schedule.progress=steps`. Both models receive the same learning-rate sequence at the same target count. A six-hour safety cap marks runs that stop early as incomplete. Check `training_summary.json` for `stop_reason=max_steps` and exactly 473,993,216 processed targets. There is still no `epochs` or resume interface.

Equal exposure does not require unique data coverage. If the question specifically requires an epoch, also add sampling without replacement over a defined set of windows and handle trailing tokens explicitly. Changing the sampler changes the experiment and needs its own verification. Do not call randomly sampled dataset-equivalents completed epochs.

## 7. Add one small capability evaluation

Build from the existing `generations.md` examples and `generate.py`, but keep a new prompt set and rubric fixed before seeing candidate outputs. Start with XL and XXL; Large can be a useful additional reference if resources allow.

- Use about 20 story beginnings and two samples per prompt/model, with recorded generation seeds.
- Have two team members independently score grammar, consistency, plot coherence, and repetition using a short written rubric and hidden model labels.
- Add paired prompts that swap names, colours, or object locations. Record whether the continuation follows the changed fact.
- Keep relevant facts within the 256-token context and mark outputs that reach the generation limit. A cut-off ending should not automatically count as a coherence failure.

`generate.py` now supports an independent generation seed and stops at EOT. `sample_stories.py` supplies the frozen 20-prompt/two-seed protocol, structured records, termination labels, blind samples, and a two-rater CSV. `context_checks.py` supplies ten paired color-swap scenarios. Neither command has produced research results yet; actual ratings require two people.

Report the raw ratings and agreement between reviewers. This tests whether the loss ranking corresponds to observable behaviour; it does not establish broad reasoning ability or developmental age.

## 8. What to record for every new run

| Record | Why it matters |
| --- | --- |
| Saved resolved config and code commit | Establish which implementation and recipe produced the result. |
| Dataset revision and cache hashes | Verify that training and assessment text actually match across runs. |
| Hardware allocation, precision, and software versions | A matching model name alone does not establish matching execution conditions. |
| Budget, actual time, optimizer steps, and processed targets | Keep time, data availability, exposure, and unique coverage distinct. |
| Common evaluation target count/hash and per-story scores | Support paired comparisons and evaluation uncertainty estimates. |
| Checkpoint location and identifier | Enable independent re-evaluation and generation. |
| Training and generation seeds | Distinguish independent samples from repeated deterministic output. |
| Experiment purpose and stopping rule | Avoid choosing a favourable endpoint after looking at the results. |

Keep the original twenty-run archive separate. Analyze each follow-up budget/protocol group in its own output directory, then compare groups using saved metadata. Do not run `create_experiments.py` over hand-edited files without reviewing its overwrite behaviour, and do not mix a best-over-data-fractions result with a fixed-data-fraction result when attributing a difference to time alone.

## 9. Recommended personal work order

1. **First session, no retraining:** correct the report's token-exposure and causal claims; obtain the finalist checkpoints; harmonize final validation and retain per-story scores.
2. **Then check reproducibility:** use the two one-hour confirmation configs on the original hardware allocation. Inspect the common validation results and hardware/throughput records.
3. **For certainty about the one-hour winner:** use the separated seed roles to obtain independent XL/XXL repeat trials with the data and evaluation fixed.
4. **For an explanation of the ranking:** choose the longer-budget pair or the equal-token experiment. They answer different questions; neither requires rerunning the entire twenty-run grid.
5. **For a stronger language-capability claim:** add the small blind story/context evaluation.
6. **Finish the report:** distinguish observed results, uncertainty, plausible explanations, and untested hypotheses. Describe exactly which conclusions changed after the follow-up.

If resources are tight, prioritize harmonized evaluation and repeat trials for the original claim. If the primary aim is understanding the capacity/throughput trade-off, prioritize the equal-token comparison after the evaluation audit. The four-hour pair is useful for investigating a larger practical budget, but completing it once would not by itself establish a statistically reliable winner.
