# Next steps: make our TinyStories conclusions more reliable

This is a practical follow-up guide built from the code, saved results, and report currently in this repository. The new experiments below are **proposals**. The four YAML files in `configs/followups/` are validated configuration files, but no training has been launched for them.

The immediate objective is to strengthen this claim:

> Among the tested configurations, XL achieved the lowest observed validation loss under our one-hour training recipe on the recorded H100 MIG allocation.

We want to learn how repeatable this result is and why it happens. We do not yet know whether XXL would overtake XL with more training, or whether the same ranking would hold under another optimizer recipe.

## 1. What we already have and should build on

| Existing asset | Where it lives | How to use it next |
| --- | --- | --- |
| Twenty one-hour training runs | `results/<run>/run_config.json` and `metrics.csv` | Use saved settings as the historical source of truth; current defaults do not describe every old run. |
| Full-window test evaluation | `results/<run>/test_results.json`, [analysis/test_summary.csv](analysis/test_summary.csv) | All twenty have test results. These are useful descriptive evidence, but the test set has already been inspected. |
| Initial generated stories | `results/<run>/generations.md`, `analysis/generations_all.md` | Use the five prompts per model to pilot a scoring rubric; they are too few for a strong capability conclusion. |
| Model checkpoints on the Hub | `dsa4212-tinystories/tinystories-nanogpt` | Obtain the relevant XL and XXL checkpoints for re-evaluation and generation. They are not included in the local `results/` archive. |
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

Before launching more training, revise these points in both report versions:

| Current claim or concern | What the saved results actually support |
| --- | --- |
| No run consumes even 25% of the split. | Incorrect as a token-volume claim: many runs process multiple cache-equivalents. Exact unique coverage is not logged. |
| Larger models always work better. | Performance improves through XL in the tested grid, while XXL is slightly worse under the one-hour recipe. |
| XXL is worse because it is undertrained and will eventually overtake XL. | Slower training is a plausible explanation; overtaking is an untested hypothesis. |
| The fitted minimum near 130M establishes the compute-optimal size. | It is an exploratory fit to five model families, using their best data-fraction results. It does not provide a precise or universal optimum. |
| Equal wall-clock time means equal FLOPs or measured GPU utilization. | Time is controlled; theoretical `6ND` is an approximation. Actual utilization was not directly measured. |
| The 30-minute runs isolate the effect of reducing the budget. | Their saved hardware label is `NVIDIA H100 NVL`, while the original grid records `NVIDIA H100 NVL MIG 3g.47gb`. Reconcile allocation/conditions before making a controlled budget claim. |

Also compare a fixed data fraction across budgets. `analyze_additions.py` currently selects the best data fraction at one hour but compares it with full-data 30-minute runs. Use `data100` consistently for a cleaner follow-up.

### 2.2 Check validation on identical targets

This is the most important immediate evaluation improvement. Training currently uses its microbatch size for validation too:

| Original full-data run | Microbatch | Accumulation | Training targets/update | Validation batches | Validation targets/evaluation |
| --- | ---: | ---: | ---: | ---: | ---: |
| Large | 16 | 1 | 4,096 | 16 | 65,536 |
| XL | 8 | 2 | 4,096 | 16 | 32,768 |
| XXL | 4 | 4 | 4,096 | 16 | 16,384 |

The effective training batch is the same, but the validation sample sizes differ. Therefore the original logged validation scores are not all computed over identical target sets. This matters especially for the small XL/XXL validation gap.

**Do now:** re-score the saved finalist checkpoints on one common validation protocol, without retraining. Reuse `eval_hf.py:evaluate_full` or the batching/loss helpers in `train.py`, but supply validation tokens. The existing `run.py evaluate` action evaluates test data only; it is not a validation-evaluation command.

On the training machine, obtain the two original checkpoints from the repository already used by `eval_hf.py`. This downloads files but does not train or evaluate anything. Record the printed Hub revision with your results:

```bash
python - <<'PY'
from huggingface_hub import HfApi, hf_hub_download
repo = "dsa4212-tinystories/tinystories-nanogpt"
revision = HfApi().model_info(repo).sha
print("Checkpoint repository revision:", revision)
for run in ("xl_data100", "xxl_data100"):
    print(hf_hub_download(repo_id=repo, filename=f"{run}/checkpoint.pt",
                          revision=revision, local_dir="hf_cache"))
PY
```

Keep disk and memory available for both checkpoint files; they include optimizer state as well as weights. `eval_hf.py` can skip runs that already have evaluation outputs, so `--keep-checkpoints` alone is not a reliable way to retrieve an already evaluated run.

For an immediate sampled check, this command reuses the existing helpers and evaluates **identical 65,536 validation targets** for both models. It requires CUDA and the original full cache on the training machine. It creates an aggregate audit file but no confidence interval:

```bash
python - <<'PY'
import hashlib
import json
from pathlib import Path
import torch
from train import (select_device, load_tokens, fixed_batches,
                   load_checkpoint_model, evaluate_model)

device = select_device("cuda")
cache = Path("data/tinystories/data100")
tokens = load_tokens(cache, "val", 256)
batches = fixed_batches(tokens, batch_size=4, block_size=256, count=64, seed=43)
cache_hash = hashlib.sha256((cache / "val.bin").read_bytes()).hexdigest()
results = []
for run in ("xl_data100", "xxl_data100"):
    model, checkpoint = load_checkpoint_model(f"hf_cache/{run}/checkpoint.pt", device)
    assert model.config.block_size == 256
    del checkpoint
    loss, ppl, count = evaluate_model(model, tokens, batches, 256, device, "bfloat16")
    row = {"run": run, "val_loss": loss, "val_perplexity": ppl,
           "scored_targets": count, "validation_seed": 43,
           "validation_cache_sha256": cache_hash}
    results.append(row)
    print(row)
    del model
    torch.cuda.empty_cache()
Path("analysis").mkdir(exist_ok=True)
Path("analysis/followup_validation_audit.json").write_text(
    json.dumps(results, indent=2) + "\n")
PY
```

For stronger final comparisons, prefer the same non-overlapping held-out windows for every candidate. **Code addition needed:** a small reusable validation evaluator that produces aggregate loss plus per-story or suitably grouped block losses, with target counts and cache hashes. This is needed for the uncertainty estimates below; the aggregate command alone cannot provide them.

For future sampled validation, existing configuration fields can already equalize the number of windows:

| Family | Microbatch | Set `evaluation.batches` to | Total windows | Total target tokens |
| --- | ---: | ---: | ---: | ---: |
| Large | 16 | 16 | 256 | 65,536 |
| XL | 8 | 32 | 256 | 65,536 |
| XXL | 4 | 64 | 256 | 65,536 |

With the same validation cache and fixed sampling seed, this gives the same window-start sequence, grouped into different batches. The new follow-up YAMLs use these settings. Small numerical differences from batching or precision can remain.

### 2.3 Use two different kinds of uncertainty correctly

- **Evaluation uncertainty:** would the difference change on another held-out sample? Use paired resampling of the same stories or suitable blocks scored by both models. Do not treat neighbouring tokens as independent observations. The current aggregate CSV cannot provide a reliable per-story confidence interval on its own.
- **Training uncertainty:** would the ranking change after another random initialization and sampled training sequence? This needs independent training trials with fixed assessment data. A confidence interval from held-out text alone does not answer it.

The current full-test pass already scores the same 957,184 targets for the original 256-context runs. Do not spend time repeatedly rerunning it unchanged. Since those test scores have already been seen, disclose this when discussing later adaptive choices; use validation for further selection. If a strictly untouched confirmation set is required, reserve genuinely unused examples before further tuning.

## 3. Reproduce the finalists before varying random seeds

### Ready-to-run confirmation configurations

The following files were created from the actual `results/xl_data100/run_config.json` and `results/xxl_data100/run_config.json`, retaining each model's architecture, microbatch, optimizer, full cache, seed, and effective training batch. They use fresh output directories and the harmonized validation counts above.

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
python -m unittest discover -s tests -v
python run.py prepare --config configs/followups/xl_full_1h_seed42.yaml
```

The preparation command should reuse the existing compatible full cache. Preserve that original cache and its provenance. If compatibility fails, investigate the cause before rebuilding; do not use `--force` simply to bypass it. The current preparation code shuffles the training stories, and old metadata does not explicitly record that algorithm/version, so matching lengths alone cannot establish historical cache identity.

Then run the pair sequentially:

```bash
python run.py train --config configs/followups/xl_full_1h_seed42.yaml
python run.py train --config configs/followups/xxl_full_1h_seed42.yaml
```

Run the second command after the first succeeds. Each folder must be unused. These are fresh random-weight runs, not resumptions of the old checkpoints. Same-seed repetition helps check reproducibility and hardware conditions, but it is not an independent random-seed trial.

Analyze only this budget group:

```bash
python analyze.py --runs-dir runs/followups_1h --out-dir analysis/followups_1h
```

Save the common-protocol validation scores with the new results. The original seed-42 checkpoints remain useful baselines; because their validation work differed, compare re-evaluated scores and account for the small difference in validation overhead when interpreting time-limited runs.

## 4. Add independent repeat trials to test the one-hour ranking

This is the most direct way to add certainty to the existing one-hour conclusion.

**Before changing seeds, a code change is required.** `experiment.seed` currently controls both training randomness and preparation, and also influences validation-window sampling. Setting it to 43 without separating these uses changes the assessment data or makes the old cache incompatible.

The minimum intended behaviour is:

| Randomness | Follow-up requirement |
| --- | --- |
| Data split and story shuffle | Keep a fixed preparation seed of 42 and reuse the exact cache. |
| Model initialization and training-window sampling | Vary the training seed, for example 42, 43, and 44. |
| Validation sample | Keep the same sample/seed across all training trials, or score the same complete held-out windows. |
| Generation sampling | Allow a separate generation seed; do not tie every sample to its training seed. |

Separate preparation/training/evaluation seeds would need to be added to the configuration validator and the corresponding preparation/training functions, with compatible handling of existing metadata. These fields are **not supported by the current YAML schema** yet.

After that addition, aim for three one-hour trials each for XL and XXL on 100% data: six model-hours total, including a seed-42 baseline if it remains comparable. Three seeds give an initial estimate of variability, not a guarantee of a precise confidence interval. More repeats may be needed if the differences remain close to the observed variability.

For every seed, record `XXL loss - XL loss` on the same validation targets. Positive differences favour XL. Report individual results, the mean difference, and variability; preserve inconvenient outcomes. Avoid presenting one best seed as the model's representative score.

## 5. Test whether the preferred size changes with a larger time budget

Question:

> Under the same shared recipe and hardware, does XXL become preferable when both models receive four hours?

This can use the existing time-limited trainer. After the validation/hardware checks above, the prepared configs are runnable:

```bash
python run.py train --config configs/followups/xl_full_4h_seed42.yaml
python run.py train --config configs/followups/xxl_full_4h_seed42.yaml
python analyze.py --runs-dir runs/followups_4h --out-dir analysis/followups_4h
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

**Code additions required:** an update/token stopping criterion and a learning-rate schedule based on progress through that budget. Use the same learning-rate value at the same update/token count for both candidates. A safety time limit must not stop one candidate before it completes the planned exposure. The current schema does not accept `max_steps`, `max_tokens`, or `epochs`.

Equal exposure does not require unique data coverage. If the question specifically requires an epoch, also add sampling without replacement over a defined set of windows and handle trailing tokens explicitly. Changing the sampler changes the experiment and needs its own verification. Do not call randomly sampled dataset-equivalents completed epochs.

## 7. Add one small capability evaluation

Build from the existing `generations.md` examples and `generate.py`, but keep a new prompt set and rubric fixed before seeing candidate outputs. Start with XL and XXL; Large can be a useful additional reference if resources allow.

- Use about 20 story beginnings and two samples per prompt/model, with recorded generation seeds.
- Have two team members independently score grammar, consistency, plot coherence, and repetition using a short written rubric and hidden model labels.
- Add paired prompts that swap names, colours, or object locations. Record whether the continuation follows the changed fact.
- Keep relevant facts within the 256-token context and mark outputs that reach the generation limit. A cut-off ending should not automatically count as a coherence failure.

The current `generate.py` reuses the checkpoint seed and does not stop at end-of-text. `eval_hf.py` has an independent fixed generation seed and truncates displayed text at end-of-text, but it still uses only five fixed prompts and one sample per prompt. Add configurable independent sample seeds, stopping/termination labels, and saved structured outputs before calling this a formal repeated-sample benchmark.

Report the raw ratings and agreement between reviewers. This tests whether the loss ranking corresponds to observable behaviour; it does not establish broad reasoning ability or developmental age.

## 8. What to record for every new run

| Record | Why it matters |
| --- | --- |
| Saved resolved config and code commit | Establish which implementation and recipe produced the result. |
| Dataset revision and cache hashes | Verify that training and assessment text actually match across runs. |
| Hardware allocation, precision, and software versions | A matching model name alone does not establish matching execution conditions. |
| Budget, actual time, optimizer steps, and processed targets | Keep time, data availability, exposure, and unique coverage distinct. |
| Common evaluation target count/hash and per-block scores | Support paired comparisons and evaluation uncertainty estimates. |
| Checkpoint location and identifier | Enable independent re-evaluation and generation. |
| Training and generation seeds | Distinguish independent samples from repeated deterministic output. |
| Experiment purpose and stopping rule | Avoid choosing a favourable endpoint after looking at the results. |

Keep the original twenty-run archive separate. Analyze each follow-up budget/protocol group in its own output directory, then compare groups using saved metadata. Do not run `create_experiments.py` over hand-edited files without reviewing its overwrite behaviour, and do not mix a best-over-data-fractions result with a fixed-data-fraction result when attributing a difference to time alone.

## 9. Recommended personal work order

1. **First session, no retraining:** correct the report's token-exposure and causal claims; obtain the finalist checkpoints; harmonize final validation and retain per-block scores.
2. **Then check reproducibility:** use the two one-hour confirmation configs on the original hardware allocation. Inspect the common validation results and hardware/throughput records.
3. **For certainty about the one-hour winner:** separate seed roles, then obtain independent XL/XXL repeat trials with the data and evaluation fixed.
4. **For an explanation of the ranking:** choose the longer-budget pair or the equal-token experiment. They answer different questions; neither requires rerunning the entire twenty-run grid.
5. **For a stronger language-capability claim:** add the small blind story/context evaluation.
6. **Finish the report:** distinguish observed results, uncertainty, plausible explanations, and untested hypotheses. Describe exactly which conclusions changed after the follow-up.

If resources are tight, prioritize harmonized evaluation and repeat trials for the original claim. If the primary aim is understanding the capacity/throughput trade-off, prioritize the equal-token comparison after the evaluation audit. The four-hour pair is useful for investigating a larger practical budget, but completing it once would not by itself establish a statistically reliable winner.
