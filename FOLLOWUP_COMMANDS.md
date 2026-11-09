# Ready-to-run TinyStories follow-ups

The setup is implemented. **No new GPU training or full validation comparison has
been run.** Use this file as the execution checklist; send back the resulting
tables and logs before we revise the research conclusions. The CPU correctness
tests have been run here, so you do not need to run them yourself.

Run commands from the project root. Use the **same GPU partition and software
environment for all runs**; the original grid used `NVIDIA H100 NVL MIG 3g.47gb`.
Run training jobs sequentially. A whole H100 and a MIG slice are different budgets.

## 1. Environment, data, and archived checkpoints

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
export TIKTOKEN_CACHE_DIR="$PWD/data/tiktoken-cache"
python run.py prepare --config configs/followups/xl_full_1h_seed42.yaml
python download_checkpoints.py --revision fa18888b1b9b599e9deb455ca740896e7d0642f3
```

The compatible full cache already exists locally at `data/tokens-full`; preparation
reuses it. On another machine, copy this exact cache or prepare it once, then reuse
it for every follow-up. New runs save cache hashes. Do not rebuild it between seeds.
The archived runs did not record cache hashes, so identical recipe/counts alone
cannot establish byte-for-byte identity with their original training cache.

Four archived checkpoints are downloaded into `hf_cache/` (about 6.62 GB total):
`large_data100`, `xl_data50`, `xl_data100`, `xxl_data100`. These cover a smaller
reference, the originally selected model, and the two full-data finalists. They
are for evaluation; **new training always starts from random weights**. The
revision and SHA-256 hashes are in `hf_cache/download_manifest.json`. On this
computer the download step has already been performed; repeat it only if files are
missing or you move to another machine. It reuses matching files.

## 2. First: audit the existing checkpoints on common validation targets

This needs inference only. It does not retrain or use the test set.

```bash
python score_checkpoints.py \
  --checkpoints hf_cache/large_data100/checkpoint.pt hf_cache/xl_data50/checkpoint.pt hf_cache/xl_data100/checkpoint.pt hf_cache/xxl_data100/checkpoint.pt \
  --data-dir data/tokens-full --device cuda --precision float32 \
  --output-dir analysis/followups/archived_validation
```

Default scoring covers **all 3,808,643 next-token targets** in `val.bin`, resetting
context every 256 tokens and including the short tail. This is a common protocol,
not the old unequal sampled validation. It also differs from the old full-test
export, which dropped its short tail. Keep new and old metric labels separate.

For a shorter audit matching the new in-loop target count, use a separate output:

```bash
python score_checkpoints.py \
  --checkpoints hf_cache/xl_data100/checkpoint.pt hf_cache/xxl_data100/checkpoint.pt \
  --data-dir data/tokens-full --device cuda --precision bfloat16 \
  --sample-windows 256 --seed 43 \
  --output-dir analysis/followups/archived_sampled_validation
```

That scores the same 65,536 sampled targets for both models. The full-pass audit
is the primary comparison; the sampled version checks agreement with in-loop scores.
`paired_comparisons.json` reports **B CE minus A CE**: positive favours A. The
story-resampling interval conditions on fixed checkpoints, assumes stories are
useful resampling units, and does not measure variation from retraining. Duplicate
stories and context across story boundaries can induce dependence; interpret the
interval as a diagnostic, not a complete uncertainty guarantee.

## 3. Priority training: three one-hour seeds per finalist

These six runs cost approximately **6 model-hours**, plus setup/final evaluation.
They retain XL's 8×2 and XXL's 4×4 microbatch/accumulation, giving 4,096 training
targets per update. Preparation seed stays 42, evaluation seed stays 43, and
validation always uses 4 sequences × 64 batches × 256 = 65,536 targets.

```bash
python run.py train --config configs/followups/xl_full_1h_seed42.yaml
python run.py train --config configs/followups/xxl_full_1h_seed42.yaml
python run.py train --config configs/followups/xxl_full_1h_seed43.yaml
python run.py train --config configs/followups/xl_full_1h_seed43.yaml
python run.py train --config configs/followups/xl_full_1h_seed44.yaml
python run.py train --config configs/followups/xxl_full_1h_seed44.yaml
```

Each command must finish successfully before the next begins. Output folders must
be unused; if a run fails, keep the failure log and choose a fresh output directory
for a retry. Seed 42 checks reproducibility under the corrected protocol; 43 and
44 are independent training trials. GPU numerical nondeterminism can remain.

Score all six on the same complete validation stream:

```bash
python score_checkpoints.py \
  --checkpoints runs/followups_1h/*/checkpoint.pt \
  --device cuda --precision float32 \
  --output-dir analysis/followups/seed_validation
python summarize_followups.py \
  --scores analysis/followups/seed_validation/scores.csv \
  --output analysis/followups/seed_summary.md
```

Interpret individual scores, mean/sample SD per family, and paired XXL−XL
differences for seeds 42/43/44. Preserve every seed, including inconvenient results.
Three seeds give an initial variability estimate; they do not establish a universal
ranking. The six-run comparison is more informative than reporting the best seed.

## 4. Optional longer equal-time experiment: four hours each

Use this to ask whether the observed ranking changes under a **larger time budget**.
It costs approximately **8 model-hours**. Both are fresh seed-42 runs; their
warmup/cosine schedules stretch to four hours. This compares budget-specific recipes,
not continuation of the one-hour checkpoints or training duration in isolation.

```bash
python run.py train --config configs/followups/xl_full_4h_seed42.yaml
python run.py train --config configs/followups/xxl_full_4h_seed42.yaml
python score_checkpoints.py \
  --checkpoints runs/followups_1h/xl_full_1h_seed42/checkpoint.pt runs/followups_1h/xxl_full_1h_seed42/checkpoint.pt runs/followups_4h/*/checkpoint.pt \
  --device cuda --precision float32 \
  --output-dir analysis/followups/budget_validation
python summarize_followups.py \
  --scores analysis/followups/budget_validation/scores.csv \
  --output analysis/followups/budget_summary.md
```

One seed at four hours is exploratory. Do not promise that XXL will overtake XL;
the result is currently unknown. Use additional seeds if a close four-hour ranking
becomes central to the report.

## 5. Optional equal-token experiment

Use this to ask about performance after the **same number of training targets**.
Each model takes **115,721 optimizer steps × 4,096 = 473,993,216 targets**. This is
980 more than one full-cache-equivalent. Warmup/cosine progress uses completed
steps, so both models see the same learning-rate sequence at the same exposure.
They also use the same seed and effective batch, giving the same sampled window
sequence despite different microbatch grouping. Different microbatch grouping can
still change dropout randomness and numerical rounding.

```bash
python run.py train --config configs/followups/xl_full_equal_tokens_seed42.yaml
python run.py train --config configs/followups/xxl_full_equal_tokens_seed42.yaml
python score_checkpoints.py \
  --checkpoints runs/followups_equal_tokens/*/checkpoint.pt \
  --device cuda --precision float32 \
  --output-dir analysis/followups/equal_token_validation
python summarize_followups.py \
  --scores analysis/followups/equal_token_validation/scores.csv \
  --output analysis/followups/equal_token_summary.md
```

Old throughputs suggest roughly **1h 42m XL + 3h 18m XXL**, but new hardware and
validation overhead can change that. The six-hour cap **per model** is a safety
limit. Check `training_summary.json`: successful runs must say `stop_reason:
"max_steps"`, `steps: 115721`, and `tokens_processed: 473993216`. A run stopped by
`time_limit` is incomplete; choose a larger cap and a fresh directory before retrying.
Sampling remains with replacement. This is **not a guaranteed epoch**, and the
saved optimizer state does not provide a resume command.

Do not compare the one-hour and equal-token losses as if only exposure changed:
their schedule progress rules differ. The controlled equal-token comparison is
between the two equal-token runs.

## 6. Capability checks, separate from model selection

These commands use the archived full-data finalists first. The same tools can be
used on the new checkpoints once the primary comparisons are complete.

```bash
python sample_stories.py \
  --checkpoints hf_cache/xl_data100/checkpoint.pt hf_cache/xxl_data100/checkpoint.pt \
  --device cuda --output-dir analysis/followups/stories
python context_checks.py \
  --checkpoints hf_cache/xl_data100/checkpoint.pt hf_cache/xxl_data100/checkpoint.pt \
  --device cuda --output-dir analysis/followups/context_checks
```

The sampling command freezes 20 prompts, temperature 0.8, top-k 40, maximum 150
new tokens, and generation seeds 101/102. It produces 80 samples for two models,
stops at EOT, and records whether EOT or the length cap ended each sample.
Do not improve prompts after seeing the outputs and still call them a frozen test.

Give raters only `blind_samples.md` and `ratings.csv`, without `samples.jsonl`
(the latter reveals model identities). Two raters independently score:

| Column | 1 | 3 | 5 |
| --- | --- | --- | --- |
| Grammar | Hard to read | Some errors | Fluent |
| Plot | Incoherent or no progression | Partly connected | Clear, connected progression |
| Consistency | Repeated contradictions | Some tracking errors | Stable characters and facts |
| Repetition | Severe loops | Noticeable repetition | Little unnecessary repetition |

Compare paired prompt/seed scores and report rater disagreement. A story hitting
the length cap is not automatically an incoherent story. The context tool uses ten
paired color-swap scenarios (20 prompts), checks both swapped versions, and saves
candidate log-probabilities. This narrow, hand-written diagnostic is **not evidence
of general reasoning ability**. Human ratings still need actual raters.

## 7. Test use and files to send back

Use validation for the comparisons above. All 20 original test scores have already
been inspected, so the test set is not fresh confirmation for adaptive follow-ups.
After fixing the conclusions, optional descriptive test evaluation is:

```bash
python score_checkpoints.py \
  --checkpoints runs/followups_1h/xl_full_1h_seed42/checkpoint.pt runs/followups_1h/xxl_full_1h_seed42/checkpoint.pt \
  --split test --device cuda --precision float32 \
  --output-dir analysis/followups/descriptive_test
```

Do not start with that command or use it to choose hyperparameters.

| Location | Files | What to return |
| --- | --- | --- |
| `analysis/followups/<comparison>/` | `scores.csv`, `protocol.json`, `paired_comparisons.json`, `*_stories.npz`, `*_training_config.json` | The entire comparison directory |
| `runs/followups_*/<run>/` | `metrics.csv`, `run_config.json`, `training_summary.json`, `checkpoint.pt` | CSV and JSON files first; preserve checkpoint for later scoring |
| `analysis/followups/` | `seed_summary.md`, optional budget/equal-token summaries | All summaries you produced |
| `analysis/followups/stories/` | `samples.jsonl`, `protocol.json`, `blind_samples.md`, completed `ratings.csv` | All files after both raters finish |
| `analysis/followups/context_checks/` | `predictions.csv`, `summary.json` | Both files |
| `hf_cache/` | `download_manifest.json` | The manifest; no need to send existing multi-GB checkpoints |

**Recommended first batch:** steps 2 and 3. Then send the results before spending
the additional eight hours or roughly five hours on steps 4 and 5. Everything is
configured, but these optional experiments need not all be run at once. The report
currently labels every follow-up as pending; it will be updated from actual results.
