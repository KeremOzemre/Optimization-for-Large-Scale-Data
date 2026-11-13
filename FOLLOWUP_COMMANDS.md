# Focused TinyStories follow-up commands

> **Status update (13 November 2026).** All stages below are complete. Stage 3
> (the equal-token XL/XXL pair) and a reduced learning-rate XXL run have been
> trained and scored; results are in `analysis/followups/equal_token_143m_validation/`
> and `analysis/followups/lr_validation/`. The evaluation artifacts described
> as missing below (`protocol.json`, `paired_comparisons.json`, per-story `.npz`,
> `training_summary.json`) are now in this repository and on the Hugging Face
> checkpoint repository. Text below that says "pending" or "recover" is kept as
> a record of the plan at the time.

The main question remains: **which tested model predicts TinyStories best after
one hour on the same GPU allocation, using our shared training recipe?**
The original 20-run grid is exploratory evidence. Prioritize common validation
and independent repeat trials before adding another research question.
**Stages 1 and 2 have returned results.** The reported full-validation scores favour
XL in all three one-hour seed pairs: mean CE 1.494657 versus 1.524387 for XXL.
See [the analysis and plots](analysis/followups/review/ANALYSIS.md). The local
handoff lacks evaluation protocol/story files and training summaries; recover
those existing artifacts before treating the comparison as fully reproducible.
Stage 3 and capability checks remain pending. Commands for completed stages are
kept as a record; their existing output folders will prevent accidental reruns.

| Stage | Priority | Additional training | What it answers |
| --- | --- | ---: | --- |
| 1. Common validation audit of saved models | Completed as supplied; recover protocol/story exports | None | Does the recorded ranking hold on identical validation targets? |
| 2. Three fresh one-hour seeds each for XL/XXL | Completed as supplied; 6 GPU-hours used | No rerun needed | Is the one-hour ranking repeatable under the shared recipe? |
| 3. Short equal-token XL/XXL pair | Next optional explanatory experiment | About 1.6 GPU-hours at the new measured rates | Does the ranking change at equal exposure under a matched step schedule? |
| Blind story ratings | Small supplement | None | Can readers notice differences in sampled stories? |
| Four-hour runs, new data-fraction study, learning-rate tuning | Deferred | Separate budgets | Different questions; choose only after the core results. |

Run from the project root, sequentially, on the **same GPU partition and software
environment**. The original grid used `NVIDIA H100 NVL MIG 3g.47gb`, bfloat16.
A whole H100 is a different allocation. Estimates exclude preparation, final
validation, checkpoint saving, and the separate full-validation audit.

## Setup once

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
export TIKTOKEN_CACHE_DIR="$PWD/data/tiktoken-cache"
python run.py prepare --config configs/followups/xl_full_1h_seed42.yaml
```

A compatible full cache already exists locally at `data/tokens-full`; preparation
reuses it. On a new machine, copy the exact cache or prepare it once, then reuse it
throughout the experiment. Keep preparation seed 42 fixed. New runs record cache
hashes and software versions; avoid changing packages between trials. Investigate
incompatibility before rebuilding: do not use `--force` to bypass a mismatch.
Old runs lack cache hashes, so matching recipe/counts cannot establish historical
byte-for-byte identity.

Four archived checkpoints and their logs/configs are already downloaded in
`hf_cache/`: Large/full, XL/50%, XL/full, XXL/full (about 6.62 GB). They are for
assessment; **new training starts from random weights**. Revision and SHA-256
hashes are in `hf_cache/download_manifest.json` and the copied
`analysis/followups/checkpoint_manifest.json`. Only on another machine or with
missing files, run:

```bash
python download_checkpoints.py --revision fa18888b1b9b599e9deb455ca740896e7d0642f3
```

## 1. Audit existing checkpoints on common validation targets

```bash
python score_checkpoints.py \
  --checkpoints hf_cache/large_data100/checkpoint.pt hf_cache/xl_data50/checkpoint.pt hf_cache/xl_data100/checkpoint.pt hf_cache/xxl_data100/checkpoint.pt \
  --data-dir data/tokens-full --device cuda --precision float32 \
  --output-dir analysis/followups/archived_validation
```

This performs inference only on all **3,808,643 next-token validation targets**,
with context reset every 256 tokens and the short tail included. It repairs the
original mismatch: Large scored 65,536 sampled targets, XL 32,768, XXL 16,384.
The new full-stream scores must be labelled separately from the old sampled scores.
The archived full-test evaluator also dropped its short tail; it is a different protocol.

Read `scores.csv`, `protocol.json`, and `paired_comparisons.json` before training.
Positive **B loss minus A loss** favours A. Per-story bootstrap intervals condition
on the fixed checkpoints; they do not measure training-seed variability. Stories
can be dependent through duplication and context across EOT, so treat intervals
as diagnostics. The original validation-overhead and cache-provenance limitations
cannot be repaired by re-scoring.

If this audit changes the ranking substantially, reconcile the data, precision,
and protocol first. Do not interpret an unresolved mismatch by launching more runs.
Large is a smaller reference; XL/50% audits the original selection. The main repeat
comparison below fixes data at 100% for both XL and XXL.

## 2. Main experiment: three one-hour seeds per finalist

These are **six fresh runs**, about six GPU-hours total. The original seed-42
checkpoints remain historical baselines, not replacements for new protocol-matched
trials. The ready configurations are:

| Model | Training seeds | YAMLs | Output root |
| --- | --- | --- | --- |
| XL | 42, 43, 44 | `configs/followups/xl_full_1h_seed42.yaml`, `xl_full_1h_seed43.yaml`, `xl_full_1h_seed44.yaml` | `runs/followups_1h/` |
| XXL | 42, 43, 44 | `configs/followups/xxl_full_1h_seed42.yaml`, `xxl_full_1h_seed43.yaml`, `xxl_full_1h_seed44.yaml` | `runs/followups_1h/` |

Run one command at a time, stopping to resolve any failure before the next command:

```bash
python run.py train --config configs/followups/xl_full_1h_seed42.yaml
python run.py train --config configs/followups/xxl_full_1h_seed42.yaml
python run.py train --config configs/followups/xxl_full_1h_seed43.yaml
python run.py train --config configs/followups/xl_full_1h_seed43.yaml
python run.py train --config configs/followups/xl_full_1h_seed44.yaml
python run.py train --config configs/followups/xxl_full_1h_seed44.yaml
```

All trials share preparation seed 42, evaluation seed 43, and the full cache.
XL uses microbatch 8 with accumulation 2; XXL uses 4 with accumulation 4. Both
process **4,096 targets per update**. Both evaluate 4 sequences x 64 batches x
256 tokens = **65,536 fixed validation targets** approximately every five minutes.
Initialization, dropout, and training-window randomness vary with training seed.
Matching numeric seeds across architectures does not align their initialization noise.

Training refuses an existing output directory. Preserve failed logs; use a fresh
output directory for a retry. Do not overwrite the historical `results/` archive.
Once all six runs succeed, score and summarize them together:

```bash
python score_checkpoints.py \
  --checkpoints runs/followups_1h/xl_full_1h_seed42/checkpoint.pt runs/followups_1h/xxl_full_1h_seed42/checkpoint.pt runs/followups_1h/xl_full_1h_seed43/checkpoint.pt runs/followups_1h/xxl_full_1h_seed43/checkpoint.pt runs/followups_1h/xl_full_1h_seed44/checkpoint.pt runs/followups_1h/xxl_full_1h_seed44/checkpoint.pt \
  --data-dir data/tokens-full --device cuda --precision float32 \
  --output-dir analysis/followups/seed_validation
python summarize_followups.py \
  --scores analysis/followups/seed_validation/scores.csv \
  --output analysis/followups/seed_summary.md
```

### Stop and interpret before spending more compute

Report every seed, mean/sample SD, paired XXL-minus-XL loss differences, actual
training time, throughput, and processed targets. Positive paired differences
favour XL. Three seeds provide an initial repeatability check, not a precise
uncertainty guarantee. A small or inconsistent difference supports an inconclusive
ranking; do not select the best seed or keep running until a preferred model wins.

If one model consistently leads, describe it as best **under this shared recipe**.
No per-size optimizer search was performed. Small numerical gaps need practical
context: a CE difference `d` corresponds to a perplexity ratio `exp(d)`; statistical
separation alone does not establish an important story-quality improvement.
These results are now available: every XL run scores better than every XXL run,
with about 2.93% lower geometric-mean perplexity for XL on average. Training controls
match, but variability includes execution speed: XXL seed 44 processes 9.8% more
targets than seed 43. Three seeds give initial observed repeatability, not a
universal or precise population-level ranking. Recover `protocol.json`,
`paired_comparisons.json`, per-story `.npz` files from both scoring folders and
the six `training_summary.json` files. Preserve the trained weights on the training
machine for independent re-evaluation; they are not included locally.

## 3. Optional: short equal-token comparison

This is a separate question about equal training exposure. The revised target is
**35,000 updates x 4,096 = 143,360,000 targets per model**. It is near XXL's original
one-hour exposure, not an attempted epoch. One full-cache-equivalent has no special
significance under random sampling. The old 474M-target proposal is superseded.

| Config | Output directory | Expected training time on the original allocation |
| --- | --- | ---: |
| `configs/followups/xl_full_equal_tokens_seed42.yaml` | `runs/followups_equal_tokens_143m/xl_full_equal_tokens_seed42` | About 32 minutes |
| `configs/followups/xxl_full_equal_tokens_seed42.yaml` | `runs/followups_equal_tokens_143m/xxl_full_equal_tokens_seed42` | About 62 minutes |

```bash
python run.py train --config configs/followups/xl_full_equal_tokens_seed42.yaml
python run.py train --config configs/followups/xxl_full_equal_tokens_seed42.yaml
python score_checkpoints.py \
  --checkpoints runs/followups_equal_tokens_143m/xl_full_equal_tokens_seed42/checkpoint.pt runs/followups_equal_tokens_143m/xxl_full_equal_tokens_seed42/checkpoint.pt \
  --data-dir data/tokens-full --device cuda --precision float32 \
  --output-dir analysis/followups/equal_token_143m_validation
python summarize_followups.py \
  --scores analysis/followups/equal_token_143m_validation/scores.csv \
  --output analysis/followups/equal_token_143m_summary.md
```

Both models start fresh with seed 42 and use `schedule.progress: steps`:
700 warmup updates, then cosine decay over the remaining update budget. The
sampled training-window sequence and learning rates match by update, while
microbatch grouping can still change dropout draws and numerical rounding.
Do not substitute the old time-scheduled XXL checkpoint for a new step-scheduled run.

Each YAML has a **7,200-second safety cap**. Before accepting the comparison, both
`training_summary.json` files must contain `stop_reason: "max_steps"`,
`steps: 35000`, and `tokens_processed: 143360000`. A time-capped run is incomplete;
preserve it and use a fresh directory and larger cap if a retry is needed.
No resume interface exists. Different hardware/overhead can change estimated times.

Report this single pair as exploratory. Equal targets allow different runtime and
compute. They do not guarantee unique coverage or convergence. Comparing this pair
with the one-hour runs does not isolate exposure as the cause of a ranking change,
because the schedule rule also changes. Compare XL and XXL within each protocol.

## 4. Supplement: blind story ratings

This uses archived full-data finalists and no further training:

```bash
python sample_stories.py \
  --checkpoints hf_cache/xl_data100/checkpoint.pt hf_cache/xxl_data100/checkpoint.pt \
  --device cuda --output-dir analysis/followups/stories
```

The frozen 20 prompts, seeds 101/102, temperature 0.8, top-k 40, and 150-token cap
produce 80 samples for two models. EOT ends a story early; termination is recorded.
Give two raters only `blind_samples.md` and `ratings.csv`; `samples.jsonl` reveals
model identities. Do not revise prompts after inspecting the outputs and still
call them a frozen evaluation. Both raters score independently:

| Criterion | 1 | 3 | 5 |
| --- | --- | --- | --- |
| Grammar | Hard to read | Some errors | Fluent |
| Plot | No coherent progression | Partly connected | Clear progression |
| Consistency | Repeated contradictions | Some tracking errors | Stable facts/characters |
| Repetition | Severe loops | Noticeable repetition | Little unnecessary repetition |

Report paired prompt/seed observations and rater disagreement. A length-capped
story is not automatically incoherent. These ratings supplement CE; they do not
confirm its ranking or establish general reasoning ability.

## 5. Deferred checks and scope limits

- **Four-hour pair:** the `*4h*.yaml` files remain as deferred templates. Their eight
  GPU-hours and one seed per model are lower priority than repeatability. A stretched
  time schedule also prevents interpreting them as continuation or duration alone.
- **Data-fraction study:** all five families have lower recorded validation CE on
  100% than on 25%, with small gaps. Small/25% processed 8.206 cache-equivalents,
  so insufficient time to revisit data cannot explain the whole grid. If this
  becomes the main question, choose one model and smaller subsets (e.g. 1/5/25/100%)
  at fixed target exposure, with repeated subsets/seeds. No new subset YAMLs are included.
- **Learning-rate sensitivity:** a later symmetric search for both finalists can
  check recipe sensitivity. Do not tune only the losing model and claim an optimized
  architecture ranking. It requires a separate budget and fixed selection protocol.
- **Context probes:** the old color-swap test admitted a first-color shortcut
  (20/20 without tracking entities). The revised `context_checks.py` diagnostic
  counterbalances query entity, mention order, and color assignment: 80 prompts
  across ten scenarios. It is versioned and tested against simple positional/color
  heuristics, but remains a narrow, correlated hand-written task. It is not a
  priority or a validated measure of general context tracking. Optional diagnostic:

```bash
python context_checks.py \
  --checkpoints hf_cache/xl_data100/checkpoint.pt hf_cache/xxl_data100/checkpoint.pt \
  --device cuda --output-dir analysis/followups/context_checks_balanced_v2
```

- **Epochs/convergence:** neither is needed for the fixed-time question. The current
  sampler uses replacement. Equal-time, equal-token, guaranteed-coverage, and
  convergence experiments are different controls.
- **Test data:** all 20 original test scores were inspected. Further choices use
  validation. Reused test evaluation may be descriptive after decisions are fixed;
  it is not fresh confirmation. No test-scoring command is part of this checklist.

## Files to send back

| Location | Return files | Purpose |
| --- | --- | --- |
| `analysis/followups/archived_validation/` | Entire directory | Common-score audit of saved models |
| `analysis/followups/seed_validation/` and `seed_summary.md` | Entire directory and summary | Main repeated-seed evidence |
| `runs/followups_1h/<run>/` | `metrics.csv`, `run_config.json`, `training_summary.json` | Actual recipe, curves, timing and exposure; preserve checkpoints |
| Optional `analysis/followups/equal_token_143m_validation/` and its summary | Entire directory and summary | Equal-exposure evidence |
| Optional `runs/followups_equal_tokens_143m/<run>/` | CSV/JSON files; preserve checkpoint | Check both runs reached the target |
| Optional `analysis/followups/stories/` | Samples, protocol, blind sheet, completed ratings | Supplementary story assessment |
| Optional `analysis/followups/context_checks_balanced_v2/` | Predictions and summary | Narrow diagnostic, kept separate from old probe results |
| `hf_cache/download_manifest.json` | Manifest only | Archived checkpoint provenance |

Scoring directories include `scores.csv`, `protocol.json`,
`paired_comparisons.json`, per-story `.npz` files, and saved training configs.
No need to send the already-downloaded multi-GB baseline checkpoints.
**Current decision:** stages 1 and 2 consistently favour XL under the shared
one-hour recipe. Recover the missing evaluation artifacts, then use stage 3 if
explaining the capacity/exposure trade-off is the next goal. It needs about
32 + 62 minutes at the new mean throughputs; keep its preplanned seed and schedule.
Four-hour training and broad new grids remain deferred. Equal-token and capability
results remain pending; no new training or re-evaluation was run during this analysis.
