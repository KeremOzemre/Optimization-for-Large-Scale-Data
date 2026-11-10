# Training a small GPT on TinyStories: model size under a one-hour budget

*DSA4212 — Optimisation for Large-Scale Data-Driven Inference, National University of Singapore*
*Authors: Steffany Lara, Kerem Ozemre, Tonio Kowalke, Ahmed Haj Hassine*

> This is the readable Markdown version. The submitted report is `report/report.tex`
> (same content). Figures are in `analysis/` and `report/figures/`, produced by
> `python analyze.py --runs-dir results`.

## Abstract

Under a fixed one-hour, from-scratch training budget on identical hardware, we train
**20 nanoGPT language models** on TinyStories: five model sizes (16–253M parameters)
crossed with four training-data fractions (25/50/75/100 % of the training split).
Holding the tokenizer, optimizer, schedule, context length, and effective training
batch fixed, XL (≈124M parameters) has the lowest recorded validation loss among
the tested families. XXL (≈253M) processes roughly half as many tokens in the hour.
This supports a conditional result about the tested fixed-time recipe; it does not
establish an optimal model size, convergence, or why XXL performs worse. Data-fraction
effects are small in these single-seed runs. Full-data exposure ranges from 0.303
to 2.077 dataset-equivalents, and repeated random windows do not guarantee an epoch.
The original validation-selected `xl_data50` has test loss 1.4846 and perplexity
4.41. A supplied common-validation audit and six fresh one-hour full-data trials
(three seeds per finalist) strengthen the fixed-time observation: mean validation
CE is **1.494657 for XL versus 1.524387 for XXL**, with XL lower in every trial.
This is initial repeatability evidence for the shared recipe, not a significance
or universal-size claim. The local exports lack evaluation protocol files and
checkpoints, limiting independent reproduction. The short equal-token pair and
capability checks remain pending; four-hour runs are deferred.

## 1. Compute budget declaration

- **Budget:** one hour (3,600 s) of in-loop training wall-clock **per model**. Data
  preparation, final validation, and checkpointing are outside the timed loop.
- **Hardware:** a single **NVIDIA H100 NVL MIG 3g.47gb** slice, `bfloat16`, identical
  for all 20 runs.
- **From scratch:** every model starts from random weights (nanoGPT init); no
  pretrained weights, embeddings, or tokenizer fine-tuning.

Equal wall-clock does **not** imply equal token exposure — quantifying that gap is
central to our result (§5). The completed repetition study adds **six one-hour
training runs**, using the same recorded GPU allocation and training precision.
Offline evaluation time is additional.

## 2. Research question

**Which model size and training-data fraction give the best language-modeling
performance under a fixed one-hour budget on identical hardware?** We hold the
tokenizer, optimizer, schedule, context length, and effective batch size fixed and
vary only (a) model size and (b) the fraction of the training split used.

## 3. Experimental setup

- **Data:** `roneneldan/TinyStories`. Official train split for training; the official
  validation split divided 80/20 into validation/test at the story level (seed 42).
  Archived test scores exist for all 20 runs. Future choices use validation;
  this test set is no longer a fresh confirmation set for adaptive follow-ups.
- **Tokenizer:** GPT-2 (`tiktoken`), vocab 50,257, end-of-text token appended per
  story. Because the tokenizer is fixed, per-token loss and perplexity are directly
  comparable across all our runs.
- **Model:** nanoGPT GPT, tied embeddings, causal attention, context length 256,
  dropout 0.1. Five sizes:

  | Size   | Layers | Width | Heads | Params (M) | Microbatch × accum. |
  |--------|-------:|------:|------:|-----------:|:--------------------|
  | small  | 4      | 256   | 4     | 16.08      | 16 × 1 |
  | medium | 6      | 384   | 6     | 30.02      | 16 × 1 |
  | large  | 8      | 512   | 8     | 51.04      | 16 × 1 |
  | xl     | 12     | 768   | 12    | 123.75     | 8 × 2  |
  | xxl    | 16     | 1024  | 16    | 253.09     | 4 × 4  |

  The microbatch × accumulation is chosen so the **effective batch is a constant 16
  sequences** (4,096 tokens) per optimizer step for every size.

  Original in-loop validation used 16 microbatches per evaluation. Consequently,
  small/medium/large scored 65,536 targets, XL 32,768, and XXL 16,384. These are
  different validation samples despite the shared seed. Close cross-family gaps
  require common-target re-evaluation. The supplied audit reports full-validation
  scores; all six repetition configs use 4 × 64 × 256 = 65,536 in-loop targets.

- **Data fractions:** 25 / 50 / 75 / 100 % of the training split → a 5×4 grid of 20 runs.
- **Optimizer / schedule:** AdamW (lr 3e-4, wd 0.1, betas 0.9/0.95, clip 1.0),
  time-based 2 % warmup then cosine decay to 10 % of peak.

## 4. Methodology / reproducibility

1. `python run.py prepare --config configs/experiments/<name>.yaml` builds/reuses the
   token cache for that data fraction.
2. The checked-in configs and each archived `run_config.json` record the experiments.
   The generator is a convenience; archived settings are the authoritative record.
3. Each config trained with `python run.py train --config <name>.yaml`; outputs land
   in `results/<name>/`. All checkpoints + logs are mirrored to the Hugging Face Hub
   at `dsa4212-tinystories/tinystories-nanogpt`.
4. The original selection metric was **validation** loss. All 20 test results
   have subsequently been inspected and must be disclosed as exploratory comparisons.
5. `python eval_hf.py` downloads every checkpoint and scores the full test split;
   `python analyze.py --runs-dir results` aggregates everything into `analysis/`.

## 5. Results

All 20 runs, sorted by **final validation loss** (the selection metric). "Tokens" is
training tokens actually processed inside the one-hour budget. Test numbers are shown
for completeness but were **not** used for selection (§5.3).

| Run            | Params (M) | Data % | Tokens (M) | Val loss | Val ppl | Test ppl |
|----------------|-----------:|-------:|-----------:|---------:|--------:|---------:|
| **xl_data50**  | 123.75     | 50     | 273.9      | **1.5025** | **4.49** | 4.41 |
| xl_data100     | 123.75     | 100    | 279.8      | 1.5068   | 4.51    | 4.40 |
| xl_data75      | 123.75     | 75     | 269.1      | 1.5099   | 4.53    | 4.43 |
| xxl_data100    | 253.09     | 100    | 143.7      | 1.5106   | 4.53    | 4.56 |
| xl_data25      | 123.75     | 25     | 283.3      | 1.5118   | 4.54    | 4.45 |
| xxl_data75     | 253.09     | 75     | 141.1      | 1.5125   | 4.54    | 4.59 |
| xxl_data50     | 253.09     | 50     | 148.2      | 1.5155   | 4.55    | 4.56 |
| xxl_data25     | 253.09     | 25     | 143.2      | 1.5233   | 4.59    | 4.61 |
| large_data100  | 51.04      | 100    | 562.4      | 1.5460   | 4.69    | 4.66 |
| large_data75   | 51.04      | 75     | 562.0      | 1.5466   | 4.70    | 4.67 |
| large_data50   | 51.04      | 50     | 557.9      | 1.5479   | 4.70    | 4.69 |
| large_data25   | 51.04      | 25     | 563.2      | 1.5509   | 4.72    | 4.70 |
| medium_data100 | 30.02      | 100    | 734.6      | 1.6444   | 5.18    | 5.16 |
| medium_data75  | 30.02      | 75     | 738.3      | 1.6462   | 5.19    | 5.17 |
| medium_data50  | 30.02      | 50     | 737.3      | 1.6527   | 5.22    | 5.20 |
| medium_data25  | 30.02      | 25     | 719.1      | 1.6558   | 5.24    | 5.21 |
| small_data75   | 16.08      | 75     | 940.8      | 1.8383   | 6.29    | 6.28 |
| small_data100  | 16.08      | 100    | 984.3      | 1.8386   | 6.29    | 6.26 |
| small_data50   | 16.08      | 50     | 985.5      | 1.8421   | 6.31    | 6.29 |
| small_data25   | 16.08      | 25     | 972.0      | 1.8452   | 6.33    | 6.31 |

**Three findings (figures in `analysis/`):**

**(1) Best observed size under the shared recipe** (`val_loss_vs_params.png`). Validation loss falls
1.84 → 1.64 → 1.55 → 1.50 from `small` to `xl`, but the best `xxl` run (1.5106) is
**worse** than the best `xl` run (1.5025). `xxl` has ~2× the parameters of `xl` but
runs at ~half the throughput, so it sees only ~144M tokens in the hour vs ~274M for
`xl`. Reduced exposure is a plausible explanation, alongside optimizer suitability
and training randomness; it has not been isolated experimentally. One seed and
unequal validation samples do not justify a precise XL–XXL ranking from this original
grid alone. The supplied common-target audit and repetitions are reported separately below.

**(2) Small observed data-fraction effects** (`val_loss_vs_data.png`). Within each size
the observed spread is about 0.007–0.014 nats and is not consistently monotonic.
Nevertheless, the 100%-data validation loss is lower than the 25%-data loss in
every family. This is a small observed benefit, not evidence that data amount is irrelevant.
The 25% cache contains **118,450,614 tokens** and the full cache **473,992,236**;
the earlier claim that even the smallest cache exceeded every run's exposure was
incorrect. Small/25% processed **8.206 cache-equivalents**, so a blanket explanation
that training was too short to revisit data is untenable. The weak fraction
effect may reflect a redundant, homogeneous corpus and this recipe, but neither its
cause nor its stability across training seeds has been established.

**(3) The core tension** (`val_loss_vs_tokens.png`, `throughput_vs_params.png`).
Throughput collapses with size — ~270k tokens/s for `small` down to ~40k tokens/s for
`xxl` (a ~7× gap in tokens seen). Larger models still reach lower loss per token —
under the tested time budget. The validation curves (`val_loss_curves.png`) show
late improvements; they do not establish convergence or identify a unique bottleneck.

### 5.2 Token exposure audit

Every logged counter matches `step × microbatch × accumulation × context_length`.
For the 100%-data runs:

| Model | Training targets | Full-cache equivalents | Estimated time for one equivalent |
| --- | ---: | ---: | --- |
| small | 984,326,144 | 2.077 | 28m 54s |
| medium | 734,642,176 | 1.550 | 38m 43s |
| large | 562,429,952 | 1.187 | 50m 34s |
| XL | 279,830,528 | 0.590 | 1h 42m |
| XXL | 143,654,912 | 0.303 | 3h 18m |

Time estimates divide cache size by measured training throughput on the original
hardware; they are estimates, not stopping guarantees. Sampling is with replacement,
so an equivalent counts repeated targets and does not establish unique-token coverage.
See `analysis/token_exposure_and_epoch_estimates.csv` for all 20 runs.

### 5.3 Final test result

Selecting purely on validation loss gives **`xl_data50`** (val loss 1.5025, perplexity
4.49). Its archived test result is **loss 1.4846, perplexity 4.41, 0.5237 bits per
byte**. The full-data XL test loss is 1.4824. These scores do not prove that 50%
is preferable to 100%. There are no repeated training seeds or uncertainty estimates
supporting the previous claim that these differences are statistically indistinguishable.
The archived full-test protocol scores 957,184 targets and drops the short tail.
The new full evaluator is designed to include the tail and record that protocol
change. The supplied follow-up CSVs report the intended target count, but the
corresponding protocol JSON files have not been supplied locally.

### 5.4 Qualitative evaluation

Generated stories (temperature 0.8, top-k 40; full set in `analysis/generations_all.md`)
provide illustrative examples. Even `small` produces
grammatical, on-topic TinyStories; the gains with size are in **coherence over longer
spans** — the `xl` winner maintains one consistent plot and characters across a full
paragraph, whereas `small` more often loops ("painted a dog, and a dog") or drops a
thread. These impressions came from a small, unblinded sample and are not a quantified
capability result. Frozen prompts, two sampling seeds, and blind rating sheets are
a secondary follow-up, with results pending. The original 20 color probes always
queried the first-mentioned object, allowing a first-color copying rule to score
100%. Their replacement is a versioned 80-prompt diagnostic that counterbalances
the queried object, mention order, and color assignment. Its scores remain pending
and would not establish general context-tracking or reasoning ability.

### 5.5 Exploratory size and compute plots

Using only the trained runs, we estimate each run's training compute with the
approximation **C ≈ 6ND** (N = parameters, D = tokens processed). It is a proxy,
not measured FLOPs or GPU utilization. A quadratic through five size-level
observations (`compute_optimal_fit.png`) is descriptive only: its vertex is sensitive
to the chosen points, functional form, single seed, and evaluation differences.
It does not establish a compute-optimal parameter count or a scaling law. The updated
plot uses 100% data at every size to avoid selecting a different fraction per point.

### 5.6 Varying the budget and the context length

Existing additional runs are exploratory because more than one condition differs.

**Budget.** We reran `large`, `xl`, `xxl` at a 30-minute budget instead of an hour. The
winner moves: at 30 min `large` (51M) leads with val loss 1.580, ahead of `xl` (1.633)
and `xxl` (1.680), whereas at a full hour `xl` was best. These are recorded rankings
at two budgets (`budget_sweep.png`). However, the additional
runs record `NVIDIA H100 NVL` and the main grid records `NVIDIA H100 NVL MIG 3g.47gb`.
Validation sample sizes also differ. This does not isolate a budget effect.
The updated plot holds the data fraction at 100%; hardware must still be reconciled.

**Context length.** Keeping `xl` and the one-hour budget, raising the context from 256
to 512 tokens drops val loss from 1.5068 to 1.4555 (test perplexity 4.40 → 4.31), even
though the longer-context run gets through fewer tokens (190M vs 280M). Its hardware
label differs, its effective targets per update double (8,192 vs 4,096), and its
validation windows differ. The observed improvement cannot be attributed solely to context.

### 5.7 Completed common-validation audit: supplied exports

`analysis/followups/archived_validation/scores.csv` reports the following scores
for the archived checkpoints, each on **3,808,643 validation targets**:

| Archived model | Validation CE | Perplexity |
| --- | ---: | ---: |
| Large / 100% | 1.546940 | 4.697073 |
| XL / 50% | 1.491074 | 4.441863 |
| XL / 100% | **1.489276** | **4.433885** |
| XXL / 100% | 1.523562 | 4.588538 |

On these exports, the full-data XXL−XL gap is **0.034285 nats**, compared with
0.003782 in the original unequal sampled validation. The common audit supports
XL's advantage for these checkpoints and shows why the original small gap was
not the right close-comparison estimate. XL/100% also edges XL/50% by 0.001798
nats, reversing their original sampled-validation ranking. This tiny single-seed
fraction difference does not establish that full data is reliably better than 50%.

The scored count matches the intended complete validation stream excluding its
first token. However, neither this directory nor the repetition-scoring directory
contains `protocol.json`, per-story NPZ files, or `paired_comparisons.json` locally.
The exact offline context policy, precision, validation hash, and story-level
intervals therefore cannot be independently checked from the supplied exports.

### 5.8 Completed one-hour repetitions: three seeds per finalist

The six supplied repetition runs retain full data and the shared recipe, varying
training seed across 42/43/44. `analysis/followups/seed_validation/scores.csv`
reports 3,808,643 scored validation targets per checkpoint:

| Seed | XL CE | XXL CE | XXL−XL CE | XL training targets (M) | XXL training targets (M) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 42 | 1.490381 | 1.529798 | +0.039417 | 271.016 | 138.121 |
| 43 | 1.495970 | 1.530280 | +0.034309 | 267.911 | 133.325 |
| 44 | 1.497618 | 1.513084 | +0.015466 | 267.891 | 146.387 |

XL's mean CE is **1.494657**, with sample SD **0.003793**; XXL's is **1.524387**,
with sample SD **0.009792**. The mean paired gap is **+0.029731 nats**, favouring
XL. Exponentiating mean CE gives geometric-mean perplexity **4.457805 versus
4.592329**, about **2.93% lower for XL**. Every XL run scores below every XXL run
in the supplied table. Three seeds provide initial repeatability evidence; these
statistics do not establish a 95% significance claim or a universal ranking.

The saved training configurations match the intended controls and record the
same hardware label, training-cache hash, code hashes, and bfloat16 precision.
All **29,961 supplied log rows** have consistent `step × 4,096` token counters,
and final losses are finite. Recorded loop durations span **3,600.008–3,600.101
seconds**. Mean processed targets are **268.939M for XL and 139.278M for XXL**,
a ratio of about **1.931×**. Equal time continues to imply unequal exposure.

The observed variation combines training randomness and throughput variability:
XXL/seed44 processes about 9.8% more targets than XXL/seed43. Consequently, its
lower loss cannot be attributed to the numeric seed alone. Checkpoints and
`training_summary.json` files are not available locally, so weights cannot be
reloaded and the exported offline scores or separate final-evaluation/save timing
cannot be reproduced here. This is an analysis of supplied results, not a new local run.

## 6. Discussion

The main grid is useful evidence about a practical fixed-time training recipe.
Equal time deliberately allows models to process different token counts; that does
not invalidate the fixed-time question. It does limit conclusions about intrinsic
capacity, convergence, and equal-exposure performance. Whether XXL overtakes XL at
a larger budget is an open question, rather than a predicted result.

The common-validation audit and three-seed repetitions strengthen XL's fixed-time
advantage under the tested recipe. They do not establish that unequal exposure
caused the difference. The next targeted explanatory comparison is the already
specified short equal-token pair, with fresh models and a common step-based schedule.

**Limitations:** the original grid has one seed per configuration; the repetition
study has only three per finalist. The shared recipe is not tuned per architecture,
training samples with replacement, throughput varies between runs, the source
revision was not pinned, and all 20 original test scores have been inspected.
Missing offline protocol/checkpoint artifacts limit independent verification.
The training-config hashes and supplied aggregate scores do not substitute for
reproducible evaluation artifacts or justify a significance claim.

### 6.1 Follow-up status and next experiment

| Check | Ready files / tool | Purpose | Status |
| --- | --- | --- | --- |
| Archived Large/full, XL/50%, XL/full, XXL/full audit | `analysis/followups/archived_validation/scores.csv` | Common-validation comparison | Completed as supplied; protocol/story artifacts missing locally |
| XL/XXL, full data, seeds 42/43/44, 1 hour each | `analysis/followups/seed_validation/scores.csv`, supplied logs/configs | Initial fixed-time repeatability; 6 GPU-hours | Completed as supplied; checkpoints/summary/protocol artifacts missing locally |
| Optional after interpretation: XL/XXL, 35,000 updates each | Two `*equal_tokens*.yaml` | Equal 143,360,000 targets, step-based schedule; about 1.5 GPU-hours | Not trained |
| Supplement: frozen prompts, two generation seeds, two raters | `sample_stories.py` | Check whether observed loss differences are noticeable in stories | Not rated |
| Exploratory diagnostic: 80 counterbalanced color prompts | `context_checks.py` | Remove the original first-color shortcut; narrow task only | Not evaluated |
| Deferred: XL/XXL, full data, seed 42, 4 hours each | Two `*4h*.yaml` | A separate larger-budget question costing about 8 GPU-hours | Deferred; not trained |

The completed training configs fix preparation seed 42, evaluation seed 43, and
65,536 in-loop validation targets. New training starts from random weights;
there is no resume command. Recover the missing evaluation artifacts for the
completed comparisons. The next optional explanatory pair uses **35,000 × 4,096
= 143,360,000 targets**. At the original measured throughputs this is roughly 31
minutes for XL and 60 minutes for XXL; a two-hour safety cap per model marks early
stops as incomplete. Both are fresh runs with the same step-based schedule, not
comparisons against the old time-scheduled checkpoint. One cache-equivalent has no
special statistical status and equal targets do not guarantee an epoch.

The supplied repetitions consistently favour XL, while three seeds remain an
initial repeatability check. This is a shared-recipe comparison; it does not
optimize learning rates or other settings separately for each architecture.
No story-resampling intervals have been supplied locally. Such intervals would
not substitute for training repeats. A later symmetric
learning-rate sensitivity study or one-model study of much smaller data subsets
would address separate questions. The deferred four-hour pair uses a stretched
budget-specific schedule and is not continuation. Exact commands, result paths,
and reporting rules are in `FOLLOWUP_COMMANDS.md`.

## 7. Conclusion

Under the recorded one-hour H100 MIG budget and shared recipe, **XL (≈124M
parameters) consistently outperforms XXL in the supplied three-seed full-data
comparison**: mean CE 1.494657 versus 1.524387, corresponding to about 2.93% lower
geometric-mean perplexity. The common-validation archive audit supports the same
direction. This strengthens the original fixed-time observation without proving
a universal best size, significance, convergence, or the cause of the gap.
Offline protocol/checkpoint artifacts are still needed for independent reproduction.
Data-fraction findings remain based on the original single-seed grid. Equal-token
training and capability ratings are pending; their outcomes are not evidence for
the present conclusion.

## Appendix — generated story samples

Full samples for all 20 runs (five prompts each) are in `analysis/generations_all.md`.
Illustration (prompt *"Once upon a time"*):

- **small_data25:** *"…there was a little girl named Lily. Lily loved to play with her
  toys… She painted a picture of a flower and a house… Later that day, Lily's mom asked
  her to help. Lily was happy to help and started to decorate the room with pretty
  flowers."* (The thread loops back on itself.)
- **xl_data50 (selected):** *"…there was a little girl named Lily. Lily loved to play
  with her toys, especially her teddy bear. One day, Lily's mommy asked her to help
  decorate the house for a party… 'Mommy, can I carry my teddy bear?' she asked… They
  arrived at the living room and Lily's mommy gave her a big box of presents."* (One
  consistent, advancing plot.)
