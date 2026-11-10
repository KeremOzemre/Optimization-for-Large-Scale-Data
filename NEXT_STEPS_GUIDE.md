# Why these TinyStories follow-ups come next

Use [FOLLOWUP_COMMANDS.md](FOLLOWUP_COMMANDS.md) for the exact commands, configuration
paths, budgets, and files to return. This document explains the focused design.
**The common-validation audit and six one-hour repetitions have now returned
results.** XL leads in all three seed pairs, with mean CE 1.494657 versus 1.524387.
See [the completed analysis](analysis/followups/review/ANALYSIS.md). Training logs
and controls are internally consistent; missing scoring protocol/story artifacts
and new checkpoints limit independent re-evaluation. The short equal-token pair
and capability checks remain pending, and four-hour runs remain deferred.

## The question we are confirming

> Under our shared training recipe and one-hour budget on the same GPU allocation,
> which of the tested model configurations predicts TinyStories best?

This is a practical question about performance per available training hour. Different
models processing different token counts is part of the trade-off. It does not
invalidate the question, and finishing an epoch is not a prerequisite.

The original grid explored five model sizes and four data fractions. The follow-up
has now supplied common validation and three fresh one-hour trials per finalist.
These consistently favour XL under the shared recipe, without proving intrinsic
size superiority. Recover the missing evaluation artifacts; the short equal-token
pair is the next targeted explanatory experiment. The sections below retain the
design rationale. Four-hour runs are deferred.

## What the existing results establish

| Existing asset | What to build on | Limitation |
| --- | --- | --- |
| Twenty one-hour runs in `results/` | Recorded configs, curves, throughput, processed targets | One training seed per configuration |
| Full-data XL validation CE 1.506811, XXL 1.510593 | A close recorded gap worth checking | Different sampled validation targets |
| Archived common-window test scores | Descriptive evidence from all 20 runs | Already inspected; not fresh confirmation for adaptive choices |
| Four downloaded checkpoints in `hf_cache/` | Large/full, XL/50%, XL/full, XXL/full can be re-scored | Old runs lack cache hashes |
| Token-exposure audit | Exact completed-update counters, repeated exposure, planning throughput | Counts do not prove unique coverage |
| Existing generated stories | Examples for a small supplementary assessment | Five prompts and unblinded inspection do not quantify capability |

The full cache contains 473,992,236 tokens. The original full-data exposure was
2.077 equivalents for Small, 1.550 for Medium, 1.187 for Large, 0.590 for XL, and
0.303 for XXL. Equivalents divide processed targets by cache size; repeated targets
are counted again. Equal time, equal targets, complete coverage, and convergence
are separate experimental conditions.

## 1. Common validation first

Original validation used the training microbatch size: Small/Medium/Large scored
65,536 targets, XL 32,768, and XXL 16,384. Close gaps cannot be assessed reliably
by comparing those aggregate values alone.

Re-score the four saved models with `score_checkpoints.py` on the same complete
validation stream, precision, and 256-token context policy. Save per-story losses,
counts, protocol metadata, and checkpoint/cache hashes. The full evaluator includes
the short tail; label its scores separately from the historical sampled protocol.

This costs inference only. It can change our interpretation before we spend another
training hour. If the common scores differ sharply from old records, reconcile the
cache and protocol before retraining. Large gives a smaller reference; XL/50% checks
the original validation-based selection. The repeat comparison fixes full data for
both XL and XXL to keep architecture and data availability from changing together.

Story-level bootstrap intervals describe evaluation-sample variation conditional
on fixed checkpoints. They do not measure the variation from retraining. Duplicate
stories and context crossing EOT also weaken an independent-story interpretation.
Use those intervals as diagnostics, not as a substitute for repeated seeds.

## 2. Confirm the one-hour comparison with six fresh trials

The six `*1h*.yaml` configs specify XL/XXL with training seeds 42, 43, and 44.
They share the exact cache, preparation seed 42, evaluation seed 43, optimizer,
time schedule, 4,096 training targets/update, and 65,536 sampled validation targets.
Microbatch/accumulation remains 8x2 for XL and 4x4 for XXL. Run sequentially on
matching hardware and software, for about six GPU-hours plus final evaluation/setup.

All six are fresh starts. The corrected validation workload differs from the old
runs, and historical cache identity is unrecorded, so old seed 42 does not silently
replace a protocol-matched repeat. Same numeric seeds across differently shaped
models do not align initialization noise. They do align the intended data-sampling
setup; hardware numerical nondeterminism can remain.

After training, evaluate all six on common full-validation targets. Report every
seed, family mean/sample standard deviation, paired XXL-minus-XL loss differences,
throughput, time, and processed targets. Positive differences favour XL. Keep all
seeds, including inconvenient outcomes; do not report only the best seed.

Possible interpretations:

- A consistent advantage, large relative to variation: evidence for a repeatable
  winner under this shared recipe and allocation.
- Small or mixed differences: an uncertain ranking; report that rather than chasing
  a preferred winner with more selective runs.
- Changed hardware, cache, or failed/stopped trials: resolve comparability first.

Three seeds give an initial variability estimate, not a precise uncertainty guarantee.
A CE difference `d` changes perplexity by a factor `exp(d)`, so distinguish a detectable
gap from a practically important improvement. No optimizer search per model was done;
this experiment does not establish the best possible architecture-specific recipe.
**Stop here and interpret before adding more training.**

## 3. Optional short equal-token pair

Question:

> With the same training targets and step-based learning-rate sequence, does the
> XL/XXL ranking change?

The revised pair uses **35,000 updates x 4,096 = 143,360,000 targets per model**.
That is near XXL's original one-hour exposure. The old full-cache-equivalent target
was arbitrary: with random sampling, one equivalent is neither an epoch nor a
convergence milestone. At the original rates, the short pair takes roughly 31
minutes for XL and 60 minutes for XXL, about 1.5 GPU-hours combined.

Both runs start fresh at seed 42, reuse the full cache, and use `schedule.progress:
steps`: 700 warmup updates then cosine decay. Their window sequence matches by
update despite microbatch grouping; dropout draws and rounding can still differ.
Do not substitute an old time-scheduled checkpoint for a new step-scheduled run.

Each YAML has `max_steps: 35000` and a 7,200-second safety cap. Fresh output root
`runs/followups_equal_tokens_143m/` separates this protocol from the earlier 474M
proposal. Both summaries must report `max_steps`, 35,000 steps, and 143,360,000
targets. A time-capped run is incomplete; there is no resume interface.

A single pair is exploratory. It controls exposure between the two models, while
runtime and approximate compute differ. Comparing it with time-scheduled runs does
not isolate exposure as the sole cause of a ranking change. Keep the conclusion
bounded to the compared protocols.

## 4. Small supplementary story assessment

`sample_stories.py` supplies 20 frozen prompts, two sampling seeds per model,
EOT stopping, recorded length-cap termination, and a blind two-rater sheet. The two
archived full-data finalists produce 80 stories. Have raters independently assess
grammar, plot progression, consistency, and repetition before revealing identities.
Report disagreement and paired prompt/seed observations. A length-capped story is
not automatically incoherent. Ratings can show whether a loss gap is perceptible;
they neither verify its numerical ranking nor establish general reasoning ability.

The original color test had a shortcut: the answer was always the first color,
allowing 20/20 without entity tracking. The revised v2 diagnostic counterbalances
query entity, mention order, and color assignment (80 prompts, ten scenarios).
Simple constant-color and first/last-color rules now score 50%. This removes those
specific shortcuts, not all possible shortcuts. Keep it a low-priority, narrow
diagnostic; do not compare its scores with the flawed v1 protocol.

## Deferred questions

| Question | Why it is deferred / what would be needed |
| --- | --- |
| Does XXL overtake XL after four hours? | Eight GPU-hours for one pair is lower priority than repeatability. Its schedule stretches too; a positive result would still need replication. Four-hour YAMLs remain deferred templates. |
| Why are data-fraction effects small? | This needs a separate focused study, not longer training for every old configuration. Consider one model at 1/5/25/100% data with fixed exposure and repeated subsets/seeds. |
| Does the learning rate favour XL? | A small symmetric sensitivity search can test the shared recipe, but needs its own budget and fixed selection protocol. Do not tune only the loser. |
| What is the compute-optimal parameter count? | Five families, one seed, and a descriptive quadratic do not establish a scaling law or universal optimum. The 6ND proxy is not measured GPU utilization. |
| Would longer context help? | Historical context runs also changed hardware labels and targets/update. A controlled context comparison is a new experiment. |
| Do all positions need to be seen? | Guaranteed coverage would require a defined without-replacement sampler. It is unnecessary for the fixed-time question. |

In the current grid, all five families have lower recorded validation CE with 100%
than with 25% data, though gaps are small (about 0.005-0.013 nats). Small/25% already
processed 8.206 cache-equivalents, so a blanket explanation that no model had enough
time to revisit text is contradicted by the exposure counts. Redundancy, sufficient
data at this scale, and recipe limitations are hypotheses, not established causes.
For an overfitting investigation, measure a fixed training sample in evaluation
mode; changing minibatch losses with dropout are not a clean generalization gap.

## Reporting and stopping rules

Keep old results separate from new experiments and retain all configs, hashes,
metrics, summaries, and checkpoints. The dataset revision remains unpinned; preserve
the actual cache rather than relying on the dataset name. All 20 original test
scores were inspected, so further model selection uses validation. Reused test
scores can be descriptive, but are not fresh confirmation.

The approved order is **common validation -> six one-hour trials -> interpret ->
optional short equal-token pair**. The first two stages are completed as supplied;
the new mean rates suggest about 32 minutes for XL and 62 for XXL in the optional
pair (roughly 1.6 GPU-hours). Keep the preplanned seed 42 and common step schedule;
do not choose the best observed seed after seeing the results. No four-hour run,
broad grid, or capability score is required to report the fixed-time observation.
Three-seed variability also includes throughput differences. The report now
distinguishes completed supplied results from pending exposure/capability checks.
