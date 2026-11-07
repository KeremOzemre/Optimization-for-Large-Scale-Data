# Training a small GPT on TinyStories under a one-hour budget: a compute-optimal sweet spot

*DSA4212 — Optimisation for Large-Scale Data-Driven Inference, National University of Singapore*
*Authors: Steffany Lara, Kerem Ozemre, Tonio Kowalke, Ahmed Haj Hassine*

> This is the readable Markdown version. The submitted report is `report/report.tex`
> (same content). Figures are in `analysis/` and `report/figures/`, produced by
> `python analyze.py --runs-dir results`.

## Abstract

Under a fixed one-hour, from-scratch training budget on identical hardware, we train
**20 nanoGPT language models** on TinyStories: five model sizes (16–253M parameters)
crossed with four training-data fractions (25/50/75/100 % of the training split).
Holding the tokenizer, optimizer, schedule, context length, and effective batch
fixed, we find a clear **compute-optimal sweet spot**: performance improves steeply
from `small` to `xl` (≈124M parameters) and then **reverses** at `xxl` (≈253M),
because the largest model is so slow that it consumes roughly half as many tokens in
the hour and stays undertrained. The data fraction is almost irrelevant — no run
exhausts even 25 % of the split within the budget, so these runs are **compute-bound,
not data-bound**. Selecting by validation loss gives `xl_data50` (val loss 1.5025,
perplexity 4.49); on the held-out test split it reaches loss 1.4846, perplexity 4.41,
and 0.524 bits per byte.

## 1. Compute budget declaration

- **Budget:** one hour (3,600 s) of in-loop training wall-clock **per model**. Data
  preparation, final validation, and checkpointing are outside the timed loop.
- **Hardware:** a single **NVIDIA H100 NVL MIG 3g.47gb** slice, `bfloat16`, identical
  for all 20 runs.
- **From scratch:** every model starts from random weights (nanoGPT init); no
  pretrained weights, embeddings, or tokenizer fine-tuning.

Equal wall-clock does **not** imply equal token exposure — quantifying that gap is
central to our result (§5).

## 2. Research question

**Which model size and training-data fraction give the best language-modeling
performance under a fixed one-hour budget on identical hardware?** We hold the
tokenizer, optimizer, schedule, context length, and effective batch size fixed and
vary only (a) model size and (b) the fraction of the training split used.

## 3. Experimental setup

- **Data:** `roneneldan/TinyStories`. Official train split for training; the official
  validation split divided 80/20 into validation/test at the story level (seed 42).
  Test used only for the final comparison in §5.3.
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

- **Data fractions:** 25 / 50 / 75 / 100 % of the training split → a 5×4 grid of 20 runs.
- **Optimizer / schedule:** AdamW (lr 3e-4, wd 0.1, betas 0.9/0.95, clip 1.0),
  time-based 2 % warmup then cosine decay to 10 % of peak.

## 4. Methodology / reproducibility

1. `python run.py prepare --config configs/experiments/<name>.yaml` builds/reuses the
   token cache for that data fraction.
2. `python create_experiments.py` generates the 20 configs in `configs/experiments/`.
3. Each config trained with `python run.py train --config <name>.yaml`; outputs land
   in `results/<name>/`. All checkpoints + logs are mirrored to the Hugging Face Hub
   at `dsa4212-tinystories/tinystories-nanogpt`.
4. Model selected by **validation** loss; only then evaluated on **test**.
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

**(1) A compute-optimal sweet spot** (`val_loss_vs_params.png`). Validation loss falls
1.84 → 1.64 → 1.55 → 1.50 from `small` to `xl`, but the best `xxl` run (1.5106) is
**worse** than the best `xl` run (1.5025). `xxl` has ~2× the parameters of `xl` but
runs at ~half the throughput, so it sees only ~144M tokens in the hour vs ~274M for
`xl`, and stays undertrained.

**(2) Data fraction is nearly irrelevant** (`val_loss_vs_data.png`). Within each size
the four fractions differ by only ~0.01 nats and not even monotonically. No run
processes more than ~985M tokens, and even the 25 % subset contains far more tokens
than that, so training (sampling windows with replacement) never exhausts even the
smallest subset. These runs are **compute-bound, not data-bound**.

**(3) The core tension** (`val_loss_vs_tokens.png`, `throughput_vs_params.png`).
Throughput collapses with size — ~270k tokens/s for `small` down to ~40k tokens/s for
`xxl` (a ~7× gap in tokens seen). Larger models still reach lower loss per token —
until `xxl`, where too few tokens finally outweigh the extra capacity. The validation
curves (`val_loss_curves.png`) show `xl`/`xxl` still improving at the 1-hour mark,
i.e. compute-limited rather than converged.

### 5.3 Final test result

Selecting purely on validation loss gives **`xl_data50`** (val loss 1.5025, perplexity
4.49). Evaluated once on the untouched test split: **test loss 1.4846, perplexity
4.41, 0.5237 bits per byte**. The entire `xl` family is statistically
indistinguishable (val 1.5025–1.5118; the 0.004 gap by which `data50` edges `data100`
is within noise), so the robust takeaway is **"`xl` (≈124M parameters) at any data
fraction"** rather than 50 % specifically.

### 5.4 Qualitative evaluation

Generated stories (temperature 0.8, top-k 40; full set in `analysis/generations_all.md`)
agree with the metrics but compress the differences. Even `small` produces
grammatical, on-topic TinyStories; the gains with size are in **coherence over longer
spans** — the `xl` winner maintains one consistent plot and characters across a full
paragraph, whereas `small` more often loops ("painted a dog, and a dog") or drops a
thread.

## 6. Discussion

The winner wins because it sits at the compute-optimal point for this budget: large
enough to have capacity the smaller models lack, small enough to still see enough
tokens in an hour to use it. The experiment is firmly **compute-bound**: throughput,
not data availability, is the binding constraint — which is exactly why the data
fraction barely moves the loss and why `xxl` underperforms `xl`. With a larger budget,
`xxl` (and beyond) should eventually overtake `xl` once given enough tokens, and the
data fraction would start to matter once a run can consume a full subset.

**Limitations:** training samples windows with replacement, so one hour guarantees
neither a full epoch nor unique coverage; the single shared recipe (lr, warmup, batch)
is not tuned per size and may slightly favour some sizes; validation used sampled
windows rather than the exhaustive test pass; the TinyStories source revision is not
pinned.

## 7. Conclusion

Under a fixed one-hour, from-scratch budget on one H100 MIG slice, the best language
model on TinyStories is **`xl` (≈124M parameters)**; selected on validation,
`xl_data50` reaches test perplexity 4.41 (0.524 bits/byte). The single most important
trade-off is **throughput vs. capacity**: because larger models process far fewer
tokens in the same wall-clock time, "bigger" stops helping — and then hurts — well
before the largest model we trained, while the amount of training data available makes
almost no difference in this compute-bound regime.

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
