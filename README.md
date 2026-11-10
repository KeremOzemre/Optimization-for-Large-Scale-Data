# TinyStories GPT from scratch

A small, single-device GPT language model trained on [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories). Its model architecture and initialization are adapted from [Andrej Karpathy's nanoGPT, commit `3adf61e154c3fe3fca428ad6bc3818b27a3b8291`](https://github.com/karpathy/nanoGPT/tree/3adf61e154c3fe3fca428ad6bc3818b27a3b8291), under the included [MIT license](LICENSE). Training starts with random weights. See [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) for the research question, confirmed decisions, and current limitations.

## Install

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Configure an experiment

The common-validation audit and six one-hour follow-up runs have returned results: XL has lower reported full-validation loss in all three seed pairs (mean CE 1.494657 versus 1.524387), with about 2.93% lower geometric-mean perplexity. See [the analysis and plots](analysis/followups/review/ANALYSIS.md) for the control audit and missing evaluation artifacts. [FOLLOWUP_COMMANDS.md](FOLLOWUP_COMMANDS.md) records the completed commands and next optional 143,360,000-target pair (about 1.6 GPU-hours at the new measured rates). Four-hour runs remain deferred; [NEXT_STEPS_GUIDE.md](NEXT_STEPS_GUIDE.md) explains the design.

See [CONFIGURATION.md](CONFIGURATION.md) for a table explaining every setting, its allowed values, and its effect.

Edit a copy of [configs/default.yaml](configs/default.yaml) for each run. It is the central place for the data, tokenizer, model, training, optimizer, schedule, evaluation, and generation settings. Give every run a new `experiment.name` and **unused** `experiment.output_dir`; training refuses to overwrite an existing output folder. [configs/smoke.yaml](configs/smoke.yaml) is a short example using the small token cache. The old script-specific training flags have been replaced by `--config` on `run.py`.

The first model-size comparisons should change `model.layers`, `model.width`, and possibly `model.heads`, while keeping the rest of the training recipe, data, and hardware fixed. `model.width` must be divisible by `model.heads`. `model.context_length` controls how many tokens the model sees at once; `training.microbatch_size` counts sequences per microbatch; `training.gradient_accumulation_steps` counts microbatches per optimizer step. These settings are different from the model's parameter count and total tokens processed. The actual parameter count and token throughput are saved with each run.

## Quick check with small data

The small cache contains 2,000 training stories and 200 stories from each held-out split. A compatible cache is reused automatically. The first preparation may still need to download TinyStories.

```bash
python run.py prepare --config configs/smoke.yaml
python run.py train --config configs/smoke.yaml
python run.py evaluate --config configs/smoke.yaml
python run.py generate --config configs/smoke.yaml --prompt "Once upon a time"
```

You can use `python run.py run --config configs/smoke.yaml` to prepare, train, and evaluate in one command. **This repository has already used `out/config-smoke` for verification.** To repeat the smoke run, copy `configs/smoke.yaml` and set `experiment.output_dir` to a new path before training. Generation prints text to the terminal.

## Full preparation and one-hour run

Review `configs/default.yaml` and choose an unused output folder. Then run:

```bash
python run.py prepare --config configs/default.yaml
python run.py train --config configs/default.yaml
python run.py evaluate --config configs/default.yaml
python run.py generate --config configs/default.yaml --prompt "Once upon a time"
```

`python run.py run --config configs/default.yaml` combines the first three steps, **including test evaluation**. For model-size comparisons, run preparation and training separately, choose a model from validation results, then evaluate that model on test. The full preparation can take time and disk space; it has not been run automatically. `--force` on `prepare` or `run` rebuilds a token cache when you intentionally change preparation settings. Otherwise, an incompatible existing cache is rejected. The old `data/tinystories` Arrow copy is not read by this pipeline; the `data/tokens-small` and `data/tokens-full` directories hold the training token caches.

## What each step does

`prepare` loads `roneneldan/TinyStories` with Hugging Face Datasets, splits its official validation stories deterministically into validation and test, and uses the GPT-2 tokenizer from `tiktoken`. It appends an end-of-text token after every story and writes `train.bin`, `val.bin`, `test.bin`, and `meta.json`. The model vocabulary size is derived from the tokenizer (50,257 for GPT-2). The test split stays separate from training and validation.

`train` samples random token windows from `train.bin`, predicting shifted next tokens with causal attention. The default AdamW schedule warms up and decays over 3,600 seconds **inside the training loop**, including periodic validation/logging. The current optimizer step finishes when the limit is reached. Preparation, initialization, final validation, and saving are outside the budget. Optional `training.max_steps` stops after an exact update count; use `schedule.progress: steps` for equal-token studies. The time cap remains a safety stop and the saved summary identifies which limit ended training. Sampling with replacement does not guarantee an epoch.

Preparation, training, evaluation, and generation now have separate seeds. New configs set them explicitly; legacy YAMLs retain their derived defaults. Validation also has its own microbatch size, so changing model training memory settings need not change the validation targets.

Training writes these files in `experiment.output_dir`:

| File | Contents |
| --- | --- |
| `run_config.json` | Resolved YAML settings, actual device and precision, tokenizer vocabulary size, parameter count, and prepared token counts. |
| `metrics.csv` | Step, elapsed seconds, training loss, validation loss and perplexity when measured, learning rate, cumulative training tokens, and average tokens per second. |
| `checkpoint.pt` | Model and optimizer states, model configuration, seed, elapsed training time, processed tokens, and resolved run settings. |
| `training_summary.json` | Completed steps/targets, stop reason, final validation score, and separate training/evaluation/checkpoint-saving durations. |

`evaluate` scores fixed sampled test windows. `score_checkpoints.py` scores multiple checkpoints on common validation targets, saving per-story losses and paired uncertainty diagnostics; its test mode is explicit. Close CE comparisons require the same tokenizer, targets, and context policy. `generate` uses the YAML prompt, length cap, temperature, top-k, and independent generation seed, stopping at EOT. Both `evaluate` and `generate` accept `--checkpoint`. `sample_stories.py` creates supplementary blind rating material. The v2 `context_checks.py` diagnostic balances queried object, mention order, and colors across 80 prompts; it replaces a flawed positional-copying test and remains exploratory.

## Checks

```bash
python -m unittest discover -s tests -v
```

The tests cover shifted targets, causal masking, tied weights, tiny-batch overfitting, independent seeds, config validation, weighted evaluation, EOT stopping, step/time stopping with checkpoint loading, equal-token budget completion checks, and balanced probe design. All 19 CPU tests passed during the focused-plan update; new GPU experiments have not been run. Smoke-generated text is not a quality benchmark.

## Scientific paper version

[report2.pdf](report/report2.pdf) is a separate paper-style account of the study,
with its source in [report2.tex](report/report2.tex). The eight-page, two-column
paper covers all 30 completed research runs: the original 20-run grid, the
30-minute and longer-context extensions, and the six repeated trials. It also
includes the historical test results and generated examples, token-exposure
and approximate-compute analyses, common-validation audit, concise training
methods, and a nine-run pilot/development inventory. Its bibliography, tables,
and four figures are stored beside the original report's assets. Differences
in hardware and scoring remain explicit; planned studies are marked pending.
The original report remains available separately.

To build the second report with a LaTeX installation, run from the project root:

```bash
(
set -e
cd report
pdflatex -interaction=nonstopmode -halt-on-error report2.tex
bibtex report2
pdflatex -interaction=nonstopmode -halt-on-error report2.tex
pdflatex -interaction=nonstopmode -halt-on-error report2.tex
)
```

## Reproducing the original report (DSA4212 submission)

Repository layout required by the course guidelines:

- `readme.md` (this file): setup, data, experiments, reproduction.
- `authors.md`: name and student number of every group member.
- `report/`: LaTeX source of the submitted report (`report/report.tex`) and its figures.
- `analysis/`: tables and figures generated by `analyze.py` and `eval_hf.py`.
- `results/<run>/`: per-run `run_config.json`, `metrics.csv`, `test_results.json`, `generations.md`.

Compute budget: every final run is one hour (3,600 s) of in-loop training on one
NVIDIA H100 NVL MIG 3g.47gb slice, bfloat16, from random weights. Twenty runs:
five model sizes (`small` 16M, `medium` 30M, `large` 51M, `xl` 124M, `xxl` 253M
parameters) times four training-data fractions (25/50/75/100 % of the TinyStories
train split). The checkpoints and training logs of all twenty runs are on the
Hugging Face Hub at `dsa4212-tinystories/tinystories-nanogpt`.

To regenerate every number and figure in the report without retraining:

```bash
python eval_hf.py                        # downloads each checkpoint, test loss / ppl / bits-per-byte, story samples
python analyze.py --runs-dir results     # summary.csv, summary_table.md, figures into analysis/
```

To retrain one run from scratch (needs a CUDA GPU; one hour per run):

```bash
python run.py prepare --config configs/experiments/<name>.yaml
python run.py train   --config configs/experiments/<name>.yaml
```

The experiment configs are produced by `python create_experiments.py` from
`configs/default.yaml`; the training recipe (AdamW, lr 3e-4, weight decay 0.1,
betas 0.9/0.95, clip 1.0, 2 % time-based warmup then cosine to 10 % of peak,
context 256, dropout 0.1) is the same for all twenty runs.
