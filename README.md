
## Authors

DSA4212 Optimisation for Large-Scale Data-Driven Inference, National University of Singapore, AY2026/27 Semester 1.

| Name | Student number |
|------|----------------|
| Kerem Ozemre | A0350378Y |
| Tonio Kowalke | A0350429E |
| Steffany Lara | A0350565H |
| Ahmed Haj Hassine | A0342398U |


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

`train` samples random token windows from `train.bin`. Each window predicts the next token at every position, with causal attention. It uses AdamW and a time-based warmup and cosine learning-rate schedule. The default budget is 3,600 seconds **inside the training loop**, including periodic validation and logging. The current optimizer step finishes when the limit is reached. Dataset preparation, model setup, final validation, and checkpoint saving are outside this budget. An epoch is not guaranteed because windows are sampled with replacement.

Training writes these files in `experiment.output_dir`:

| File | Contents |
| --- | --- |
| `run_config.json` | Resolved YAML settings, actual device and precision, tokenizer vocabulary size, parameter count, and prepared token counts. |
| `metrics.csv` | Step, elapsed seconds, training loss, validation loss and perplexity when measured, learning rate, cumulative training tokens, and average tokens per second. |
| `checkpoint.pt` | Model and optimizer states, model configuration, seed, elapsed training time, processed tokens, and resolved run settings. |

`evaluate` loads the checkpoint and scores fixed, seeded windows from the separate test split. Validation and test cross-entropy and perplexity are comparable **when the tokenizer is the same**. `generate` loads the checkpoint and uses `generation.prompt`, `max_new_tokens`, `temperature`, and `top_k` from the YAML; `--prompt` overrides only the text for that invocation. Both commands accept `--checkpoint` to use a specific saved checkpoint.

## Checks

```bash
python -m unittest discover -s tests -v
```

The tests cover next-token targets, causal masking, tied weights, and overfitting a tiny repeated batch. A short smoke run checks the full file workflow, but its generated text is not a meaningful quality benchmark.

## Reproducing the report (DSA4212 submission)

Repository layout required by the course guidelines:

- `readme.md` (this file): setup, data, experiments, reproduction.
- `authors.md`: name and student number of every group member.
- `report/`: LaTeX source of the submitted report (`report/final_report.tex`, which `\input`s the `report2_*.tex` modules and uses `report2_references.bib`) and its figures.
- `analysis/`: tables and figures from `analyze.py`/`eval_hf.py`; follow-up scores are in `analysis/followups/` (seed, matched-target, and learning-rate subfolders).
- `results/<run>/`: per-run `run_config.json`, `metrics.csv`, `test_results.json`, `generations.md`.

Compute budget: every final run is one hour (3,600 s) of in-loop training on one
NVIDIA H100 NVL MIG 3g.47gb slice, bfloat16, from random weights. Twenty runs:
five model sizes (`small` 16M, `medium` 30M, `large` 51M, `xl` 124M, `xxl` 253M
parameters) times four training-data fractions (25/50/75/100 % of the TinyStories
train split). Three follow-up experiments extend the screen: XL and XXL retrained across seeds
42/43/44, a matched-target XL/XXL pair (35,000 updates, 143M tokens each), and a
reduced learning-rate XXL run. The checkpoints and logs of all 29 runs (20 screen
plus 9 follow-ups) are on the Hugging Face Hub at `dsa4212-tinystories/tinystories-nanogpt`.

To regenerate every number and figure in the report without retraining:

```bash
python eval_hf.py                        # downloads each checkpoint, test loss / ppl / bits-per-byte, story samples
python analyze.py --runs-dir results     # summary.csv, summary_table.md, figures into analysis/
python score_checkpoints.py --checkpoints <paths to checkpoint.pt> --data-dir data/tokens-full --device cuda --precision float32 --output-dir analysis/followups/<name>   # full commands per experiment in FOLLOWUP_COMMANDS.md
python analyze_followups.py              # builds analysis/followups/ tables and figures
```

See `FOLLOWUP_COMMANDS.md` and `scripts/` (SLURM) for the exact follow-up training/scoring commands.

To retrain one run from scratch (needs a CUDA GPU; one hour per run):

```bash
python run.py prepare --config configs/experiments/<name>.yaml
python run.py train   --config configs/experiments/<name>.yaml
```

The experiment configs are produced by `python create_experiments.py` from
`configs/default.yaml`; the training recipe (AdamW, lr 3e-4, weight decay 0.1,
betas 0.9/0.95, clip 1.0, 2 % time-based warmup then cosine to 10 % of peak,
context 256, dropout 0.1) is the same for all runs except where a follow-up config changes it (learning rate, step-based schedule).
