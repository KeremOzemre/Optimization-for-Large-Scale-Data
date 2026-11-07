# Training a small GPT on TinyStories under a one-hour budget

*DSA4212 — Optimisation for Large-Scale Data-Driven Inference*
*Authors: Steffany, Kerem, Tonio, [name]*

> Draft skeleton. Each `> TODO` is a prompt for what to write; delete it once the
> section is filled. Figures referenced below are produced by `analyze.py` into
> `analysis/`.

---

## 1. Compute budget declaration

> TODO: State the budget plainly and confirm the rules. One short paragraph.

- **Budget:** one hour (3,600 s) of in-loop training wall-clock **per model**, on
  the same hardware for every run. Data preparation, final validation, and
  checkpointing are outside the timed loop.
- **Hardware:** [GPU model / cluster node used on NUS — fill in from
  `run_config.json → runtime.hardware`].
- **From scratch:** every model starts from random weights (nanoGPT init). No
  pretrained weights, embeddings, or tokenizer fine-tuning.

## 2. Research question

> TODO: One or two sentences.

**Which model size and training-data fraction gives the best language-modeling
performance under a fixed one-hour budget on identical hardware?** We hold the
tokenizer, optimizer, schedule, context length, and batch settings fixed and vary
only (a) model size and (b) the fraction of the training split used.

## 3. Experimental setup

> TODO: Fill the concrete numbers from configs/ and run_config.json.

- **Data:** `roneneldan/TinyStories`. Official train split for training; official
  validation split divided 80/20 into validation/test at the story level (seed 42).
  Test used only for the final comparison in §5.3.
- **Tokenizer:** GPT-2 (`tiktoken`), vocab 50,257. End-of-text token appended per
  story. Because the tokenizer is fixed, per-token loss and perplexity are directly
  comparable across all our runs.
- **Model:** nanoGPT GPT, tied embeddings, causal attention, dropout 0.
  Three sizes:

  | Size   | Layers | Width | Heads | Params (M) |
  |--------|-------:|------:|------:|-----------:|
  | small  | 4      | 256   | 4     | [fill]     |
  | medium | 6      | 384   | 6     | [fill]     |
  | large  | 8      | 512   | 8     | [fill]     |

- **Data fractions:** 25 / 50 / 75 / 100 % of the training split → a 3×4 grid of
  12 runs.
- **Optimizer / schedule:** AdamW (lr 3e-4, wd 0.1, betas 0.9/0.95, clip 1.0),
  time-based 2% warmup then cosine decay to 10% of peak. Context 256, microbatch
  16, accumulation 1.
- **Metric:** held-out cross-entropy (and perplexity = exp(loss)) on fixed seeded
  validation windows every ~300 s; a final full validation pass at the end.

## 4. Methodology / reproducibility

> TODO: Describe how runs were launched and collected, so a reader could repeat it.

1. `python run.py prepare --config configs/default.yaml` builds the full token cache.
2. `python create_experiments.py` generates the 12 configs in `configs/experiments/`.
3. Each config trained with `python run.py train --config configs/experiments/<name>.yaml`
   on the same node; outputs land in `runs/<name>/`.
4. Model selected by **validation** loss; only then evaluated on **test**.
5. `python analyze.py` aggregates every `runs/*/` into `analysis/` (tables + figures).

## 5. Results

> TODO: Paste the generated table and reference each figure with one sentence of
> interpretation. Do not just show plots — say what they mean.

### 5.1 Summary table

Paste `analysis/summary_table.md` here (sorted by final validation loss).

### 5.2 Model size vs. data vs. budget

- **Validation loss over time** (`analysis/val_loss_curves.png`): [which runs
  converge fastest? does any plateau or still look like it's improving at 1 h?]
- **Loss vs. model size** (`analysis/val_loss_vs_params.png`): [is bigger always
  better within the hour, or is there a sweet spot?]
- **Loss vs. data fraction** (`analysis/val_loss_vs_data.png`): [does more data
  help, or does the budget bottleneck make it irrelevant?]
- **Loss vs. tokens processed** (`analysis/val_loss_vs_tokens.png`) and
  **throughput** (`analysis/throughput_vs_params.png`): the key tension — larger
  models see **fewer** tokens in the same hour. [quantify the trade-off].

### 5.3 Final test result

After selecting the best configuration on validation, its test loss/perplexity is
[fill from the evaluate step]. We evaluate only this selected model on test.

## 6. Discussion

> TODO: Interpret. Why did the winner win? Was it compute-bound or data-bound?
> What would change with a bigger budget? Note limitations: windows sampled with
> replacement (no guaranteed epoch), fixed recipe may not be optimal for every size.

## 7. Conclusion

> TODO: One paragraph answering the research question with the winning config and
> the single most important trade-off you found.

## Appendix

- Generated story samples per model (`python run.py generate ...`) — qualitative
  check that supplements, but does not replace, the held-out metrics.
- Exact configs: `configs/experiments/`. Per-run settings: `runs/*/run_config.json`.
