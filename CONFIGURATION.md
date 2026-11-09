# Configuration reference

This reference explains every setting accepted by `CONFIG_KEYS` in `run.py`. Defaults come from [configs/default.yaml](configs/default.yaml); [configs/smoke.yaml](configs/smoke.yaml) uses smaller settings for checking the pipeline.

Copy a configuration, edit its values, and pass it to the runner:

```bash
python run.py train --config configs/my-experiment.yaml
```

Every section and original key is required. The six newer keys (`preparation_seed`, `max_steps`, `progress`, evaluation `microbatch_size`/`seed`, and generation `seed`) have compatibility defaults described below; unknown keys are rejected. Paths are relative to the project root. `null` means no optional limit. The follow-up YAMLs specify all new settings explicitly; see [FOLLOWUP_COMMANDS.md](FOLLOWUP_COMMANDS.md).

## Experiment

| Key | Default | Allowed values | Meaning and effect |
| --- | --- | --- | --- |
| `experiment.name` | `baseline` | Nonempty text, such as `width-128` or `depth-6`. | Gives the experiment a readable label that is saved in its run configuration. It does not automatically choose or rename the output directory, so set `experiment.output_dir` separately. |
| `experiment.seed` | `42` | A nonnegative integer. | Controls random model initialization, dropout, and training-window sampling. With explicit preparation/evaluation/generation seeds, changing it leaves the cache and held-out targets fixed. Identical seeds do not guarantee identical results across hardware or software. |
| `experiment.output_dir` | `out/baseline` | A nonempty path to a directory that does not yet exist when training starts. | Selects where training writes `run_config.json`, `metrics.csv`, and `checkpoint.pt`. Training refuses an existing directory, including an empty one, to protect earlier runs. Evaluation and generation look for `checkpoint.pt` here unless you supply `--checkpoint`. |

## Data

| Key | Default | Allowed values | Meaning and effect |
| --- | --- | --- | --- |
| `data.dataset` | `roneneldan/TinyStories` | Only `roneneldan/TinyStories` is currently supported. | Identifies the Hugging Face dataset loaded during preparation. The configuration validator rejects other datasets because this pipeline currently supports TinyStories. |
| `data.preparation_seed` | `42`; omitted: `experiment.seed` | Nonnegative integer. | Controls the held-out story split and training-story shuffle before selecting a subset. Keep 42 and reuse the same cache across training-seed trials. Metadata retains the existing `seed` field, so compatible caches do not need retokenization. |
| `data.train_split` | `train` | An existing dataset split name, different from `data.heldout_split`; TinyStories provides `train` and `validation`. | Chooses the source stories that become `train.bin`. Keep it at `train` for the confirmed project setup so training uses the official training split. Changing it requires a compatible new or rebuilt token cache. |
| `data.heldout_split` | `validation` | An existing dataset split name, different from `data.train_split`. | Chooses the source stories that preparation divides into validation and test sets. Keep it at `validation` for the confirmed setup, which reserves the official validation stories for held-out evaluation. The division happens at the story level before tokenization. |
| `data.test_ratio` | `0.2` | A number strictly between `0` and `1`. | Sets the fraction of `data.heldout_split` assigned to the test set; the rest becomes validation. For example, `0.2` gives an 80/20 validation/test division of the held-out source split. It does not take 20% of the training data, and changing it requires a different or rebuilt token cache. |
| `data.train_stories` | `null` | `null` or a positive integer, such as `2000`. | Limits preparation to the first N stories of the selected training split, using all available stories if N exceeds its size. `null` uses the entire split. This changes the amount of cached training data, while the training loop still samples windows with replacement from that cache. |
| `data.validation_stories` | `null` | `null` or a positive integer, such as `200`. | Limits the validation stories retained after the held-out source split has been divided into validation and test. It takes the first N stories of that resulting validation set, capped at its available size. This limits the validation cache; `evaluation.batches` separately controls how much of it is sampled during evaluation. |
| `data.test_stories` | `null` | `null` or a positive integer, such as `200`. | Limits the test stories retained after the held-out source split has been divided, capped at the available number. `null` retains all test stories. Training does not read these tokens; the `evaluate` action uses them after loading a checkpoint. |
| `data.cache_dir` | `data/tokens-full` | A nonempty directory path, such as `data/tokens-small`. | Selects where preparation saves `train.bin`, `val.bin`, `test.bin`, and `meta.json`, and where training reads them. Multiple model configurations can share a cache when the preparation settings match. An incompatible completed cache is rejected unless you deliberately rebuild it with `--force`; evaluation uses the cache path recorded in the checkpoint. |

## Tokenizer

| Key | Default | Allowed values | Meaning and effect |
| --- | --- | --- | --- |
| `tokenizer.name` | `gpt2` | Only `gpt2` is currently supported. | Selects the existing GPT-2 tokenizer provided by `tiktoken`; it does not load GPT-2 model weights. Preparation converts each story to token IDs and appends the end-of-text token, and the model vocabulary size is derived automatically as 50,257. Adding other tokenizers would require code changes, and their token-level losses and perplexities would not be directly comparable. |

## Model

These settings build a new randomly initialized model during `train`. Evaluation and generation reconstruct the architecture saved in the checkpoint, so editing this section does not resize an already trained model.

| Key | Default | Allowed values | Meaning and effect |
| --- | --- | --- | --- |
| `model.layers` | `4` | A positive integer, such as `2`, `4`, or `8`. | Sets the number of Transformer blocks stacked in the model. More layers add parameters and computation, which can reduce the number of training tokens processed within a fixed time budget. |
| `model.width` | `256` | A positive integer divisible by `model.heads`. | Sets the number of values in each token's embedding and hidden representation. Increasing width enlarges the embeddings, attention projections, and feed-forward layers, generally increasing parameter count, memory use, and computation. For example, width `256` works with `4` heads because each head receives 64 values. |
| `model.heads` | `4` | A positive integer that divides `model.width` exactly. | Splits each attention operation into this many heads, each with width `model.width / model.heads`. At a fixed model width, changing the head count does not change the sizes of the main projection matrices and generally leaves parameter count unchanged. It changes how attention is arranged and can affect performance and execution speed. |
| `model.context_length` | `256` | A positive integer, smaller than the token count in each split being read. | Sets the number of input tokens in a training or evaluation sequence and the model's maximum attention window. Longer contexts require more memory and attention computation and also increase the learned position-embedding table. During generation, a longer prompt is accepted, but each prediction uses only the most recent context-length tokens. |
| `model.dropout` | `0.0` | A number from `0` inclusive to `1` exclusive. | Sets the probability of dropping selected activations during training, which introduces noise that can help reduce overfitting. `0.0` disables dropout. Evaluation and generation disable dropout automatically through evaluation mode. |

## Training

| Key | Default | Allowed values | Meaning and effect |
| --- | --- | --- | --- |
| `training.device` | `auto` | `auto`, `cuda`, `mps`, or `cpu`. | Chooses the device used for training, evaluation, and generation. `auto` tries CUDA first, then Apple MPS, then CPU. Selecting an unavailable device explicitly produces an error, and the actual device and hardware are recorded for training runs. |
| `training.precision` | `auto` | `auto`, `float32`, or `bfloat16`. | Controls the computation precision used during training and evaluation. `auto` enables bfloat16 autocasting on supported CUDA devices and otherwise uses float32; explicitly requesting `bfloat16` requires supported CUDA hardware. Generation currently uses the loaded model in float32 and does not apply this setting. |
| `training.time_limit_seconds` | `3600` | A positive number of seconds; `300` gives a five-minute training budget. | Limits elapsed wall-clock time inside the training loop, including periodic validation and logging. The optimizer step already underway finishes before stopping, so the loop can exceed the requested time slightly. Dataset preparation, model setup, final validation, and checkpoint saving are outside the budget, so the total command takes longer. |
| `training.microbatch_size` | `16` | A positive integer. | Sets how many token sequences are processed together in one forward/backward pass. Larger values usually need more memory and may improve throughput. Evaluation has its own microbatch setting; it only inherits this value in legacy YAMLs that omit the newer field. |
| `training.max_steps` | `null` | `null` or a positive integer. | Stops after this many completed optimizer updates, or earlier if the time cap is reached. Combined with a fixed effective batch/context, this supplies an exact processed-target budget. Inspect the saved stop reason before accepting an equal-token comparison; this does not guarantee an epoch or enable resume. |
| `training.gradient_accumulation_steps` | `1` | A positive integer. | Sets how many microbatches contribute gradients before one optimizer update, with their losses averaged across the accumulation steps. The effective number of sequences per update is `microbatch_size × gradient_accumulation_steps`. Increasing accumulation can give a larger effective batch without holding all its activations in memory at once, but it takes more computation per optimizer step. |
| `training.log_interval_steps` | `10` | A positive integer. | Controls how often normal training rows are printed and appended to `metrics.csv`, measured in optimizer steps. Validation events and the final result also create rows regardless of this interval. Changing it does not change update frequency, although more frequent logging adds overhead within the time budget. |

## Optimizer

The optimizer is always PyTorch AdamW in the current pipeline.

| Key | Default | Allowed values | Meaning and effect |
| --- | --- | --- | --- |
| `optimizer.learning_rate` | `0.0003` | A positive number. | Sets the peak learning rate reached by the time-based schedule, controlling the scale of optimizer updates. A larger value can speed up learning but can also make training unstable. The learning rate used on each step is saved in `metrics.csv`. |
| `optimizer.weight_decay` | `0.1` | A number greater than or equal to `0`. | Controls AdamW's tendency to shrink weights during optimization, which can help reduce overfitting. The nanoGPT optimizer setup applies decay to matrix parameters such as embeddings and linear weights, while excluding one-dimensional parameters such as normalization weights. `0` disables weight decay. |
| `optimizer.betas` | `[0.9, 0.95]` | A list of two numbers, each strictly between `0` and `1`. | Sets AdamW's two moving-average factors: the first tracks gradients, and the second tracks squared gradients. Values closer to one retain information from more previous updates, while smaller values react more quickly to recent gradients. The first and second entries must remain in that order. |
| `optimizer.gradient_clip` | `1.0` | A number greater than or equal to `0`. | Limits the combined gradient norm before each optimizer update, after all accumulated microbatches have completed. If the norm exceeds the threshold, the gradients are scaled down together, which can help control unusually large updates. `0` disables clipping in this implementation. |

## Schedule

The schedule uses time or completed optimizer steps, as selected by `progress`.

| Key | Default | Allowed values | Meaning and effect |
| --- | --- | --- | --- |
| `schedule.progress` | `time` | `time` or `steps`. | Chooses the progress variable for warmup and cosine decay. `time` divides elapsed seconds by the time budget; `steps` divides completed updates by `training.max_steps`, which must then be set. Use `steps` to give equal-token candidates the same learning rate at the same exposure. |
| `schedule.warmup_fraction` | `0.02` | A number from `0` inclusive to `1` exclusive. | Sets the fraction of the time budget spent increasing the learning rate from its minimum value to its peak. With a 3,600-second budget, `0.02` gives 72 seconds of warmup, followed by cosine decay for the remaining time. `0` skips warmup and starts the schedule at the peak learning rate. |
| `schedule.min_lr_ratio` | `0.1` | A number greater than `0` and less than or equal to `1`. | Sets the minimum learning rate as a fraction of `optimizer.learning_rate`, both at the start of warmup and at the end of cosine decay. With peak `0.0003` and ratio `0.1`, the minimum is `0.00003`. Setting it to `1` makes the learning rate constant throughout the run. |

## Evaluation

| Key | Default | Allowed values | Meaning and effect |
| --- | --- | --- | --- |
| `evaluation.interval_seconds` | `300` | A positive number of seconds. | Sets how often training checks validation loss, using elapsed training-loop time and checking after optimizer steps. `60` requests validation about once a minute, while a final validation always runs after training stops. Periodic validation consumes the training budget, so evaluating more often can reduce the tokens available for training. |
| `evaluation.batches` | `16` | A positive integer. | Sets the number of fixed batches scored per validation or sampled test evaluation. Target count is `batches × evaluation.microbatch_size × context_length`. Windows may overlap; increasing this count costs evaluation time without guaranteeing full coverage. |
| `evaluation.microbatch_size` | `16`; omitted: training microbatch | Positive integer. | Controls inference batching independently of training. Keep this and `batches` the same across model-size comparisons to score the same number of windows. The follow-ups use 4 × 64, giving 65,536 targets at context 256. |
| `evaluation.seed` | `43`; omitted: training seed + 1 | Nonnegative integer. | Selects the fixed validation offsets. Sampled test evaluation uses this seed plus one, independent of checkpoint training seed when explicitly configured. `score_checkpoints.py` has its own recorded full/sampled protocol and defaults to validation. |

## Generation

These settings affect the `generate` action after loading a checkpoint. They do not change the trained weights or training losses.

| Key | Default | Allowed values | Meaning and effect |
| --- | --- | --- | --- |
| `generation.prompt` | `Once upon a time` | Nonempty text. | Supplies the story beginning, included in the printed result. Override it for one invocation with `--prompt "Your story beginning"`. The tokenizer comes from the checkpoint, while the sampling seed comes from `generation.seed`. |
| `generation.max_new_tokens` | `100` | A nonnegative integer. | Caps additional tokens after the prompt, stopping sooner when EOT is sampled. Tokens are pieces of text, so this is not a word count. `0` prints only the prompt. |
| `generation.seed` | `101`; omitted: training seed | Nonnegative integer. | Controls multinomial token sampling independently of model training. Change it to produce another reproducible sample under the same prompt and sampling settings. `sample_stories.py` accepts an explicit seed list and records termination details for multiple models. |
| `generation.temperature` | `0.8` | A positive number; `0` is not supported. | Scales the next-token scores before sampling. Values below one make the distribution more concentrated on likely tokens, while values above one spread probability more broadly. It changes sampling behavior without changing the checkpoint's weights. |
| `generation.top_k` | `40` | A positive integer. | Restricts sampling to tokens with the highest next-token scores at each generation step. Smaller values restrict the choices more strongly, while larger values allow more alternatives; the code caps the requested count at the vocabulary size. A value at least as large as the vocabulary considers all tokens, while `1` normally chooses the highest-scoring token. |

## Useful relationships

- **Tokens per optimizer update:** `training.microbatch_size × training.gradient_accumulation_steps × model.context_length`.
- **Tokens per validation evaluation:** `evaluation.batches × evaluation.microbatch_size × model.context_length`.
- **Equal-token run target:** `training.max_steps × tokens per optimizer update`; the time cap can stop early.
- **Width of one attention head:** `model.width / model.heads`.
- **Warmup progress:** time limit or maximum steps, multiplied by `schedule.warmup_fraction`.
- **Minimum learning rate:** `optimizer.learning_rate × schedule.min_lr_ratio`.

The focused follow-up pair uses `max_steps: 35000`, 4,096 targets/update,
`schedule.progress: steps`, and a 7,200-second safety cap: 143,360,000 targets per
model. Accept it only when both saved summaries report `stop_reason: max_steps`
and the exact target count. See [FOLLOWUP_COMMANDS.md](FOLLOWUP_COMMANDS.md) for
the main one-hour study and optional equal-token commands.

For the first model-size comparisons, keep the data, tokenizer, training budget, hardware, optimizer, schedule, and evaluation settings fixed while changing layers or width. Give every run a fresh output directory and compare validation loss alongside the actual parameter count and processed training tokens recorded in the output files. Use the test set after selecting a model from validation results.
