# Token exposure and epoch-duration estimates

This calculation reads the final metrics and saved configurations of 20 runs under `results/`. All 262,432 logged rows were checked against the optimizer-step token formula. Exact values and source paths are in [token_exposure_and_epoch_estimates.csv](token_exposure_and_epoch_estimates.csv).

## What the logs establish

The full prepared training stream contains **473,992,236 tokens**. Five runs have processed fewer targets than their own prepared training stream contains: `xl_data75`, `xl_data100`, `xxl_data50`, `xxl_data75`, `xxl_data100`.

| Model | Parameters | Prepared tokens | Processed targets | Exposure | Time for one equivalent |
| --- | --- | --- | --- | --- | --- |
| small | 16,079,360 | 473,992,236 | 984,326,144 | 2.077x | 28m 54s |
| medium | 30,018,816 | 473,992,236 | 734,642,176 | 1.550x | 38m 43s |
| large | 51,037,184 | 473,992,236 | 562,429,952 | 1.187x | 50m 34s |
| xl | 123,747,840 | 473,992,236 | 279,830,528 | 0.590x | 1h 41m 38s |
| xxl | 253,085,696 | 473,992,236 | 143,654,912 | 0.303x | 3h 17m 59s |

The smaller models processed more than one full-data-equivalent, but this does not verify unique coverage. We count each training target once; input tokens and shifted targets are not added together. Validation targets are excluded. The final metrics row repeats the last training step and is not another optimizer update.

## Exposure relative to each run's own cache

An entry of 1.000x means that processed training targets equal the number of tokens in that run's prepared cache. It does not mean a sequential epoch completed.

| Model | 25% stories | 50% stories | 75% stories | 100% stories |
| --- | --- | --- | --- | --- |
| small | 8.206x | 4.159x | 2.647x | 2.077x |
| medium | 6.071x | 3.111x | 2.078x | 1.550x |
| large | 4.754x | 2.354x | 1.581x | 1.187x |
| xl | 2.392x | 1.156x | 0.757x | 0.590x |
| xxl | 1.209x | 0.625x | 0.397x | 0.303x |

## Estimated total training-loop time for one cache-equivalent

For each run:

`observed throughput = processed training targets / recorded training-loop seconds`

`estimated seconds for one equivalent = prepared training-cache tokens / observed throughput`

| Model | 25% stories | 50% stories | 75% stories | 100% stories |
| --- | --- | --- | --- | --- |
| small | 7m 19s | 14m 26s | 22m 40s | 28m 54s |
| medium | 9m 54s | 19m 18s | 28m 53s | 38m 43s |
| large | 12m 38s | 25m 30s | 37m 57s | 50m 34s |
| xl | 25m 06s | 51m 55s | 1h 19m 15s | 1h 41m 38s |
| xxl | 49m 39s | 1h 35m 58s | 2h 31m 06s | 3h 17m 59s |

These are estimates from the recorded NVIDIA H100 NVL MIG 3g.47gb, bfloat16 runs. They include periodic validation and logging overhead through the observed average throughput. They exclude setup, final evaluation, and saving, and assume comparable sustained throughput. They are not guaranteed exact runtimes for a different sampler, device load, batch configuration, or training schedule.

For XXL with 100% data, the calculation is:

`473,992,236 / 39,903.07 = 11,878.59 seconds`

That is about **3h 17m 59s total**, or **2h 17m 59s beyond the completed hour** at the same rate. The current training interface does not support resuming a checkpoint, so this additional duration is an arithmetic difference, not a supported continuation command.

A shared **four-hour budget (14,400 seconds)** would provide a margin for every tested model to process at least one full-data-equivalent at the observed rates. It would give faster models more than one equivalent and would still not guarantee unique coverage. This is a proposed follow-up budget; it does not change the confirmed one-hour default.

## One equivalent is not a true epoch

`random_batch` in `train.py` chooses random window starts independently with replacement. Some intervals can be sampled repeatedly while others are never selected. Therefore increasing only the time limit cannot guarantee that every prepared position is visited.

To guarantee an epoch, the sampler needs to visit a defined collection of training windows without replacement before beginning the next pass. For example, construct non-overlapping windows, shuffle their order, and process each once. Handle the trailing partial window explicitly, or document any dropped positions. Standard next-token prediction also needs to distinguish the stream's first token, which has no preceding input token.

That sampler and an epoch/token stopping criterion would be an implementation change. They should be verified and rebenchmarked before calling a run an exact one-epoch experiment. The times above are useful planning estimates, not measured epoch durations under that new implementation.

## What this means for the conclusions

The fixed-time experiment remains valid for the stated question: which tested configuration gives the lowest held-out loss after one hour on the recorded hardware with the shared recipe? Different token exposure is part of that trade-off, not a requirement that invalidates the comparison.

The broad statement that larger networks always work better is unsupported. The observed validation ranking improves up to XL, and XXL is slightly worse. The defensible claim is that XL achieved the best observed result among the tested configurations at this budget. One seed, one shared optimizer recipe, and small differences limit the strength of the ranking.

A claim about performance after equal training exposure requires a different comparison. Use the same prepared data and token/epoch budget for all candidates, let runtime differ, and specify a common progress-based learning-rate policy. Equal time, equal data exposure, and equal theoretical FLOPs are different controls.

The quadratic curve in `analyze_compute_optimal.py` is a descriptive fit to a small set of observed best-per-family scores; it does not establish a universal compute-optimal parameter count. The current runs share wall-clock time, not necessarily equal estimated FLOPs, and the fitted points can use different data subsets.

## Suggested follow-up

1. Keep the one-hour results and describe them explicitly as a fixed-time comparison.
2. To probe whether XXL is time-limited, compare XL and XXL on the same full cache with a longer shared budget, such as four hours, and record the complete learning curves.
3. To guarantee data coverage or compare equal exposure, implement and test sampling without replacement and an explicit token/epoch stop before launching those experiments.
4. A fresh longer run needs a fresh output directory. The time-based learning-rate schedule also stretches with the new budget, so a four-hour run is not identical to appending three hours to the existing one-hour run.
