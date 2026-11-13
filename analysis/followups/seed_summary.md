# Follow-up validation results

Source: `analysis/followups/seed_validation/scores.csv`.

All scores use the common protocol saved beside the source CSV. Lower CE is better. Standard deviations describe training-seed variation, while paired story intervals in `paired_comparisons.json` describe evaluation-sample variation conditional on each checkpoint.

| Budget | Model | Seeds | Mean CE | Sample SD | Min CE | Max CE |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 1h | xl | 42, 43, 44 | 1.494657 | 0.003793 | 1.490381 | 1.497618 |
| 1h | xxl | 42, 43, 44 | 1.524387 | 0.009792 | 1.513084 | 1.530280 |

Paired training-seed differences (XXL CE minus XL CE; positive favours XL):

- 1h: seed 42: +0.039417; seed 43: +0.034309; seed 44: +0.015466. Mean difference: +0.029731.

Three seeds are an initial variability check. Do not equate a small mean gap with a universal ranking. Longer-budget runs use budget-specific schedules; equal-token runs still sample with replacement and do not guarantee an epoch.
