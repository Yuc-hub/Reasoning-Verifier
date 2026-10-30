# Candidate pairwise comparison model

## Idea and implementation

For each training question, label all four candidates by comparing their
parsed final numeric answers with the training reference. Construct every
within-question pair where one candidate is correct and the other is wrong.
For each such pair, subtract the wrong candidate's feature vector from the
correct candidate's vector. Learn weights that make this difference score
positive, with target margin 1:

`min_w sum_pairs ((x_correct - x_wrong) @ w - 1)^2 + 100 * ||w||²`.

Each candidate vector concatenates the 20 text features used by the text
feature model with its supplied 1024-dimensional embedding. Features are
standardized using means and standard deviations computed from **training
inputs only**. Questions where every candidate is correct have no informative
correct/wrong pair and do not contribute to the fit. If a question has
multiple correct candidates, each contributes pairs against every wrong one.

At inference, `code/methods/pairwise_comparison.py` scores the four candidates
using their standardized features and selects the highest score. References
are not opened during inference. Training is in
`code/train_pairwise_comparison.py`.

## Parameters

| Item | Value |
|---|---|
| Training questions | 4,755 |
| Training correct/wrong pairs | 8,271 |
| Features per candidate | 20 text features + 1,024 fixed embedding dimensions = 1,044 |
| Fit precision | float64 |
| Objective | Pairwise squared margin error with L2 regularization |
| Target margin | 1 |
| Ridge coefficient | 100.0 |
| Scaling | Training mean and standard deviation per feature; near-zero scale set to 1 |
| Prediction | Highest of four linear scores; lowest index wins ties |
| Random seed | Not applicable; deterministic closed-form fitting |
| Checkpoint | `code/runs/pairwise_comparison/model.npz` |
| Checkpoint size | About 13 KiB file; three 1,044-element float32 arrays, about 12.2 KiB of learned arrays |
| Checkpoint SHA-256 | `040975eb2c7ab71d59d303b1ab18a7e1e752a905c1d29aae6b7279fcb29a5457` |

## Validation result

Environment: Miniconda `mp1`, Python 3.12.14, NumPy 2.5.3.
Official validation scorer: **734 / 798 = 91.97994987468671%**.
Previous results on the same split: answer agreement 688 / 798, text features
719 / 798, embedding linear 712 / 798. The prediction runner reported 0.207
seconds, excluding some process startup; this is not the required complete
evaluation time.

```bash
cd code
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  conda run -n mp1 python train_pairwise_comparison.py \
  --output runs/pairwise_comparison/model.npz
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  conda run -n mp1 python predict.py --method methods/pairwise_comparison.py \
  --checkpoint runs/pairwise_comparison/model.npz --split validation \
  --output runs/pairwise_comparison/validation-predictions.json
conda run -n mp1 python score_predictions.py --split validation \
  --predictions runs/pairwise_comparison/validation-predictions.json \
  --output runs/pairwise_comparison/validation-score.json
```

## Interpretation and limitation

This experiment changes both the feature set and the fitting objective
relative to the earlier methods. Its higher validation accuracy does not by
itself isolate the benefit of pairwise training. A fair ablation would compare
pointwise and pairwise training with identical 1,044-dimensional features.
The test split has not been used for method selection. `runs/` is Git-ignored;
a final submission using this model must publish its matching checkpoint.

## Test evaluation

Using the unchanged model checkpoint, the official test scorer reported
**378 / 467 = 80.94218415417559%**. Files:
`code/runs/pairwise_comparison/test-predictions.json` and
`code/runs/pairwise_comparison/test-score.json`. Local fresh-process timings
were 0.38 seconds for prediction and 0.08 seconds for scoring (Apple Silicon
`arm64`, Miniconda `mp1`); the reference Xeon determines eligibility.

```bash
cd code
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  conda run -n mp1 python predict.py --method methods/pairwise_comparison.py \
  --checkpoint runs/pairwise_comparison/model.npz --split test \
  --output runs/pairwise_comparison/test-predictions.json
conda run -n mp1 python score_predictions.py --split test \
  --predictions runs/pairwise_comparison/test-predictions.json \
  --output runs/pairwise_comparison/test-score.json
```

This exceeds the previously frozen fusion model's test score by one question,
while the fusion model was higher on validation. The test comparison is
descriptive; test results must not be used to tune or select a method.
