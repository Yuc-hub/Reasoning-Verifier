# Bradley–Terry loss ablation

## Research question

With exactly the same 1,044-dimensional candidate features and the same
8,271 within-question correct/wrong pairs as the existing pairwise ridge
model, does a Bradley–Terry logistic preference loss improve reranking?

## Method

For a correct candidate `j` and wrong candidate `k`, define
`d = x_j - x_k`, where `x` concatenates the existing 20 text features and
1,024 fixed embedding dimensions after training-only standardization. The
model assigns a scalar reward `r_j = x_j @ w` and preference probability
`P(j > k) = sigmoid(d @ w)`. It minimizes

`sum_pairs log(1 + exp(-d @ w)) + (lambda / 2) * ||w||²`.

This is a genuine Bradley–Terry pairwise log loss. The prior pairwise model
used a **squared margin loss**, not Bradley–Terry and not hinge loss.
Training uses full-batch Newton steps with a monotonic backtracking line
search. The trained weights, feature means and scales are stored as float32
arrays. Inference uses the same `predict()` implementation as the prior
pairwise model, exposed through `code/methods/bradley_terry.py`.

## Parameters and fair-comparison controls

| Item | Value |
|---|---|
| Training questions | 4,755 |
| Informative correct/wrong pairs | 8,271, identical to pairwise ridge |
| Candidate features | 20 text + 1,024 embedding = 1,044, identical to pairwise ridge |
| Feature scaling | Training mean and standard deviation, identical to pairwise ridge |
| Objective | Summed Bradley–Terry log loss + `lambda/2 * ||w||²` |
| L2 coefficient `lambda` | 25.0 |
| Why 25.0 | At zero margin, logistic curvature is 0.25; `25 / 0.25 = 100`, matching the prior ridge model's relative `lambda = 100` curvature scale at initialization |
| Newton iteration cap | 20; converged in 10 iterations |
| Stop threshold | Maximum absolute accepted weight step below `1e-7` |
| Prediction | Highest of four scalar rewards; lowest index wins ties |
| Random seed | Not applicable; deterministic training |
| Checkpoint | `code/runs/bradley_terry/model.npz` |
| Checkpoint size | About 13 KiB |
| Checkpoint SHA-256 | `3f81e0e51d986116d8a2420e9918ce42b0418690b27cc9fad33f43337cb00fdc` |

The stored training means and scales were checked with `numpy.array_equal`
against the prior pairwise ridge checkpoint; both comparisons returned true.
No validation references were used to fit weights or select the regularizer.

## Validation result

Environment: Miniconda `mp1`, Python 3.12.14, NumPy 2.5.3.
Official validation scorer: **739 / 798 = 92.60651629072682%**.
The existing pairwise ridge model scored 734 / 798 on the same split, a gain
of five correct questions for Bradley–Terry. The prior score-fusion method
scored 738 / 798. The prediction runner reported 0.201 seconds, excluding
some process startup; this is not the complete reference-machine timing.

```bash
cd code
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  conda run -n mp1 python train_bradley_terry.py \
  --output runs/bradley_terry/model.npz
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  conda run -n mp1 python predict.py --method methods/bradley_terry.py \
  --checkpoint runs/bradley_terry/model.npz --split validation \
  --output runs/bradley_terry/validation-predictions.json
conda run -n mp1 python score_predictions.py --split validation \
  --predictions runs/bradley_terry/validation-predictions.json \
  --output runs/bradley_terry/validation-score.json
```

## Interpretation and limits

The controlled comparison supports a validation benefit from changing the
pairwise loss. It does not establish a general improvement on unseen banks;
five questions are a small difference. This experiment was developed after
earlier methods' test results were seen, so its test score is labeled
exploratory and must not drive further tuning or retrospective method selection.
The model uses no external data, pretrained weights, or online API calls.
The checkpoint is Git-ignored under `runs/` and must be published separately
if this method is submitted for reproduction.

## Revalidation and exploratory test evaluation

The unchanged checkpoint hash above was verified before running both splits.
The official scorer reconfirmed **739 / 798 = 92.60651629072682%** on
validation and reported **382 / 467 = 81.79871520342613%** on test.
Outputs are saved under `code/runs/bradley_terry/` as
`validation-predictions.json`, `validation-score.json`,
`test-predictions.json`, and `test-score.json`.

```bash
cd code
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  conda run -n mp1 python predict.py --method methods/bradley_terry.py \
  --checkpoint runs/bradley_terry/model.npz --split test \
  --output runs/bradley_terry/test-predictions.json
conda run -n mp1 python score_predictions.py --split test \
  --predictions runs/bradley_terry/test-predictions.json \
  --output runs/bradley_terry/test-score.json
```

Local fresh-process timing with the `mp1` Python executable was 0.25 seconds
for validation prediction plus 0.06 seconds for validation scoring, and 0.18
seconds for test prediction plus 0.06 seconds for test scoring. Hardware is
Apple Silicon `arm64`; the course reference Xeon determines eligibility.
No model, hyperparameter, or checkpoint was changed after this test run.
