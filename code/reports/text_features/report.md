# Text feature candidate reranker

## Idea and implementation

Each candidate receives a 20-dimensional feature vector from its text and
question. The features cover final-answer agreement and numeric properties,
text length and structure, question overlap, and the candidate's difference
from the other three candidates. See `FEATURE_NAMES` in
`code/methods/text_features.py` for the exact ordered list and formulas.

`code/train_text_features.py` loads **training inputs and training references**,
labels each candidate by comparing its parsed final answer with the reference,
standardizes each feature using training means and standard deviations, and
fits binary logistic regression with an L2 penalty using Newton updates.
At inference, `code/methods/text_features.py` applies the saved scaling and
weights to each candidate and selects the highest score within the question.
The shared logistic intercept is omitted from the checkpoint because it cannot
change the ordering of four candidates. The model does not use embeddings or
references during inference.

## Parameters

| Item | Value |
|---|---|
| Training questions | 4,755 |
| Training candidate labels | 19,020, of which 14,353 are correct |
| Features | 20, float64 during fitting and inference |
| Objective | Binary logistic loss plus L2 weight penalty |
| L2 regularization coefficient | 1.0 |
| Newton iteration cap | 30 |
| Newton stopping threshold | Maximum absolute step below 1e-7 |
| Feature scaling | Training mean and standard deviation; near-zero scale set to 1 |
| Prediction | Argmax of four candidate scores; lowest index wins ties |
| Random seed | Not applicable; deterministic fitting |
| Checkpoint | `code/runs/text_features/model.npz` |
| Checkpoint size | About 1.2 KiB |
| Checkpoint SHA-256 | `9a488e9098cc9d7c0245f0ad72fa5330c7b9124bc5f4d1d4ac625172a4fa7218` |

## Validation result

Environment: Miniconda `mp1`, Python 3.12.14, NumPy 2.5.3.
Official validation scorer: **719 / 798 = 90.10025062656642%**.
The answer agreement baseline scored 688 / 798 on the same split.
The prediction runner reported 0.191 seconds; this does not include all
process startup and is not the required complete evaluation time.

```bash
cd code
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  conda run -n mp1 python train_text_features.py \
  --output runs/text_features/model.npz
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  conda run -n mp1 python predict.py --method methods/text_features.py \
  --checkpoint runs/text_features/model.npz --split validation \
  --output runs/text_features/validation-predictions.json
conda run -n mp1 python score_predictions.py --split validation \
  --predictions runs/text_features/validation-predictions.json \
  --output runs/text_features/validation-score.json
```

`runs/` is currently Git-ignored. A final submission using this model must
publish its matching checkpoint separately so reviewers can reproduce the
score without retraining. No test result is used for method selection here.

## Test evaluation

Using the unchanged model checkpoint, the official test scorer reported
**350 / 467 = 74.94646680942184%**. Files:
`code/runs/text_features/test-predictions.json` and
`code/runs/text_features/test-score.json`. Local fresh-process timings were
0.36 seconds for prediction and 0.07 seconds for scoring (Apple Silicon
`arm64`, Miniconda `mp1`); the reference Xeon determines eligibility.

```bash
cd code
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  conda run -n mp1 python predict.py --method methods/text_features.py \
  --checkpoint runs/text_features/model.npz --split test \
  --output runs/text_features/test-predictions.json
conda run -n mp1 python score_predictions.py --split test \
  --predictions runs/text_features/test-predictions.json \
  --output runs/text_features/test-score.json
```
