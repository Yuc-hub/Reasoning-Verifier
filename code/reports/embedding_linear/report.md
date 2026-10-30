# Fixed embedding linear reranker

## Idea and implementation

Each candidate has a supplied 1024-dimensional whole-trace embedding. The
training script labels each training candidate by comparing its parsed final
answer with the training reference answer. It standardizes each embedding
dimension using training means and standard deviations, then fits a linear
ridge-regression model to the binary correctness labels. The score is

`score(candidate) = standardized_embedding @ weights`.

The trained global intercept is unnecessary for ranking the four candidates
of one question, so it is not saved. At inference, the method ranks the four
scores and returns the highest-scoring candidate index. Candidate texts are
not used at inference, except that texts were needed to create training labels.
The supplied fixed embeddings are inputs, not additional model assets.

Implementation: `code/methods/embedding_linear.py` and
`code/train_embedding_linear.py`. The method is independent of the answer
agreement and text feature methods; select it with `predict.py --method`.

## Parameters

| Item | Value |
|---|---|
| Training questions | 4,755 |
| Training candidate labels | 19,020, of which 14,353 are correct |
| Embedding dimensions | 1,024, supplied float32 |
| Fit precision | float64 |
| Objective | Sum of squared prediction errors plus ridge coefficient times squared weight norm |
| Ridge coefficient | 100.0 |
| Scaling | Training mean and standard deviation per dimension; near-zero scale set to 1 |
| Prediction | Highest of four linear scores; lowest index wins ties |
| Random seed | Not applicable; deterministic closed-form fitting |
| Checkpoint | `code/runs/embedding_linear/model.npz` |
| Checkpoint size | About 13 KiB uncompressed file; three 1,024-element float32 arrays, 12 KiB of learned arrays |
| Checkpoint SHA-256 | `e449b826d89dece073447aa21a2bcd6c64f06711b8c7997a523d3f78f0d1e05a` |

## Validation result

Environment: Miniconda `mp1`, Python 3.12.14, NumPy 2.5.3.
Official validation scorer: **712 / 798 = 89.22305764411027%**.
For comparison, answer agreement scored 688 / 798 and text features scored
719 / 798 on the same validation set. The prediction runner reported 0.062
seconds, excluding some process startup; this is not the required complete
evaluation time.

```bash
cd code
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  conda run -n mp1 python train_embedding_linear.py \
  --output runs/embedding_linear/model.npz
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  conda run -n mp1 python predict.py --method methods/embedding_linear.py \
  --checkpoint runs/embedding_linear/model.npz --split validation \
  --output runs/embedding_linear/validation-predictions.json
conda run -n mp1 python score_predictions.py --split validation \
  --predictions runs/embedding_linear/validation-predictions.json \
  --output runs/embedding_linear/validation-score.json
```

The test split has not been used for method selection. `runs/` is Git-ignored,
so a final submission using this model must publish the matching checkpoint
for reproduction without retraining.

## Test evaluation

Using the unchanged model checkpoint, the official test scorer reported
**360 / 467 = 77.08779443254818%**. Files:
`code/runs/embedding_linear/test-predictions.json` and
`code/runs/embedding_linear/test-score.json`. Local fresh-process timings
were 0.27 seconds for prediction and 0.07 seconds for scoring (Apple Silicon
`arm64`, Miniconda `mp1`); the reference Xeon determines eligibility.

```bash
cd code
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  conda run -n mp1 python predict.py --method methods/embedding_linear.py \
  --checkpoint runs/embedding_linear/model.npz --split test \
  --output runs/embedding_linear/test-predictions.json
conda run -n mp1 python score_predictions.py --split test \
  --predictions runs/embedding_linear/test-predictions.json \
  --output runs/embedding_linear/test-score.json
```
