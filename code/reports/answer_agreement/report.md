# Answer agreement baseline

## Idea and implementation

For each question, use the official `common.parse_answer` function to extract
the final numeric answer from each of the four candidate texts. Count how many
candidates have each nonmissing parsed answer, then select a candidate from
the largest answer group. Equivalent fractions and decimals share the same
canonical value because the parser returns a reduced `Fraction` string.

Implementation: `code/methods/answer_agreement.py`. The method uses candidate
texts only. It does not use embeddings, references, or learned parameters.

## Parameters and tie rules

| Item | Value |
|---|---|
| Candidates per question | 4, supplied in fixed order |
| Score | Number of candidates with the same parsed final answer |
| Tie break | Lowest candidate index |
| All answers unparseable | Candidate index 0 |
| Training | None |
| Random seed | Not applicable |
| Checkpoint / inference assets | None |

## Validation result

Environment: Miniconda `mp1`, Python 3.12.14, NumPy 2.5.3.
Official validation scorer: **688 / 798 = 86.2155388471178%**.
The prediction runner reported 0.070 seconds; this does not include all
process startup and is not the required complete evaluation time.

```bash
cd code
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  conda run -n mp1 python predict.py --method methods/answer_agreement.py \
  --split validation --output runs/answer_agreement/validation-predictions.json
conda run -n mp1 python score_predictions.py --split validation \
  --predictions runs/answer_agreement/validation-predictions.json \
  --output runs/answer_agreement/validation-score.json
```

Validation contains easier groups than test. No test result is used here.

## Test evaluation

With the same fixed method, the official test scorer reported **323 / 467 =
69.16488222698072%**. Files: `code/runs/answer_agreement/test-predictions.json`
and `code/runs/answer_agreement/test-score.json`. Local fresh-process timings
were 0.27 seconds for prediction and 0.07 seconds for scoring (Apple Silicon
`arm64`, Miniconda `mp1`); the reference Xeon determines eligibility.

```bash
cd code
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  conda run -n mp1 python predict.py --method methods/answer_agreement.py \
  --split test --output runs/answer_agreement/test-predictions.json
conda run -n mp1 python score_predictions.py --split test \
  --predictions runs/answer_agreement/test-predictions.json \
  --output runs/answer_agreement/test-score.json
```
