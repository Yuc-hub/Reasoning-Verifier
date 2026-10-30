# Tournament pairwise verification adaptation

## Source and scope

Inspired by Singh et al., [*V1: Unifying Generation and Self-Verification for
Parallel Reasoners*](https://proceedings.mlr.press/v306/singh26b.html), ICML
2026. The paper's V1-Infer uses tournament-based pairwise ranking and
uncertainty-guided allocation of verification compute. This coursework
adaptation compares all six pairs among four supplied candidates exactly once.
It does not reproduce the paper's adaptive compute policy, generator, or
PairRL training. No external weights, datasets, or API calls are used.

## Idea and implementation

For each candidate pair `(j, k)`, concatenate each candidate's 20 text
features and 1024 fixed embedding dimensions. Standardize these dimensions
using training-only means and standard deviations. The pair feature contains:

1. The signed difference `x_j - x_k` (1,044 dimensions).
2. The signed difference of the 20 text features multiplied by pair embedding
   cosine similarity (20 dimensions).
3. The same text difference multiplied by the absolute difference in the
   candidates' standardized log absolute final-answer values
   (20 dimensions).

The 1,084-dimensional feature is antisymmetric under swapping the candidates.
The cosine interactions depend on **both** candidates, so the comparator is
not generally equivalent to independently assigning every candidate one
fixed linear score. Training uses only same-question pairs where one final
answer is correct and the other is wrong. Each pair is oriented so the
correct candidate should win, and a ridge-regression margin objective is
fitted:

`min_w sum_pairs (pair_feature @ w - 1)^2 + 100 * ||w||²`.

At inference, a positive pair margin awards one win to the first candidate;
otherwise the second wins. Each candidate's win rate is its wins divided by
three. The candidate with the most wins is selected. Total signed margin
breaks equal-win ties, followed by lowest index. Training code:
`code/train_tournament_pairwise.py`; inference code:
`code/methods/tournament_pairwise.py`.

## Parameters

| Item | Value |
|---|---|
| Training questions | 4,755 |
| Informative correct/wrong training pairs | 8,271 |
| Pair features | 1,084 |
| Pair comparisons per question | 6 |
| Objective | Pairwise squared margin error with L2 regularization |
| Target margin | 1 |
| Ridge coefficient | 100.0 |
| Scaling | Training mean and standard deviation per candidate feature; near-zero scale set to 1 |
| Fit precision | float64; checkpoint arrays float32 |
| Tournament score | Hard win count, equivalent to win rate divided by 3 |
| Tie break | Total signed margin, then lowest candidate index |
| Random seed | Not applicable; deterministic fitting |
| Checkpoint | `code/runs/tournament_pairwise/model.npz` |
| Checkpoint size | About 13 KiB |
| Checkpoint SHA-256 | `2388d0090f9662052dd705120f1617e2a085ee858b2bc720be0c50bb421b800f` |

## Validation result

Environment: Miniconda `mp1`, Python 3.12.14, NumPy 2.5.3.
Official validation scorer: **732 / 798 = 91.72932330827068%**. The prior
linear pairwise model scored 734 / 798; the two methods selected different
candidates on 56 validation questions. The score-fusion model
scored 738 / 798 on validation. The prediction runner reported 0.241 seconds,
excluding some startup, and the independent scorer reported 0.020 seconds.

```bash
cd code
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  conda run -n mp1 python train_tournament_pairwise.py \
  --output runs/tournament_pairwise/model.npz
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  conda run -n mp1 python predict.py --method methods/tournament_pairwise.py \
  --checkpoint runs/tournament_pairwise/model.npz --split validation \
  --output runs/tournament_pairwise/validation-predictions.json
conda run -n mp1 python score_predictions.py --split validation \
  --predictions runs/tournament_pairwise/validation-predictions.json \
  --output runs/tournament_pairwise/validation-score.json
```

## Interpretation and limits

This is a genuine pair-dependent tournament, but its validation accuracy did
not exceed the existing linear pairwise model. The paper's advantages do not
automatically transfer to a small ridge model over frozen embeddings. The
fixed six-match schedule also omits V1-Infer's adaptive verification compute.
The test bank had already been examined for earlier methods before this
experiment. Its later test score is therefore exploratory and must not be
used for tuning or retrospective method selection.

## Revalidation and exploratory test evaluation

The unchanged checkpoint hash above was checked before rerunning both splits.
The official scorer reconfirmed **732 / 798 = 91.72932330827068%** on
validation and reported **376 / 467 = 80.51391862955032%** on test.
Saved files are `code/runs/tournament_pairwise/{validation,test}-predictions.json`
and corresponding `{validation,test}-score.json` files.

```bash
cd code
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  conda run -n mp1 python predict.py --method methods/tournament_pairwise.py \
  --checkpoint runs/tournament_pairwise/model.npz --split test \
  --output runs/tournament_pairwise/test-predictions.json
conda run -n mp1 python score_predictions.py --split test \
  --predictions runs/tournament_pairwise/test-predictions.json \
  --output runs/tournament_pairwise/test-score.json
```

Local fresh-process timing with the `mp1` Python executable was 0.43 seconds
for validation prediction plus 0.07 seconds for validation scoring, and 0.22
seconds for test prediction plus 0.06 seconds for test scoring. Hardware is
Apple Silicon `arm64`; the course reference Xeon determines eligibility.
The test score is below the unchanged linear pairwise model's 378 / 467 and
fusion model's 377 / 467. No model, hyperparameter, or checkpoint was changed
after this test evaluation.
