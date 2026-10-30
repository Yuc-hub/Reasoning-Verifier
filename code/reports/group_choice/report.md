# Local four-candidate group-choice model

## Idea and implementation

This is a small, local, Jev-inspired one-pass choice model, **not Jev itself**.
It uses no external model weights, data, or API calls. For each question it
receives the four supplied candidates together and scores their existing
1,044-dimensional features (20 text features and 1,024 fixed embeddings).
The text features already include answer agreement and several within-group
relative values. A shared linear reward head produces four scores, and a
group softmax converts them to four choice probabilities.

Training allows multiple correct candidates. If `C` is the set of correct
candidate indices for one question and `p_j` is the group-softmax probability,
the loss is `-log(sum_{j in C} p_j)`. This matches the official rule that any
correct candidate earns the point. L2 regularization is added across all
training questions. At inference, the shared head selects the largest score;
the softmax need not be computed because it preserves score ordering.

The model's distinction from Bradley–Terry is its **training objective**:
Bradley–Terry compares correct/wrong pairs; this model optimizes the
probability mass on all correct candidates in each four-way group. Its
inference scorer is deliberately the same shared linear head, isolating the
effect of the objective. Code: `code/train_group_choice.py` and
`code/methods/group_choice.py`.

## Parameters

| Item | Value |
|---|---|
| Training questions | 4,755 |
| Correct candidates | 14,353 of 19,020 |
| Candidate features | 20 text + 1,024 embedding = 1,044 |
| Feature scaling | Training mean and standard deviation; near-zero scale set to 1 |
| Objective | Summed multi-correct group softmax loss + `lambda/2 * ||w||²` |
| L2 coefficient `lambda` | 25.0 |
| Optimizer | Deterministic full-batch Adam |
| Epochs | 300 |
| Learning rate | 0.02 |
| Adam moments | `beta1=0.9`, `beta2=0.999`, denominator epsilon `1e-8` |
| Initial / final training objective | 1800.859 / 753.485 |
| Prediction | Highest of four shared-head scores; lowest index wins ties |
| Random seed | Not applicable; no random initialization or shuffling |
| Checkpoint | `code/runs/group_choice/model.npz` |
| Checkpoint size | About 13 KiB |
| Checkpoint SHA-256 | `56fb8d564f7296f37415184c00639b6b9a1a89e33e82a26fe08e5ba967af28a5` |

## Official validation and exploratory test results

Environment: Miniconda `mp1`, Python 3.12.14, NumPy 2.5.3.
Validation: **742 / 798 = 92.98245614035088%**.
Test: **385 / 467 = 82.44111349036403%**.
Bradley–Terry scored 739 / 798 and 382 / 467, respectively; the group-choice
model selected three more correct questions on each split. These are small
differences. Test results of earlier methods had been viewed before this
experiment; this test result is exploratory and was not used to change the
model after evaluation.

```bash
cd code
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  conda run -n mp1 python train_group_choice.py \
  --output runs/group_choice/model.npz
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  conda run -n mp1 python predict.py --method methods/group_choice.py \
  --checkpoint runs/group_choice/model.npz --split validation \
  --output runs/group_choice/validation-predictions.json
conda run -n mp1 python score_predictions.py --split validation \
  --predictions runs/group_choice/validation-predictions.json \
  --output runs/group_choice/validation-score.json
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  conda run -n mp1 python predict.py --method methods/group_choice.py \
  --checkpoint runs/group_choice/model.npz --split test \
  --output runs/group_choice/test-predictions.json
conda run -n mp1 python score_predictions.py --split test \
  --predictions runs/group_choice/test-predictions.json \
  --output runs/group_choice/test-score.json
```

Local fresh-process timing using the `mp1` Python executable and four thread
environment variables: validation prediction 0.25 s + scoring 0.06 s;
test prediction 0.18 s + scoring 0.06 s. The local machine is Apple Silicon
`arm64`; the reference Xeon remains the eligibility machine. `runs/` is
Git-ignored, so a final submission using this model must publish its matching
checkpoint for reproduction without retraining.
