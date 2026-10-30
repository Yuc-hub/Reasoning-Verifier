# Question-condition coverage feature ablation

## Research question and implementation

Does checking whether candidates **use** conditions from the question help
the four-candidate group-choice model? This experiment keeps the original
1,044 candidate features, multi-correct group loss, optimizer and training
hyperparameters, then appends ten condition-coverage features. The original
1,044 training means and scales were verified equal to the group-choice
checkpoint.

`code/methods/condition_coverage.py` extracts distinct numeric quantities
from each question. It separately measures whether each quantity appears in
a candidate and whether it participates in an explicit numeric expression
such as `5 × 17` or `7 / 1.40`. Participation is detected by a mathematical
operator between adjacent numeric tokens, so it is stronger evidence than a
mere mention, but it cannot prove the calculation applies the condition
correctly. Numeric values are canonicalized with rational arithmetic so
`1.40` and `1.4` match.

The extractor also recognizes small predefined groups of physical, money
and time units, plus condition phrases such as `each`, `remaining`, `half`,
`twice`, `before`, `after`, `including`, `excluding`, `discount`, `tax` and
`profit`. It checks whether relevant units occur near a calculation and
whether simple `half` or `twice` operations are visible. These are lexical
proxies: synonyms outside the fixed vocabulary, implicit reasoning, variable
algebra and cross-sentence use can be missed. A missing or unused condition
is a model feature, never a hard rejection. No test or validation reference
is read during inference.

The ten features are: question-number mention fraction, question-number
calculation fraction, mentioned-but-not-calculated fraction, unit-group
mention fraction, unit-group-near-calculation fraction, time-unit mention,
time-unit-near-calculation, constraint-group mention fraction, visible
half-or-double operation, and missing-condition fraction. Training code is
`code/train_condition_coverage.py`.

## Parameters

| Item | Value |
|---|---|
| Training questions | 4,755 |
| Correct training candidates | 14,353 / 19,020 |
| Candidate features | 1,044 original + 10 condition = 1,054 |
| Objective | Summed multi-correct group softmax loss + `lambda/2 * ||w||²` |
| L2 coefficient | 25.0 |
| Optimizer | Deterministic full-batch Adam |
| Epochs / learning rate | 300 / 0.02 |
| Adam settings | `beta1=0.9`, `beta2=0.999`, epsilon `1e-8` |
| Training objective | 1800.859 → 744.258 |
| Random seed | Not applicable; no random initialization or shuffling |
| Checkpoint | `code/runs/condition_coverage/model.npz` |
| Checkpoint size | About 13 KiB |
| Checkpoint SHA-256 | `c6f09d885d9e1d1fbc2176dad8fafdc2b8c1aaeaeedb125a6141bf8467627102` |

## Official validation and exploratory test results

Environment: Miniconda `mp1`, Python 3.12.14, NumPy 2.5.3.

| Method | Validation | Test |
|---|---:|---:|
| Original group choice | 742 / 798 = 92.98245614035088% | 385 / 467 = 82.44111349036403% |
| + Condition coverage | 740 / 798 = 92.73182957393483% | 386 / 467 = 82.65524625267666% |

On validation, 3,133 / 3,192 candidates mentioned at least one question
number, while 2,729 / 3,192 placed one in a recognized calculation. On test,
the corresponding counts were 1,846 / 1,868 and 1,590 / 1,868. Recognized
unit groups were mentioned by 1,471 validation and 1,037 test candidates;
1,346 and 935 placed one near a calculation. These counts describe parser
coverage, not proof of correct reasoning.

Compared with the original model, 55 validation selections changed: four
improved and six regressed. On test, 27 changed: three improved and two
regressed. The validation and test changes have opposite signs and are small.
This test result is exploratory because earlier test scores were already
visible; no code or parameter was changed after seeing this result.

```bash
cd code
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  conda run -n mp1 python train_condition_coverage.py \
  --output runs/condition_coverage/model.npz
for split in validation test; do
  OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    conda run -n mp1 python predict.py \
    --method methods/condition_coverage.py \
    --checkpoint runs/condition_coverage/model.npz --split "$split" \
    --output "runs/condition_coverage/$split-predictions.json"
  conda run -n mp1 python score_predictions.py --split "$split" \
    --predictions "runs/condition_coverage/$split-predictions.json" \
    --output "runs/condition_coverage/$split-score.json"
done
```

Local fresh-process timing with the `mp1` Python executable and four thread
environment variables: validation prediction 0.85 s + scoring 0.06 s;
test prediction 0.56 s + scoring 0.05 s. The local machine is Apple Silicon
`arm64`; the reference Xeon remains the eligibility machine. `runs/` is
Git-ignored, so a final submission using this model must publish its matching
checkpoint.
