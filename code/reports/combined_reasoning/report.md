# Combined reasoning-feature group-choice model

## Research question and implementation

Do four complementary feature families help when used together? This method
retains the original group-choice model's 1,044 candidate features and adds:

1. **Arithmetic consistency (7):** counts of recognized simple equations,
   erroneous equations, and whether equation results connect to the final
   answer.
2. **Question-condition coverage (10):** whether question numbers, units,
   time and constraint phrases are mentioned or visibly used in calculations.
3. **Candidate divergence localization (8):** numeric-token and equation-result
   sequence overlap and shared-prefix measures among the four candidates.
4. **Final-answer/process links (4):** fraction of the last three explicit
   equation results matching the final answer, whether the last result
   conflicts with it, whether the last arithmetically verified result matches
   it, and whether any verified result equals it.

The total is **1,073 candidate features**. All features are computed from
supplied question/candidate inputs. The model fits the same multi-correct
four-way softmax loss and uses the same full-batch Adam settings as the
original group-choice model. The first 1,044 training means and scales were
verified equal to the original checkpoint. Training code is
`code/train_combined_reasoning.py`; inference code is
`code/methods/combined_reasoning.py`.

These feature families are partly correlated, especially arithmetic
consistency and final-answer links. They are surface-level checks: a visible
equation can be correct while the problem interpretation is wrong, and a
candidate with a wrong intermediate step may still have the correct final
numeric answer. The model learns feature weights; no rule automatically
rejects a candidate. No reference answer is used during inference.

## Parameters

| Item | Value |
|---|---|
| Training questions | 4,755 |
| Correct training candidates | 14,353 / 19,020 |
| Candidate features | 1,044 base + 7 arithmetic + 10 condition + 8 divergence + 4 final link = 1,073 |
| Objective | Summed multi-correct group softmax loss + `lambda/2 * ||w||²` |
| L2 coefficient | 25.0 |
| Optimizer | Deterministic full-batch Adam |
| Epochs / learning rate | 300 / 0.02 |
| Adam settings | `beta1=0.9`, `beta2=0.999`, epsilon `1e-8` |
| Training objective | 1800.859 → 731.808 |
| Random seed | Not applicable; no random initialization or shuffling |
| Checkpoint | `code/runs/combined_reasoning/model.npz` |
| Checkpoint size | About 13 KiB |
| Checkpoint SHA-256 | `fdcd602d65d2e39fbe9b67c59ce5d3f233e5c415047cfa0b7ae16d6ce0f4bf6e` |

## Official validation and exploratory test results

Environment: Miniconda `mp1`, Python 3.12.14, NumPy 2.5.3.

| Method | Validation | Test |
|---|---:|---:|
| Original group choice | 742 / 798 = 92.98245614035088% | 385 / 467 = 82.44111349036403% |
| Combined reasoning | 739 / 798 = 92.60651629072682% | 389 / 467 = 83.29764453961457% |

Relative to the original model, 125 validation selections changed: five
improved and eight regressed. On test, 44 changed: eight improved and four
regressed. The opposite direction across splits prevents a claim of stable
improvement. This test result is exploratory because earlier test scores were
already visible; no code or parameter was changed after seeing this result.

For the final-link feature family, 570 / 3,192 validation candidates and
328 / 1,868 test candidates had a last explicit equation result different
from their parsed final answer. At least one arithmetically verified equation
result matched the final answer in 1,502 validation and 823 test candidates.
These counts describe parser outputs, not verified full reasoning.

```bash
cd code
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  conda run -n mp1 python train_combined_reasoning.py \
  --output runs/combined_reasoning/model.npz
for split in validation test; do
  OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    conda run -n mp1 python predict.py \
    --method methods/combined_reasoning.py \
    --checkpoint runs/combined_reasoning/model.npz --split "$split" \
    --output "runs/combined_reasoning/$split-predictions.json"
  conda run -n mp1 python score_predictions.py --split "$split" \
    --predictions "runs/combined_reasoning/$split-predictions.json" \
    --output "runs/combined_reasoning/$split-score.json"
done
```

Local fresh-process timing with the `mp1` Python executable and four thread
environment variables: validation prediction 1.12 s + scoring 0.06 s;
test prediction 0.76 s + scoring 0.06 s. The local machine is Apple Silicon
`arm64`; the reference Xeon remains the eligibility machine. `runs/` is
Git-ignored, so a final submission using this model must publish its matching
checkpoint.
