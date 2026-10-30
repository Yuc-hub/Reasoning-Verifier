# Calculation-chain process verifier features

## Method

This method augments the 1,073-dimensional combined-reasoning group-choice
model with nine features from a conservative calculation-chain verifier.
The verifier reads candidate reasoning text, finds explicit equality chains,
and parses arithmetic-only expressions using a restricted Python AST
evaluator. It supports numeric constants, parentheses, unary signs,
addition, subtraction, multiplication, division, simple numeric `\\frac`
forms and percentages. It rejects names, function calls and unsupported
syntax; it does not execute candidate text as code.

For each parsed equality, the verifier checks whether its two sides agree
within an absolute tolerance of 0.001. It links a verified result to a later
step when that value appears as a later operand. This creates a small
chronological calculation graph. It then traces whether the parsed final
answer matches a verified result, how deep that derivation is, and how many
question numbers are connected to it. The nine features are: log verified
step count, log invalid step count, verified-step fraction, final-chain depth
fraction, final-chain question-number coverage, existence of a verified
final derivation, fraction of verified results reused later, fraction of
orphan verified results, and whether the final answer equals a question
number. Training uses the same multi-correct group-choice objective and
hyperparameters as the combined model. No hard rule excludes a candidate.

Implementation: `code/methods/process_verifier.py` and
`code/train_process_verifier.py`. All training labels come from the supplied
training references; inference uses only supplied questions, candidate
texts, fixed embeddings and the checkpoint.

## Parameters and assets

| Item | Value |
|---|---|
| Training questions | 4,755 |
| Candidate features | 1,073 combined + 9 process = 1,082 |
| Equality tolerance | Absolute rational difference at most 0.001 |
| Objective | Summed multi-correct group softmax loss + `lambda/2 * ||w||²` |
| L2 coefficient | 25.0 |
| Optimizer | Deterministic full-batch Adam |
| Epochs / learning rate | 300 / 0.02 |
| Adam settings | `beta1=0.9`, `beta2=0.999`, epsilon `1e-8` |
| Training objective | 1800.859 → 692.070 |
| Random seed | Not applicable for full training |
| Checkpoint | `code/runs/process_verifier/model.npz` |
| Checkpoint size | About 13 KiB |
| Checkpoint SHA-256 | `c017a9edd5b9edc4c9cbe51a5946797212004b46972bd2636c96ced4e3f5d2ed` |

## Evaluation

Environment: Miniconda `mp1`, Python 3.12.14, NumPy 2.5.3.

| Model | Validation | Exploratory test |
|---|---:|---:|
| Combined reasoning | 739 / 798 = 92.60651629072682% | 389 / 467 = 83.29764453961457% |
| + Process-chain features | 742 / 798 = 92.98245614035088% | 391 / 467 = 83.72591006423983% |

Relative to combined reasoning, validation changed 109 selected candidates:
seven became correct and four became wrong. Test changed 68: sixteen became
correct and fourteen became wrong. Among 3,192 validation candidates, 2,860
had at least one recognized arithmetic step and 2,348 had a verified step
matching their final answer. On test, the corresponding counts were 1,642
and 1,268 of 1,868 candidates. These are parser-coverage counts, not proof
that the full reasoning is valid.

A separate five-fold train-only OOF comparison rose from 1,709 to 1,731
correct among 2,114 hard groups; see `reports/oof_error_analysis/report.md`.
Its feature design was informed by OOF errors, so it is not an untouched
holdout result. The test result is exploratory because earlier test scores
had already been viewed. No model or parameter was changed after this test
run.

```bash
cd code
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  conda run -n mp1 python train_process_verifier.py \
  --output runs/process_verifier/model.npz
for split in validation test; do
  OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    conda run -n mp1 python predict.py \
    --method methods/process_verifier.py \
    --checkpoint runs/process_verifier/model.npz --split "$split" \
    --output "runs/process_verifier/$split-predictions.json"
  conda run -n mp1 python score_predictions.py --split "$split" \
    --predictions "runs/process_verifier/$split-predictions.json" \
    --output "runs/process_verifier/$split-score.json"
done
```

Local fresh-process timing with the `mp1` Python executable and four thread
environment variables: validation prediction 1.42 s + scoring 0.07 s;
test prediction 0.96 s + scoring 0.06 s. The local machine is Apple Silicon
`arm64`; the course reference Xeon remains the eligibility machine.
`runs/` is Git-ignored, so the matching checkpoint must be published for
reproduction if this method is submitted.

## Limits

The graph verifies local numeric calculations and their reuse. It cannot
prove that a quantity was assigned the right role, that a fixed fee was
charged the right number of times, or that a percent applies to the correct
base. Its parser may miss equations expressed in prose or algebra and may
mistake a coincidentally repeated number for a dependency. The official
scorer checks only the final numeric answer, so a locally invalid step can
coexist with a correct final answer. These signals therefore remain learned
features rather than a hard correctness verdict.
