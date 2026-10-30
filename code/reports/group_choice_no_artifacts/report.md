# Length and answer-format feature ablation

## Method and rationale

This experiment tests whether six **explicit hand-crafted** surface features
help the group-choice model: log character count, log word count, log line
count, boxed-answer marker, `Final Answer` marker and `####` marker. These
columns are removed, then the same multi-correct group-choice model is
retrained from the training split. The remaining 1,038 features comprise 14
text features and all 1,024 supplied embedding dimensions. The objective,
regularization, optimizer and epochs match the original group-choice model.

This is a limited artifact ablation. The embeddings may still encode length,
format or stylistic cues, and retained numeric/operator counts may correlate
with text length. Therefore a similar score does **not** prove the model is
free of surface bias. Inference is in
`code/methods/group_choice_no_artifacts.py`; training is in
`code/train_group_choice_no_artifacts.py`.

## Parameters

| Item | Value |
|---|---|
| Training questions | 4,755 |
| Candidate features | 1,038 after removing six explicit features |
| Objective | Summed multi-correct group softmax loss + `lambda/2 * ||w||²` |
| L2 coefficient | 25.0 |
| Optimizer | Deterministic full-batch Adam |
| Epochs / learning rate | 300 / 0.02 |
| Adam settings | `beta1=0.9`, `beta2=0.999`, epsilon `1e-8` |
| Training objective | 1800.859 → 754.626 |
| Random seed | Not applicable |
| Checkpoint | `code/runs/group_choice_no_artifacts/model.npz` |
| Checkpoint size | About 13 KiB |
| Checkpoint SHA-256 | `6a47c0b5f369d7a41b03a9425bca155d8421f274410b9187d30d33bb6e58b2c0` |

The retained feature scales match the original checkpoint exactly; the
largest retained mean difference is only `2.41e-20` from floating-point
reduction order.

## Official validation and exploratory test results

Environment: Miniconda `mp1`, Python 3.12.14, NumPy 2.5.3.

| Model | Validation | Test |
|---|---:|---:|
| Original group choice | 742 / 798 = 92.98245614035088% | 385 / 467 = 82.44111349036403% |
| Without six surface features | 743 / 798 = 93.10776942355889% | 383 / 467 = 82.01284796573876% |

Validation changed 24 selections relative to the original model: one
previously wrong choice improved and none regressed. Test changed 12:
one improved and three regressed. These small opposite-direction changes do
not establish that the six features help or harm generalization. The test
result is exploratory; no code or parameter was changed after seeing it.

```bash
cd code
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  conda run -n mp1 python train_group_choice_no_artifacts.py \
  --output runs/group_choice_no_artifacts/model.npz
for split in validation test; do
  OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    conda run -n mp1 python predict.py \
    --method methods/group_choice_no_artifacts.py \
    --checkpoint runs/group_choice_no_artifacts/model.npz --split "$split" \
    --output "runs/group_choice_no_artifacts/$split-predictions.json"
  conda run -n mp1 python score_predictions.py --split "$split" \
    --predictions "runs/group_choice_no_artifacts/$split-predictions.json" \
    --output "runs/group_choice_no_artifacts/$split-score.json"
done
```

Local fresh-process timing with the `mp1` Python executable: validation
prediction 0.33 s + scoring 0.07 s; test prediction 0.19 s + scoring 0.06 s.
The local machine is Apple Silicon `arm64`; the reference Xeon remains the
eligibility machine. The checkpoint is Git-ignored under `runs/` and must be
published for a submission using this model.
