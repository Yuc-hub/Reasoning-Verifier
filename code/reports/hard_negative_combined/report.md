# Highest-scoring wrong-candidate training ablation

## Method and order

This was the second experiment, after the fixed combined-model cluster rule
was measured. It keeps the combined-reasoning model's 1,073 features,
training labels, feature scaling, 300-epoch full-batch Adam optimizer and
original multi-correct group loss. The training means and scales were
verified exactly equal to the original combined checkpoint.

For each training question with both correct and wrong candidates, the model
finds the currently highest-scoring wrong candidate and highest-scoring
correct candidate. In addition to the group loss, it pays a smooth margin
penalty `0.5 * log(1 + exp(1 + score_hard_wrong - score_best_correct))`.
The selected wrong candidate can change during training. This is train-only
hard-negative selection within each group. Group softmax already emphasizes
high-scoring wrong candidates, so the term tests **extra** emphasis.

Training: `code/train_hard_negative_combined.py`. Single-candidate inference:
`code/methods/hard_negative_combined.py`. The fixed answer-cluster rule can
also consume the new checkpoint through
`code/methods/answer_cluster_combined.py`.

## Parameters and assets

| Item | Value |
|---|---|
| Training questions | 4,755 |
| Candidate features | Same 1,073 combined features |
| Base objective | Multi-correct four-way group softmax loss |
| Auxiliary objective | Softplus of margin + hard-wrong score − best-correct score |
| Auxiliary weight / target margin | 0.5 / 1.0, fixed before evaluation |
| L2 coefficient | 25.0 |
| Optimizer | Deterministic full-batch Adam |
| Epochs / learning rate | 300 / 0.02 |
| Adam settings | `beta1=0.9`, `beta2=0.999`, epsilon `1e-8` |
| Combined training objective | 3450.972 → 1232.137 |
| Random seed | Not applicable; no random initialization or shuffling |
| Checkpoint | `code/runs/hard_negative_combined/model.npz` |
| Checkpoint size | About 13 KiB |
| Checkpoint SHA-256 | `a57391401524108a8e23046e941ba5589d58af9cfacd359db0934293fbe220e0` |

## Evaluation

Environment: Miniconda `mp1`, Python 3.12.14, NumPy 2.5.3.

| Model and rule | Validation | Exploratory test |
|---|---:|---:|
| Original combined, single candidate | 739 / 798 = 92.60651629072682% | 389 / 467 = 83.29764453961457% |
| Original combined, answer cluster | 744 / 798 = 93.23308270676692% | 384 / 467 = 82.22698072805139% |
| Hard-negative combined, single candidate | 745 / 798 = 93.35839598997494% | 384 / 467 = 82.22698072805139% |
| Hard-negative combined, answer cluster | 743 / 798 = 93.10776942355889% | 383 / 467 = 82.01284796573876% |

Against original combined single-candidate inference, hard-negative training
changed 67 validation choices (eight improved, two regressed) and 31 test
choices (three improved, eight regressed). Its cluster variant changed 72
validation choices (nine improved, five regressed) and 42 test choices
(eight improved, fourteen regressed). The auxiliary weight and margin were
not changed after seeing either split's score.

## Five-fold train-only error check

Using the same seed-2026 fold assignment as the original OOF analysis,
hard-group correct rose from 1,709 / 2,114 for combined reasoning to
**1,725 / 2,114** with the auxiliary loss. Overall OOF correct rose from
4,350 / 4,755 to 4,366 / 4,755. The idea was developed after inspecting
the original OOF errors, so this is diagnostic rather than untouched
validation evidence. Validation gain and exploratory test decline point in
opposite directions; no stable improvement has been established.

```bash
cd code
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  conda run -n mp1 python train_hard_negative_combined.py \
  --output runs/hard_negative_combined/model.npz
for split in validation test; do
  OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    conda run -n mp1 python predict.py \
    --method methods/hard_negative_combined.py \
    --checkpoint runs/hard_negative_combined/model.npz --split "$split" \
    --output "runs/hard_negative_combined/$split-predictions.json"
  conda run -n mp1 python score_predictions.py --split "$split" \
    --predictions "runs/hard_negative_combined/$split-predictions.json" \
    --output "runs/hard_negative_combined/$split-score.json"
  OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    conda run -n mp1 python predict.py \
    --method methods/answer_cluster_combined.py \
    --checkpoint runs/hard_negative_combined/model.npz --split "$split" \
    --output "runs/hard_negative_cluster/$split-predictions.json"
  conda run -n mp1 python score_predictions.py --split "$split" \
    --predictions "runs/hard_negative_cluster/$split-predictions.json" \
    --output "runs/hard_negative_cluster/$split-score.json"
done
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  conda run -n mp1 python analyze_oof_errors.py \
  --method hard_negative_combined \
  --output runs/oof_analysis/hard-negative-summary.json
```

Local fresh-process timing for single-candidate prediction plus scoring:
validation 1.19 s + 0.07 s; test 0.80 s + 0.06 s. For the cluster variant:
validation 1.17 s + 0.07 s; test 0.81 s + 0.06 s. The local machine is
Apple Silicon `arm64`; the reference Xeon remains the eligibility machine.
The checkpoint is Git-ignored under `runs/` and must be published if submitted.
