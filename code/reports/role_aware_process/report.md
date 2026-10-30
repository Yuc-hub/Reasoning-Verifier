# Role-aware process verifier ablation

## Motivation and implementation

The earlier process verifier checks whether arithmetic steps are locally
valid and connected to the final answer. Five-fold train-only OOF error
inspection found cases where calculations were internally consistent but the
candidate answered the wrong target quantity, such as original price instead
of amount paid, or omitted a percentage condition. This method adds eight
fixed, low-cost features to the 1,082-dimensional process-verifier model.

`code/methods/role_aware_process.py` extracts the last question phrase
beginning with a target cue such as `how many`, `how much`, or `what is`.
It compares that phrase with the candidate's text immediately preceding
its final-answer marker. Features measure target keyword recall and overlap,
whether a small vocabulary of target roles appears in the conclusion, and
two narrow conflicts: asked amount paid but concluding original price, or
asked amount remaining but concluding amount used. It also checks whether
question percentages (converted to factors) and currency amounts appear as
literals in locally verified equations, plus whether the conclusion mentions
a target unit. These are **lexical and numeric proxies**, not a semantic
proof. A candidate can legitimately use a synonym outside the vocabulary or
use a percentage as `25/100` rather than `0.25`, which this feature may miss.

The original 1,082 features, group-choice loss and training hyperparameters
remain unchanged. Their stored training means and scales were verified equal
to the process-verifier checkpoint. Training code:
`code/train_role_aware_process.py`. No reference answer is read at inference.

## Parameters and assets

| Item | Value |
|---|---|
| Training questions | 4,755 |
| Candidate features | 1,082 process + 8 role/source = 1,090 |
| Objective | Summed multi-correct group softmax loss + `lambda/2 * ||w||²` |
| L2 coefficient | 25.0 |
| Optimizer | Deterministic full-batch Adam |
| Epochs / learning rate | 300 / 0.02 |
| Adam settings | `beta1=0.9`, `beta2=0.999`, epsilon `1e-8` |
| Training objective | 1800.859 → 689.628 |
| Random seed | Not applicable for full training |
| Checkpoint | `code/runs/role_aware_process/model.npz` |
| Checkpoint size | About 14 KiB |
| Checkpoint SHA-256 | `680c173df6c42dcbb02295d24acf6b38b1b0d8e3b5003d5788bce72885d6cb4e` |

## Official validation and exploratory test results

Environment: Miniconda `mp1`, Python 3.12.14, NumPy 2.5.3.

| Method | Validation | Test |
|---|---:|---:|
| Process verifier | 742 / 798 = 92.98245614035088% | 391 / 467 = 83.72591006423983% |
| + Target-role/source features | 743 / 798 = 93.10776942355889% | 394 / 467 = 84.36830835117773% |

Relative to the process verifier, 30 validation selections changed: two
improved and one regressed. On test, nine changed: three improved and none
regressed. These are small differences. The test score is exploratory because
earlier test results were visible; no code or parameter was changed after
this evaluation. Parser coverage on validation: 22 / 3,192 candidates had
a target-role conflict, 207 used a recognized question percentage factor in
a verified equation, and 746 used a recognized question money amount.
The corresponding test counts were 16, 122 and 446 of 1,868 candidates.

## Train-only OOF check

With the same seed-2026 five-fold split as the earlier process analysis,
hard-group OOF correct was **1,730 / 2,114**, compared with
**1,731 / 2,114** for the process verifier; overall OOF correct was
4,371 / 4,755 versus 4,372 / 4,755. The role features were designed after
inspecting those OOF errors, so this is diagnostic and not an independent
holdout. The OOF non-improvement and small validation/test gains do not
establish stable generalization.

```bash
cd code
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  conda run -n mp1 python train_role_aware_process.py \
  --output runs/role_aware_process/model.npz
for split in validation test; do
  OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    conda run -n mp1 python predict.py \
    --method methods/role_aware_process.py \
    --checkpoint runs/role_aware_process/model.npz --split "$split" \
    --output "runs/role_aware_process/$split-predictions.json"
  conda run -n mp1 python score_predictions.py --split "$split" \
    --predictions "runs/role_aware_process/$split-predictions.json" \
    --output "runs/role_aware_process/$split-score.json"
done
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  conda run -n mp1 python analyze_oof_errors.py \
  --method role_aware_process \
  --output runs/oof_analysis/role-aware-summary.json
```

Local fresh-process timing with the `mp1` Python executable and four thread
environment variables: validation prediction 1.78 s + scoring 0.07 s;
test prediction 1.18 s + scoring 0.06 s. The local machine is Apple Silicon
`arm64`; the reference Xeon remains the eligibility machine. The checkpoint
is Git-ignored under `runs/` and must be published if this model is submitted.
