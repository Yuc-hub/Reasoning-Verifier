# Five-fold out-of-fold error analysis

## Purpose and protocol

This analysis uses the **training split only**. With seed 2026, training
questions are randomly assigned to five fixed folds. For each fold, the
combined-reasoning model is retrained on the other four folds using the same
feature pipeline, group-choice objective, L2 coefficient 25, 300 full-batch
Adam epochs and learning rate 0.02. The held-out fold is then predicted.
Every recorded prediction is therefore made by a model that did not train on
that question. No test references, IDs or predictions are used for fitting.

Implementation: `code/analyze_oof_errors.py`. Results are written to
`code/runs/oof_analysis/summary.json` and
`code/runs/oof_analysis/process-summary.json`. These files are **analysis
artifacts**, not inference assets or ID lookup tables.

## Combined-reasoning error findings

| Metric | Result |
|---|---:|
| Training questions | 4,755 |
| Questions with at least two distinct parsed candidate answers | 2,114 |
| Out-of-fold overall correct | 4,350 / 4,755 |
| Out-of-fold hard-group correct | 1,709 / 2,114 = 80.84% |
| Hard-group errors | 405 |

Among the 405 hard-group errors, the selected wrong candidate had these
**nonexclusive heuristic signals**:

| Signal | Error cases |
|---|---:|
| A simple equation flagged as arithmetically wrong | 174 |
| A question number mentioned but not used in a recognized calculation | 197 |
| Last extracted equation result differs from final answer | 76 |
| Same wrong final answer shared by another candidate | 152 |
| None of these signals | 50 |

These counts are not causal explanations. A parser can misread a long
equation; a wrong answer can have entirely correct arithmetic but misuse a
condition. For example, an inspected training error charged a fixed car
rental fee once per travel day, and another treated a discount amount as the
amount paid. Both show why verifying only local arithmetic is insufficient.

## Process-feature follow-up under the same folds

The nine process-chain features described in
`reports/process_verifier/report.md` were added to the combined model. Using
the same five fold assignment and training configuration, overall out-of-fold
correct rose to **4,372 / 4,755** and hard-group correct to
**1,731 / 2,114 = 81.88%**. Compared question by question, 76 baseline errors
were corrected and 54 new errors appeared, a net gain of 22.

The process features were designed after inspecting the original OOF errors,
so the +22 result is **diagnostic evidence, not an untouched independent
validation estimate**. The supplied validation split remains the separate
development comparison. Earlier public test scores were already viewed;
later test results are exploratory.

The later hard-negative training ablation used the same folds and raised
hard-group OOF correct from 1,709 to 1,725 / 2,114. See
`reports/hard_negative_combined/report.md`; its auxiliary loss was also
designed after inspecting OOF errors, so this comparison is diagnostic.

The role-aware process follow-up scored 1,730 / 2,114 hard-group OOF correct,
one fewer than the process verifier; see
`reports/role_aware_process/report.md`. Its target-role features were
designed from the inspected OOF mistakes, so this is diagnostic evidence.

```bash
cd code
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  conda run -n mp1 python analyze_oof_errors.py \
  --method combined_reasoning --output runs/oof_analysis/summary.json
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  conda run -n mp1 python analyze_oof_errors.py \
  --method process_verifier \
  --output runs/oof_analysis/process-summary.json
```

Each five-fold run took about 25 seconds locally in the Miniconda `mp1`
environment. This is analysis/training time, not part of frozen inference.
