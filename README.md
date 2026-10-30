# Reasoning Verifier (DASE7506 MP2)

This repository contains an offline reranker for four candidate reasoning traces per math question. The frozen submission uses calculation-chain checks, question-condition cues, candidate comparison, fixed supplied embeddings, and target/source alignment. It selects one of the four supplied candidates without generating new text.

**Official result:** validation 743/798 (93.10776942355889%); exploratory test 394/467 (84.36830835117773%). The test bank was viewed during later method development, so this result is not an untouched holdout or a validation-only method selection. The limitation is disclosed in the report.

- [Experimental report (PDF)](code/reports/final_report.pdf) and [pdflatex source](code/reports/final_report.tex)
- [Reproduction and submission commands](code/SUBMISSION.md)
- [Full experiment log](code/EXPERIMENTS.md) and [per-method reports](code/reports/)
- [Frozen entry point](code/submission.py), [matching checkpoint](code/checkpoints/role_aware_process/model.npz), and [official score JSON](code/results/role_aware_process/test-score.json)

From `code/`, install Python 3.12 with `numpy==2.5.3`, then run:

```bash
python check_data.py
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  python predict.py --method submission.py \
  --checkpoint checkpoints/role_aware_process/model.npz \
  --split test --output runs/submission/test-predictions.json
python score_predictions.py --split test \
  --predictions runs/submission/test-predictions.json \
  --output runs/submission/test-score.json
```

The checkpoint SHA-256 is `680c173df6c42dcbb02295d24acf6b38b1b0d8e3b5003d5788bce72885d6cb4e`. It includes all learned means, scales, and weights. Dataset attribution and licenses are retained under `code/data/`.
