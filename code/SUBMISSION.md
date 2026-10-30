# MP2 submission and reproduction

## Frozen entry point

- Method: `submission.py` (`methods/role_aware_process.py`)
- Checkpoint: `checkpoints/role_aware_process/model.npz`
- Checkpoint SHA-256: `680c173df6c42dcbb02295d24acf6b38b1b0d8e3b5003d5788bce72885d6cb4e`
- Asset size: 13,830 bytes on disk; three 1,090-element float32 arrays (13,080 bytes raw), far below 16 MiB.
- Dependencies: Python 3.12.14 and `requirements.txt` (NumPy 2.5.3). No extra weights, network access or API calls.
- Training: supplied train inputs and train references only; deterministic full-batch Adam, no random seed for the final fit.
- Official score JSON: `results/role_aware_process/validation-score.json` and `results/role_aware_process/test-score.json`.
- Report: `final_report.pdf` (3 pages), editable pdflatex source `reports/final_report.tex`; experiment log: `EXPERIMENTS.md`. Per-method working notes are kept locally and excluded from Git.

Run from `code/` after installing Miniconda:

```bash
conda create -n mp1 python=3.12 -y
conda run -n mp1 python -m pip install -r requirements.txt
conda run -n mp1 python check_data.py
```

Reproduce training (optional; evaluation uses the included checkpoint):

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  conda run -n mp1 python train_role_aware_process.py \
  --output runs/retrained-role-aware/model.npz
```

Reproduce official scoring without retraining:

```bash
for split in validation test; do
  OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    conda run -n mp1 python predict.py --method submission.py \
    --checkpoint checkpoints/role_aware_process/model.npz \
    --split "$split" --output "runs/submission/$split-predictions.json"
  conda run -n mp1 python score_predictions.py --split "$split" \
    --predictions "runs/submission/$split-predictions.json" \
    --output "runs/submission/$split-score.json"
done
```

Expected official results: validation 743/798 = 93.10776942355889%; test 394/467 = **84.36830835117773%**. Submit the latter `accuracy` value, not the fraction or rounded 84.37.

The direct `submission.py` entry point also works without `--checkpoint`, using its bundled checkpoint. The explicit path above pins the asset for review. For a new audit bank, pass `--split audit --data-dir PATH`; inference reads only the supplied bank and embeddings, and does not open reference answers.

## Resource record

On local Apple Silicon arm64 with four thread environment variables, an external `/usr/bin/time -p` around one fresh prediction process plus one independent scoring process reported **3.24 s real** (2.66 s user, 0.40 s system). The official runner reported 1.1387 s for test prediction and 0.0120 s for scoring within that measurement. This local timing cannot prove the 30 s reference-Xeon requirement; eligibility is determined on the reference Xeon Platinum 8457C.

## Academic and submission notes

The test bank was repeatedly evaluated during development. This method was developed and chosen after earlier test results were visible. Its test score is therefore **exploratory**; selecting it because it scored highest on test does **not** satisfy the guide's rule to use validation for method selection. This must be disclosed; do not represent the test result as a clean single-use holdout. If strict compliance is required, use a model selected independently of test feedback, or seek an instructor ruling before claiming this as an eligible final model.

AI assistance: Codex was used substantially for method design discussion, Python implementation, experiment automation, analysis, packaging and documentation. The student should review and understand all code and claims before submission. Reused material: supplied dataset, fixed embeddings, parser and evaluator; no extra pretrained model or external training data. The project retains the supplied `data/LICENSE.txt` and `data/NOTICE.txt`.

Before submission, publish the exact version of this repository and confirm that the final report, `methods/`, training scripts, `results/`, and `checkpoints/` are committed. The current `runs/` directory is Git-ignored. Include the commit SHA and public code/checkpoint URLs in the MP2 website form. Then click **Submit on GitHub** and actually create the prepared issue before 2026-10-31 23:59 UTC+8. Use the same account and student ID for updates. The report must be at most 10 pages in its submitted rendering. Later peer review requires running the official scorer on a peer's submitted code and reporting the reproduced score.
