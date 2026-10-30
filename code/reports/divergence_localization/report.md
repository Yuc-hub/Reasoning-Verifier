# Candidate divergence localization feature ablation

## Research question and implementation

Can a group-choice model use where the four candidate traces first diverge
numerically? This method retains the original 1,044 candidate features,
multi-correct group softmax objective, optimizer and hyperparameters, then
adds eight group-comparison features. The original 1,044 feature means and
scales were verified equal to the group-choice checkpoint.

`code/methods/divergence_localization.py` extracts two ordered sequences from
each candidate: all numeric tokens (canonicalized as rational values) and
the stated results of explicit simple equations. For each of the other three
candidates, it finds the length of the shared prefix in each sequence,
normalizes by the shorter sequence length, computes set overlap, and checks
whether the first divergent numeric token comes from the question. The
candidate's eight features summarize these three pair comparisons: mean and
maximum numeric-prefix ratio, mean and maximum equation-result-prefix ratio,
mean numeric and result Jaccard overlap, and fractions of comparisons where
the candidate's or peer's first divergent token appears in the question.

These are **surface proxies for divergence**, not a semantic identification
of the first incorrect reasoning step. Differences in wording, number order,
or equation formatting may move the apparent divergence. The equation
extractor is shared with the arithmetic-consistency experiment; this model
uses the extracted result sequence, not the equation-correctness flags.
Training is in `code/train_divergence_localization.py` and uses training
inputs and references only. Inference never reads references.

## Parameters

| Item | Value |
|---|---|
| Training questions | 4,755 |
| Correct training candidates | 14,353 / 19,020 |
| Candidate features | 1,044 original + 8 divergence = 1,052 |
| Comparison peers per candidate | 3 |
| Objective | Summed multi-correct group softmax loss + `lambda/2 * ||w||²` |
| L2 coefficient | 25.0 |
| Optimizer | Deterministic full-batch Adam |
| Epochs / learning rate | 300 / 0.02 |
| Adam settings | `beta1=0.9`, `beta2=0.999`, epsilon `1e-8` |
| Training objective | 1800.859 → 744.425 |
| Random seed | Not applicable; no random initialization or shuffling |
| Checkpoint | `code/runs/divergence_localization/model.npz` |
| Checkpoint size | About 13 KiB |
| Checkpoint SHA-256 | `3cb981d4fa00405dd1a1f9021d3030492cc2bb71254e1b1d9a2f6c0e60c402d4` |

## Official validation and exploratory test results

Environment: Miniconda `mp1`, Python 3.12.14, NumPy 2.5.3.

| Method | Validation | Test |
|---|---:|---:|
| Original group choice | 742 / 798 = 92.98245614035088% | 385 / 467 = 82.44111349036403% |
| + Divergence localization | 745 / 798 = 93.35839598997494% | 385 / 467 = 82.44111349036403% |

Relative to the original model, 86 validation selections changed: five
previously wrong choices became correct and two previously correct choices
became wrong. On test, 36 selections changed: eight improved and eight
regressed. Thus the validation gain did not produce a net test gain. The
test result is exploratory because earlier test scores were already visible;
no code or parameter was changed after seeing this result.

```bash
cd code
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  conda run -n mp1 python train_divergence_localization.py \
  --output runs/divergence_localization/model.npz
for split in validation test; do
  OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    conda run -n mp1 python predict.py \
    --method methods/divergence_localization.py \
    --checkpoint runs/divergence_localization/model.npz --split "$split" \
    --output "runs/divergence_localization/$split-predictions.json"
  conda run -n mp1 python score_predictions.py --split "$split" \
    --predictions "runs/divergence_localization/$split-predictions.json" \
    --output "runs/divergence_localization/$split-score.json"
done
```

Local fresh-process timing with the `mp1` Python executable and four thread
environment variables: validation prediction 0.40 s + scoring 0.06 s;
test prediction 0.28 s + scoring 0.05 s. The local machine is Apple Silicon
`arm64`; the reference Xeon remains the eligibility machine. `runs/` is
Git-ignored, so a final submission using this model must publish its matching
checkpoint.
