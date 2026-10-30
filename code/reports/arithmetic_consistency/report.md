# Arithmetic consistency feature ablation

## Research question and implementation

Does checking explicitly written simple arithmetic improve the existing
four-candidate group-choice model? This experiment keeps the original 1,044
candidate features, multi-correct group loss, optimizer and hyperparameters,
then appends seven equation features. The first 1,044 training means and scales
were verified equal to the group-choice checkpoint.

`code/methods/arithmetic_consistency.py` normalizes common math notation
(`\\times`, `\\div`, `\\cdot`, `×`, `÷`, boxed results and Markdown bold)
and extracts explicit `number operator number = number` expressions. It
checks addition, subtraction, multiplication and division with exact rational
arithmetic, allowing an absolute difference of at most 0.001 for displayed
rounded decimals. It does **not** parse algebraic expressions, multi-step
equations, variable substitutions or all LaTeX forms. An unparsed expression
does not count as an error. A detected error is a learned feature, never a
hard rejection: official scoring checks the final answer, not every step.

The seven added features are: log checked count, log wrong count, fraction
wrong, any correct equation, last equation correct, any equation result equal
to the parsed final answer, and last equation result equal to the final
answer. Training is in `code/train_arithmetic_consistency.py` and uses only
training inputs and references. Inference reads candidate inputs and the
checkpoint, never references.

## Parameters

| Item | Value |
|---|---|
| Training questions | 4,755 |
| Correct training candidates | 14,353 / 19,020 |
| Candidate features | 1,044 original + 7 arithmetic = 1,051 |
| Training candidates with a parsed simple equation | 15,744 / 19,020 (82.78%) |
| Arithmetic tolerance | Absolute rational difference at most 0.001 |
| Objective | Summed multi-correct group softmax loss + `lambda/2 * ||w||²` |
| L2 coefficient | 25.0 |
| Optimizer | Deterministic full-batch Adam |
| Epochs / learning rate | 300 / 0.02 |
| Adam settings | `beta1=0.9`, `beta2=0.999`, epsilon `1e-8` |
| Training objective | 1800.859 → 751.085 |
| Random seed | Not applicable; no random initialization or shuffling |
| Checkpoint | `code/runs/arithmetic_consistency/model.npz` |
| Checkpoint size | About 13 KiB |
| Checkpoint SHA-256 | `d03a3ee3310f47c212ce1941494c51b98ebe9f14f28a0e1a0bfdf9d7c2caad3f` |

## Official validation and exploratory test results

Environment: Miniconda `mp1`, Python 3.12.14, NumPy 2.5.3.

| Method | Validation | Test |
|---|---:|---:|
| Original group choice | 742 / 798 = 92.98245614035088% | 385 / 467 = 82.44111349036403% |
| + Arithmetic features | 741 / 798 = 92.85714285714286% | 384 / 467 = 82.22698072805139% |

Equation extraction covered 2,649 / 3,192 validation candidates and
1,509 / 1,868 test candidates. It flagged at least one wrong equation in
1,130 validation and 679 test candidates. Relative to the original model,
the new method changed 30 validation selections (1 improved, 2 regressed)
and 18 test selections (2 improved, 3 regressed). Thus the added features
did not improve selection accuracy in this experiment. The test result is
exploratory because earlier test scores were already visible; no code or
parameter was changed after seeing this result.

```bash
cd code
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  conda run -n mp1 python train_arithmetic_consistency.py \
  --output runs/arithmetic_consistency/model.npz
for split in validation test; do
  OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    conda run -n mp1 python predict.py \
    --method methods/arithmetic_consistency.py \
    --checkpoint runs/arithmetic_consistency/model.npz --split "$split" \
    --output "runs/arithmetic_consistency/$split-predictions.json"
  conda run -n mp1 python score_predictions.py --split "$split" \
    --predictions "runs/arithmetic_consistency/$split-predictions.json" \
    --output "runs/arithmetic_consistency/$split-score.json"
done
```

Local fresh-process timing with the `mp1` Python executable and four thread
environment variables: validation prediction 0.33 s + scoring 0.06 s;
test prediction 0.23 s + scoring 0.06 s. The local machine is Apple Silicon
`arm64`; the reference Xeon remains the eligibility machine. `runs/` is
Git-ignored, so a final submission using this model must publish its matching
checkpoint.
