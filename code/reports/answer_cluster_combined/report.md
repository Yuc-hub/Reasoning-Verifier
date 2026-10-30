# Answer-cluster postprocessing on combined reasoning

## Method and order

This was the first experiment. It kept the combined-reasoning checkpoint and
all 1,073 features unchanged. Candidates with the same canonical parsed
final numeric answer form a cluster. Each cluster receives the sum of its
members' `exp(score_j - max_score)` values. The shared softmax denominator
does not affect cluster ranking. The highest-scoring member of the winning
cluster is returned. Unparseable answers remain separate singleton clusters.

Implementation: `code/methods/answer_cluster_combined.py`. This is a fixed
postprocessing rule with no new training, threshold search or inference
assets. It reuses `code/runs/combined_reasoning/model.npz`, SHA-256
`fdcd602d65d2e39fbe9b67c59ce5d3f233e5c415047cfa0b7ae16d6ce0f4bf6e`.
Ties use highest member score, then lowest candidate index.

## Official validation and exploratory test results

Environment: Miniconda `mp1`, Python 3.12.14, NumPy 2.5.3.

| Rule | Validation | Test |
|---|---:|---:|
| Combined, highest candidate | 739 / 798 = 92.60651629072682% | 389 / 467 = 83.29764453961457% |
| Combined, highest answer cluster | 744 / 798 = 93.23308270676692% | 384 / 467 = 82.22698072805139% |

Validation changed 12 selections relative to the base rule: eight became
correct and three became wrong. Test changed 17: six became correct and
eleven became wrong. The validation gain did not transfer to test. The test
score is exploratory because earlier test results had already been seen.
The cluster rule was not changed after either result.

```bash
cd code
for split in validation test; do
  OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    conda run -n mp1 python predict.py \
    --method methods/answer_cluster_combined.py \
    --checkpoint runs/combined_reasoning/model.npz --split "$split" \
    --output "runs/answer_cluster_combined/$split-predictions.json"
  conda run -n mp1 python score_predictions.py --split "$split" \
    --predictions "runs/answer_cluster_combined/$split-predictions.json" \
    --output "runs/answer_cluster_combined/$split-score.json"
done
```

Local fresh-process timing: validation prediction 1.35 s plus scoring 0.07 s;
test prediction 0.87 s plus scoring 0.06 s. The local machine is Apple
Silicon `arm64`; the reference Xeon remains the eligibility machine.
The reused checkpoint is Git-ignored under `runs/` and must be published
for a submission using this rule.
