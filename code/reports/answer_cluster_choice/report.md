# Answer-cluster probability aggregation

## Method and rationale

This experiment uses the **unchanged** group-choice checkpoint. Its original
training loss rewards total probability assigned to any correct candidate,
but its original inference rule selects the single highest-scoring candidate.
Here, candidates with the same parsed final numeric answer form a cluster.
For each cluster, sum `exp(score_j - max_score)` over its members and select
the cluster with the largest sum. The softmax denominator is shared across
clusters and therefore does not affect their order. Within the winning
cluster, choose its highest-scoring candidate. Equivalent fractions and
decimals share a cluster through `common.parse_answer`. Unparseable answers
are kept as separate singleton clusters; they cannot establish agreement.

This is a frozen-checkpoint inference ablation: no training, parameter search
or validation-selected threshold is introduced. Implementation:
`code/methods/answer_cluster_choice.py`.

## Parameters and assets

| Item | Value |
|---|---|
| Base model | Group choice, 1,044 features |
| Cluster key | Canonical parsed final numeric answer |
| Cluster score | Sum of candidate exponentiated centered scores |
| Within-cluster choice | Highest candidate score |
| Tie break | Highest member score, then lowest index |
| Training | No new training |
| Random seed | Not applicable |
| Checkpoint | Existing `code/runs/group_choice/model.npz` |
| Checkpoint SHA-256 | `56fb8d564f7296f37415184c00639b6b9a1a89e33e82a26fe08e5ba967af28a5` |
| Additional inference assets | None |

## Official validation and exploratory test results

Environment: Miniconda `mp1`, Python 3.12.14, NumPy 2.5.3.

| Rule | Validation | Test |
|---|---:|---:|
| Single highest candidate | 742 / 798 = 92.98245614035088% | 385 / 467 = 82.44111349036403% |
| Highest answer-cluster mass | 745 / 798 = 93.35839598997494% | 387 / 467 = 82.86937901498929% |

Compared with the original rule, validation changed 11 selections: seven
improved and four regressed. Test changed 17: nine improved and seven
regressed. The two- and three-question gains are small. The test result is
exploratory because earlier test scores had already been seen; no code or
weight was changed after this result.

```bash
cd code
for split in validation test; do
  OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    conda run -n mp1 python predict.py \
    --method methods/answer_cluster_choice.py \
    --checkpoint runs/group_choice/model.npz --split "$split" \
    --output "runs/answer_cluster_choice/$split-predictions.json"
  conda run -n mp1 python score_predictions.py --split "$split" \
    --predictions "runs/answer_cluster_choice/$split-predictions.json" \
    --output "runs/answer_cluster_choice/$split-score.json"
done
```

Local fresh-process timing with the `mp1` Python executable: validation
prediction 0.38 s + scoring 0.07 s; test prediction 0.19 s + scoring 0.06 s.
The local machine is Apple Silicon `arm64`; the reference Xeon remains the
eligibility machine. The checkpoint is Git-ignored under `runs/` and must be
published for a submission using this rule.
