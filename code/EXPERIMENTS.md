# Method experiments

The official `predict.py --method` argument selects a method file. Keep each
method in `methods/` and write its predictions and scores to a separate folder
under `runs/`. After comparing methods on validation, copy or import the frozen
winner from `submission.py` and record its matching checkpoint. Do not use test
references to select a method.

Each method also has a `reports/<method>/report.md` documenting implementation,
parameters, commands, resource records and results. Add or update that report
whenever a method changes.
For each subsequent method change, run both the official validation and test
prediction/scoring commands in the `mp1` environment and record both results.
Because test results have already been viewed during development, label new
test comparisons exploratory and never use them for further tuning.

## Answer agreement baseline

For each question, parse the four final answers with the official
`common.parse_answer`. Choose a candidate whose parsed answer appears most
often. Ties go to the lowest candidate index. If no answer parses, choose 0.
The method has no training step, checkpoint, random seed, or additional assets.

From `code/`, using the Miniconda `mp1` environment:

```bash
conda run -n mp1 python check_data.py
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  conda run -n mp1 python predict.py --method methods/answer_agreement.py \
  --split validation --output runs/answer_agreement/validation-predictions.json
conda run -n mp1 python score_predictions.py --split validation \
  --predictions runs/answer_agreement/validation-predictions.json \
  --output runs/answer_agreement/validation-score.json
```

For another method, change `--method` and the output folder. Once the method
is frozen, run the same two commands with `--split test` for final scoring.

Initial validation result in `mp1` (Python 3.12.14, NumPy 2.5.3):
688/798 correct, **86.2155388471178%**. The runner reported 0.070 seconds
for prediction; this field excludes some process startup, so it is not the
official end-to-end resource measurement. Validation includes easier groups
than test, so this score should not be treated as a test estimate.

## Text feature model

This model extracts 20 numeric features from each candidate and its question:
answer agreement, answer properties, text length and structure, overlap with
the question, and within-question relative values. It fits L2-regularized
logistic regression on the **training split only**, then ranks the four
candidates by score. Inference uses neither embeddings nor reference answers.
The checkpoint is a small `.npz` file containing feature means, scales and
weights. The method has no randomness; regularization defaults to 1.0.

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  conda run -n mp1 python train_text_features.py \
  --output runs/text_features/model.npz
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  conda run -n mp1 python predict.py --method methods/text_features.py \
  --checkpoint runs/text_features/model.npz --split validation \
  --output runs/text_features/validation-predictions.json
conda run -n mp1 python score_predictions.py --split validation \
  --predictions runs/text_features/validation-predictions.json \
  --output runs/text_features/validation-score.json
```

Initial validation result: 719/798 correct, **90.10025062656642%**.
The runner reported 0.191 seconds for prediction, excluding some startup.
This is 31 more correct questions than the answer agreement baseline on the
same validation split. Test scoring and final method selection remain separate.

## Fixed embedding linear model

See `reports/embedding_linear/report.md` for the implementation, exact
parameters and commands. It uses only the supplied 1024-dimensional candidate
embeddings and a 12 KiB learned checkpoint. Initial validation result:
712/798 correct, **89.22305764411027%**.

## Candidate pairwise comparison model

See `reports/pairwise_comparison/report.md` for the implementation, exact
parameters, limitations and commands. It trains on 8,271 within-question
correct/wrong pairs using 20 text features and 1,024 fixed embedding
dimensions. Initial validation result: 734/798 correct,
**91.97994987468671%**.

## Group-normalized score fusion

See `reports/score_fusion/report.md` for the grid, selected weights, exact
parameters, assets, limitations and commands. The selected pairwise +
embedding fusion scored 738/798 on validation, **92.4812030075188%**.
After freezing those weights, its official test result was 377/467,
**80.72805139186295%**. See the method report for the test commands and
local end-to-end timing.

## Test results across all methods

The unchanged method files and checkpoints were each run through the official
test scorer for descriptive comparison. The first five methods were developed
before their test runs. The tournament method was added after earlier test
results had been seen, so its test result is exploratory. The test bank was
not used to alter weights or code.

| Method | Validation correct / 798 | Test correct / 467 | Test accuracy |
|---|---:|---:|---:|
| Answer agreement | 688 | 323 | 69.16488222698072% |
| Text features | 719 | 350 | 74.94646680942184% |
| Embedding linear | 712 | 360 | 77.08779443254818% |
| Pairwise comparison | 734 | 378 | 80.94218415417559% |
| Score fusion | 738 | 377 | 80.72805139186295% |
| Tournament pairwise (exploratory) | 732 | 376 | 80.51391862955032% |
| Bradley–Terry (exploratory) | 739 | 382 | 81.79871520342613% |
| Group choice (exploratory) | 742 | 385 | 82.44111349036403% |
| Arithmetic consistency (exploratory) | 741 | 384 | 82.22698072805139% |
| Divergence localization (exploratory) | 745 | 385 | 82.44111349036403% |
| Condition coverage (exploratory) | 740 | 386 | 82.65524625267666% |
| Combined reasoning (exploratory) | 739 | 389 | 83.29764453961457% |
| Answer-cluster choice (exploratory) | 745 | 387 | 82.86937901498929% |
| Group choice without surface features (exploratory) | 743 | 383 | 82.01284796573876% |
| Process-chain verifier (exploratory) | 742 | 391 | 83.72591006423983% |
| Combined answer-cluster postprocessing (exploratory) | 744 | 384 | 82.22698072805139% |
| Hard-negative combined, single candidate (exploratory) | 745 | 384 | 82.22698072805139% |
| Hard-negative combined, answer cluster (exploratory) | 743 | 383 | 82.01284796573876% |
| Role-aware process (exploratory) | 743 | 394 | 84.36830835117773% |

The fusion model was selected on validation before test evaluation. Its
four-question validation gain did not carry over to test; the pairwise model
scored one test question higher. These test results are evidence about
generalization, not a basis for further method selection.

## Tournament pairwise adaptation

See `reports/tournament_pairwise/report.md` for its relation to V1-Infer,
implementation, parameters, validation result and limitations. It scored
732/798 on validation, **91.72932330827068%**. This experiment was added
after the earlier methods' test results were seen. At the user's request,
the unchanged checkpoint was then evaluated on test: 376/467,
**80.51391862955032%**. This is an exploratory comparison, not a basis for
development or method selection.

## Bradley–Terry loss ablation

See `reports/bradley_terry/report.md` for the objective, implementation,
controls and commands. The 1,044 features, training pairs and feature scaling
match the existing pairwise ridge method; only the fitting loss changes.
Bradley–Terry scored 739/798 on validation, **92.60651629072682%**, versus
734/798 for pairwise ridge. Its exploratory test result is 382/467,
**81.79871520342613%**. It was added after earlier test results had been
seen; the test result was not used to change the model.

## Local four-candidate group choice

See `reports/group_choice/report.md` for the multi-correct softmax objective,
implementation, parameters, commands and resource records. It uses the same
1,044 candidate features as the pairwise and Bradley–Terry models, but fits
the probability mass of each four-way group on any correct candidate. Official
validation: 742/798, **92.98245614035088%**. Exploratory test: 385/467,
**82.44111349036403%**. The model was not changed after test evaluation.

## Arithmetic consistency feature ablation

See `reports/arithmetic_consistency/report.md` for parsing coverage,
implementation, exact parameters, commands, resource records and failure
analysis. It adds seven simple-equation features to the group-choice model,
with the original 1,044 features and training configuration unchanged.
Official validation: 741/798, **92.85714285714286%**. Exploratory test:
384/467, **82.22698072805139%**. Both are one correct question below the
original group-choice model; the arithmetic features did not help here.

## Candidate divergence localization feature ablation

See `reports/divergence_localization/report.md` for the eight group-comparison
features, limitations, commands and resource records. It uses numeric-token
and equation-result sequence prefixes to locate where candidates first
diverge. Official validation: 745/798, **93.35839598997494%**. Exploratory
test: 385/467, **82.44111349036403%**. Test accuracy equals the original
group-choice model despite a three-question validation improvement.

## Question-condition coverage feature ablation

See `reports/condition_coverage/report.md` for the ten condition features,
parser coverage, limitations, commands and resource records. It distinguishes
question numbers merely mentioned from numbers used in a recognized
calculation, and includes unit, time and constraint cues. Official validation:
740/798, **92.73182957393483%**. Exploratory test: 386/467,
**82.65524625267666%**. The change is small and has opposite signs across
validation and test.

## Combined reasoning features

See `reports/combined_reasoning/report.md` for all four feature families,
implementation, commands and resource records. It appends arithmetic,
condition-coverage, candidate-divergence and final-answer/process-link
features to the original group-choice model. Official validation: 739/798,
**92.60651629072682%**. Exploratory test: 389/467,
**83.29764453961457%**. The validation decrease and test increase point in
opposite directions; test was not used to change the method.

## Answer-cluster score aggregation

See `reports/answer_cluster_choice/report.md`. With the original group-choice
checkpoint unchanged, this rule sums candidate probability within each
parsed-final-answer cluster. Official validation: 745/798,
**93.35839598997494%**. Exploratory test: 387/467,
**82.86937901498929%**.

## Length and answer-format feature ablation

See `reports/group_choice_no_artifacts/report.md`. This retrains group choice
after removing six explicit length and final-answer formatting features.
Official validation: 743/798, **93.10776942355889%**. Exploratory test:
383/467, **82.01284796573876%**. The test change is small and opposite to
validation; embeddings can still carry surface signals.

## Out-of-fold error analysis and process verifier

See `reports/oof_error_analysis/report.md` and
`reports/process_verifier/report.md`. Five-fold train-only OOF analysis of
combined reasoning found 405 errors among 2,114 questions with divergent
candidate answers. Nine calculation-chain features were then added to that
model. Under the same folds, hard-group correct rose from 1,709 to 1,731.
Official validation rose from 739/798 to 742/798,
**92.98245614035088%**. Exploratory test rose from 389/467 to 391/467,
**83.72591006423983%**. The process parser verifies local numeric steps,
not full semantic correctness; test was not used to change the method.

## Combined answer-cluster rule, then hard-negative training

See `reports/answer_cluster_combined/report.md` and
`reports/hard_negative_combined/report.md`. The experiments were run in the
requested order. Applying the fixed answer-cluster rule to the unchanged
combined checkpoint raised validation from 739 to 744 / 798 but lowered
exploratory test from 389 to 384 / 467. Training with a predeclared
high-scoring-wrong auxiliary loss raised validation to 745 / 798 with
single-candidate inference, while exploratory test was 384 / 467. Adding
cluster postprocessing to that checkpoint gave 743 / 798 validation and
383 / 467 test. The auxiliary loss improved five-fold train-only OOF hard
accuracy by 16 / 2,114 but did not improve the exploratory test score.

## Role-aware process verifier

See `reports/role_aware_process/report.md`. Eight target-role and
quantity-source features were added to the process verifier after inspecting
train-only OOF errors. Official validation rose from 742/798 to 743/798,
**93.10776942355889%**. Exploratory test rose from 391/467 to 394/467,
**84.36830835117773%**. The same five-fold OOF hard-group result fell by one
question (1,731 to 1,730 / 2,114), so the small external-split gains are
not evidence of robust improvement. No code or parameter was changed after
the test run.
