# Selecting Reliable LLM Reasoning: Role-Aware Process Verification

DASE7506 MP2, Fall 2026. **Submission method:** `code/submission.py`, checkpoint `code/checkpoints/role_aware_process/model.npz` (SHA-256 `680c173df6c42dcbb02295d24acf6b38b1b0d8e3b5003d5788bce72885d6cb4e`). This Markdown report is intended for a rendering of at most 10 pages. The author must fill in their name, student ID, final repository commit and external full-process timing before submission.

## Question and hypothesis

Can shallow checks of a candidate's intermediate calculations and its alignment with the *quantity asked for* improve selection beyond voting, text statistics and a learned embedding score? My hypothesis was that a candidate can perform valid arithmetic while answering the wrong target quantity, so explicit target and source cues may help distinguish it.

The task provides 4,755 training, 798 validation and 467 test questions, each with four candidate traces and 1,024-dimensional supplied embeddings. A candidate is correct if the official parser finds a final numeric answer equal to the reference. More than one candidate may be correct. The score is the fraction of groups where the selected candidate is correct.

## Method

The base process verifier combines four families of candidate evidence: (1) final-answer agreement and text structure; (2) the supplied whole-trace embeddings; (3) arithmetic consistency and links between verified steps and the final answer; and (4) question-condition coverage and candidate divergence. The original 1,082 features are constructed by `methods/process_verifier.py` and its dependencies. The final ablation adds eight fixed features in `methods/role_aware_process.py`: target keyword recall and overlap near the conclusion, target-role mentions, a narrow role-conflict flag, use of question percentages and currency amounts in verified equations, missing-source fraction, and a target-unit mention flag. These are string and arithmetic proxies. They do not establish that the whole reasoning is logically valid.

Each training group has a 4-by-1,090 feature matrix. Per-feature means and standard deviations are fitted on the *training inputs only*. A linear score is computed for each candidate. For group (g), the model maximizes the softmax probability mass assigned to all correct candidates: (L_g=-\log[\sum_{i\in C_g}\exp(s_{gi})/\sum_j\exp(s_{gj})]). The total objective adds (25\|w\|_2^2/2). Training uses full-batch Adam for 300 epochs, learning rate 0.02, betas 0.9/0.999, epsilon 1e-8. Initialization is zero and there is no data shuffling; the final fit has no random seed. The checkpoint stores float32 means, scales and weights, 13,080 raw array bytes. At inference, `submission.py` selects the largest of four scores, with NumPy's first-index tie rule. It never reads references, IDs for lookup, or any other asset. There are no API calls or generated candidates.

## Experiments and ablation

All entries below use the supplied splits and the independent official scorer. Detailed implementation, commands and per-method results are in `EXPERIMENTS.md` and `reports/<method>/report.md`.

| Method | Validation correct / 798 | Test correct / 467 |
|---|---:|---:|
| Answer-agreement vote | 688 (86.22%) | 323 (69.16%) |
| Text-feature logistic model | 719 (90.10%) | 350 (74.95%) |
| Supplied-embedding linear model | 712 (89.22%) | 360 (77.09%) |
| Pairwise candidate model | 734 (91.98%) | 378 (80.94%) |
| Process verifier, without target/source features | 742 (92.98%) | 391 (83.73%) |
| **Process verifier + target/source features** | **743 (93.11%)** | **394 (84.37%)** |

The target/source feature ablation changes 30 validation selections, giving a net improvement of one question. It changes nine test selections, with a net improvement of three. A separate five-fold train-only out-of-fold check on 2,114 hard groups found 1,730 correct for the final model and 1,731 for its process-verifier parent. That diagnostic result does not support a reliable generalization gain. The small validation and test gains are vulnerable to sampling variation, and the test result is exploratory for the reason below.

## Failure analysis and limits

Local arithmetic checks find explicit elementary equations and track some reused numeric results. They miss implicit calculations, multi-step dependencies and equivalent mathematical forms. A correct solution may use a synonym or a percentage as a fraction rather than the recognized decimal factor, causing a false missing-condition signal. Conversely, a candidate can mention the requested unit or a source number without using it correctly. The target-role conflict covers only two narrow cases, such as answering original price when asked for amount paid. These features are interpretable but do not constitute a full verifier.

The 1,090-feature model is more expensive than a simple vote because it parses text, constructs candidate comparisons and multiplies the supplied embeddings. On local Apple Silicon arm64 with four thread variables, an external `/usr/bin/time -p` around a fresh prediction plus independent scoring process measured 3.24 s real, including startup. The reference Xeon 30 s limit still requires independent verification. The asset is 13,830 bytes as an `.npz` file, well below 16 MiB.

## Protocol and reproducibility

Training labels use only supplied training references. Validation results guided development. Earlier test results were visible before this feature set was built, and the user selected this model partly on its test score. Thus the test score is **not an untouched holdout**, and using it for final model selection conflicts with the assignment rule that test references are for final scoring and reproduction only. This result should not be described as a clean, compliant model-selection estimate. A strictly compliant final submission needs an independently validation-selected frozen version or an instructor ruling. The model is frozen as recorded here; `SUBMISSION.md` provides exact installation, training and evaluation commands and the official score JSON paths.

AI and reuse disclosure: Codex contributed substantially to method discussion, implementation, analysis, experiment running and documentation. The supplied candidate data, embeddings, parser and evaluator were reused; no external dataset, additional pretrained weights or inference API were used. The student is responsible for checking the code and claims before submitting.
