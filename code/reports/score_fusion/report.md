# Group-normalized score fusion

## Idea and implementation

Use the already-trained pairwise, text-feature and embedding models to compute
four scores for each question. For each model separately, subtract that
question's mean score and divide by its root-mean-square centered score, with
a floor of `1e-8` on the denominator. This makes score scales comparable
within a question. The fused score is a weighted sum of the normalized scores;
the highest-scoring candidate is selected.

`code/select_score_fusion.py` evaluates a **prespecified grid of 10 weight
combinations on validation only**. The grid always gives the pairwise model
weight 1.0; text and embedding weights are drawn from a small set shown in the
script and in `code/runs/score_fusion/validation-grid.json`. The first
maximum-scoring grid entry wins ties. The selector copies the necessary frozen
base checkpoints and writes `config.json`; models with zero selected weight
are omitted from the final bundle. `code/methods/score_fusion.py` loads that
bundle and does not read references during inference.

The selected weights are **pairwise 1.0, text 0.0, embedding 0.25**. Although
the separate text model has zero weight, the pairwise model itself uses 20
text features alongside embeddings.

## Parameters and assets

| Item | Value |
|---|---|
| Candidate score normalization | Center and divide by within-question RMS; denominator floor 1e-8 |
| Grid size | 10 prespecified combinations |
| Selection split | Validation, 798 questions |
| Selected weights | Pairwise 1.0; text 0.0; embedding 0.25 |
| Prediction | Argmax of four fused scores; lowest index wins ties |
| Random seed | Not applicable; deterministic |
| Checkpoint directory | `code/runs/score_fusion/model_bundle` |
| Inference assets | `pairwise.npz`, `embedding.npz`, `config.json` |
| Bundle size | About 26 KiB of files; well below 16 MiB |
| `pairwise.npz` SHA-256 | `040975eb2c7ab71d59d303b1ab18a7e1e752a905c1d29aae6b7279fcb29a5457` |
| `embedding.npz` SHA-256 | `e449b826d89dece073447aa21a2bcd6c64f06711b8c7997a523d3f78f0d1e05a` |
| `config.json` SHA-256 | `970cb0940ca591c798da67a3284f98722cf8216c463eacd5da25b2a2f7422c79` |

The base model training parameters are in their respective reports. All base
model weights were fitted on training only; validation selected the fusion
weights. No test references were used.

## Validation result

Environment: Miniconda `mp1`, Python 3.12.14, NumPy 2.5.3.
Official validation scorer: **738 / 798 = 92.4812030075188%**.
The pairwise model alone scored 734 / 798. The best fusion adds four correct
questions on this validation bank. Adding the separate text score alone did
not improve on 734 / 798. A grid entry with text 0.25 and embedding 0.25 also
scored 738 / 798; the simpler earlier entry was selected by the tie rule.

The prediction runner reported 0.234 seconds and the scorer 0.019 seconds.
These fields exclude some process startup and are not the required complete
evaluation measurement.

```bash
cd code
# First train the three base models using their report commands.
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  conda run -n mp1 python select_score_fusion.py
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  conda run -n mp1 python predict.py --method methods/score_fusion.py \
  --checkpoint runs/score_fusion/model_bundle --split validation \
  --output runs/score_fusion/validation-predictions.json
conda run -n mp1 python score_predictions.py --split validation \
  --predictions runs/score_fusion/validation-predictions.json \
  --output runs/score_fusion/validation-score.json
```

The 4-question gain is small and was selected on validation; it may not
generalize. The final method should be frozen before the single test run.
`runs/` is Git-ignored, so the matching bundle must be published with any
final submission that uses this fusion.

## Frozen test evaluation

The selected weights and three bundle file hashes above were checked before
the test run. No method or weight was changed after seeing the test result.
The official test scorer reported **377 / 467 = 80.72805139186295%**.
The saved outputs are `code/runs/score_fusion/test-predictions.json` and
`code/runs/score_fusion/test-score.json`.

```bash
cd code
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  conda run -n mp1 python predict.py --method methods/score_fusion.py \
  --checkpoint runs/score_fusion/model_bundle --split test \
  --output runs/score_fusion/test-predictions.json
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  conda run -n mp1 python score_predictions.py --split test \
  --predictions runs/score_fusion/test-predictions.json \
  --output runs/score_fusion/test-score.json
```

Local hardware reports `arm64` (Apple Silicon). With the `mp1` Python
executable and four thread environment variables, `/usr/bin/time -p` measured
0.39 seconds for the fresh prediction process and 0.07 seconds for the
independent scoring process, 0.46 seconds combined. This is local evidence;
the course's Xeon Platinum 8457C remains the eligibility reference machine.
The test bank deliberately contains at least two distinct valid candidate
answers per question, unlike many easier validation questions, so the two
split accuracies are not directly comparable as equal-difficulty estimates.
