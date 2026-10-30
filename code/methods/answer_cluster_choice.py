"""Select an answer cluster by summed group-choice probability."""

from pathlib import Path

import numpy as np

from common import parse_answer
from methods.pairwise_comparison import make_inputs


def predict(bank, arrays, checkpoint):
    if checkpoint is None:
        raise ValueError('Answer-cluster choice requires --checkpoint')
    with np.load(Path(checkpoint), allow_pickle=False) as saved:
        mean, scale, weights = (saved[k] for k in ('mean', 'scale', 'weights'))
    features = make_inputs(bank, arrays)
    if features.shape[-1] != len(weights):
        raise ValueError('Feature/checkpoint mismatch')
    scores = ((features - mean) / scale) @ weights
    choices = []
    for row, score_row in zip(bank, scores):
        shifted = score_row - score_row.max()
        probability = np.exp(shifted)
        clusters = {}
        for j, candidate in enumerate(row['candidates']):
            answer = parse_answer(candidate['text'])
            # Unparseable answers cannot establish agreement with one another.
            key = answer if answer is not None else ('unparseable', j)
            clusters.setdefault(key, []).append(j)
        winner = max(clusters.values(), key=lambda indices: (
            sum(probability[j] for j in indices),
            max(score_row[j] for j in indices),
            -min(indices),
        ))
        choices.append(int(max(winner, key=lambda j: (score_row[j], -j))))
    return choices
