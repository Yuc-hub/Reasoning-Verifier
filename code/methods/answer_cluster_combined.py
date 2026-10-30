"""Aggregate combined-model scores by parsed final-answer cluster."""

from pathlib import Path

import numpy as np

from common import parse_answer
from methods.combined_reasoning import make_combined_inputs


def choose_clusters(bank, scores):
    choices = []
    for row, score_row in zip(bank, scores):
        shifted = score_row - score_row.max()
        masses = np.exp(shifted)
        clusters = {}
        for j, candidate in enumerate(row['candidates']):
            answer = parse_answer(candidate['text'])
            key = answer if answer is not None else ('unparseable', j)
            clusters.setdefault(key, []).append(j)
        winner = max(clusters.values(), key=lambda indices: (
            sum(masses[j] for j in indices),
            max(score_row[j] for j in indices),
            -min(indices),
        ))
        choices.append(int(max(winner, key=lambda j: (score_row[j], -j))))
    return choices


def predict(bank, arrays, checkpoint):
    if checkpoint is None:
        raise ValueError('Combined cluster choice requires --checkpoint')
    with np.load(Path(checkpoint), allow_pickle=False) as saved:
        mean, scale, weights = (saved[key] for key in ('mean', 'scale', 'weights'))
    features = make_combined_inputs(bank, arrays)
    if features.shape[-1] != len(weights):
        raise ValueError('Feature/checkpoint mismatch')
    scores = ((features - mean) / scale) @ weights
    return choose_clusters(bank, scores)
