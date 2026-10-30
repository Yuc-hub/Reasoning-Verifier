"""Tournament reranker with a genuinely pair-dependent preference model."""

from pathlib import Path

import numpy as np

from methods.pairwise_comparison import make_inputs
from methods.text_features import FEATURE_NAMES


TEXT_DIM = len(FEATURE_NAMES)
BASE_DIM = TEXT_DIM + 1024
PAIR_DIM = BASE_DIM + 2 * TEXT_DIM
PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


def pair_features(normalized):
    """Return antisymmetric features for the six ordered j<k pairs."""
    emb = normalized[..., TEXT_DIM:]
    norms = np.linalg.norm(emb, axis=-1)
    vectors = []
    for j, k in PAIRS:
        delta = normalized[:, j] - normalized[:, k]
        cosine = np.sum(emb[:, j] * emb[:, k], axis=-1) / np.maximum(
            norms[:, j] * norms[:, k], 1e-8)
        cosine = np.clip(cosine, -1.0, 1.0)
        answer_distance = np.abs(
            normalized[:, j, 3] - normalized[:, k, 3])
        # Symmetric pair context times antisymmetric candidate difference.
        # Unlike plain delta, this generally cannot be reduced to a single
        # fixed score for each candidate across all opponents.
        context = np.column_stack([
            delta[:, :TEXT_DIM] * cosine[:, None],
            delta[:, :TEXT_DIM] * answer_distance[:, None],
        ])
        vectors.append(np.concatenate([delta, context], axis=-1))
    return np.stack(vectors, axis=1)


def preference_margins(bank, arrays, checkpoint):
    with np.load(Path(checkpoint), allow_pickle=False) as saved:
        mean = saved['mean']
        scale = saved['scale']
        weights = saved['weights']
    if len(weights) != PAIR_DIM:
        raise ValueError('Tournament checkpoint dimension mismatch')
    base = make_inputs(bank, arrays)
    normalized = (base - mean) / scale
    return pair_features(normalized) @ weights


def predict(bank, arrays, checkpoint):
    if checkpoint is None:
        raise ValueError('Tournament model requires --checkpoint')
    margins = preference_margins(bank, arrays, checkpoint)
    wins = np.zeros((len(bank), 4), dtype=np.int64)
    strength = np.zeros((len(bank), 4), dtype=np.float64)
    for p, (j, k) in enumerate(PAIRS):
        margin = margins[:, p]
        wins[:, j] += margin >= 0
        wins[:, k] += margin < 0
        strength[:, j] += margin
        strength[:, k] -= margin
    # Win count is primary (win rate = wins / 3). Total margin breaks ties.
    # Remaining exact ties go to the lowest input index.
    choices = []
    for win_row, strength_row in zip(wins, strength):
        best_wins = win_row.max()
        eligible = np.flatnonzero(win_row == best_wins)
        choices.append(int(eligible[np.argmax(strength_row[eligible])]))
    return choices
