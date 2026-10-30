"""Group-choice model without length and final-answer formatting features."""

from pathlib import Path

import numpy as np

from methods.pairwise_comparison import make_inputs
from methods.text_features import FEATURE_NAMES


REMOVED = (
    'text_log_chars', 'text_log_words', 'text_log_lines',
    'has_boxed', 'has_final_answer', 'has_hash_answer',
)
KEEP = [j for j, name in enumerate(FEATURE_NAMES) if name not in REMOVED]
KEEP.extend(range(len(FEATURE_NAMES), len(FEATURE_NAMES) + 1024))


def make_inputs_without_artifacts(bank, arrays):
    return make_inputs(bank, arrays)[..., KEEP]


def predict(bank, arrays, checkpoint):
    if checkpoint is None:
        raise ValueError('Artifact ablation requires --checkpoint')
    with np.load(Path(checkpoint), allow_pickle=False) as saved:
        mean, scale, weights = (saved[k] for k in ('mean', 'scale', 'weights'))
    features = make_inputs_without_artifacts(bank, arrays)
    if features.shape[-1] != len(weights):
        raise ValueError('Feature/checkpoint mismatch')
    scores = ((features - mean) / scale) @ weights
    return [int(i) for i in scores.argmax(axis=1)]
