"""Pairwise-trained reranker using both text features and fixed embeddings."""

from pathlib import Path

import numpy as np

from methods.text_features import FEATURE_NAMES, make_features


def make_inputs(bank, arrays):
    text = make_features(bank)
    embeddings = arrays['embeddings'].astype(np.float64)
    if embeddings.shape != (len(bank), 4, 1024):
        raise ValueError('Expected [N, 4, 1024] embeddings')
    return np.concatenate([text, embeddings], axis=-1)


def predict(bank, arrays, checkpoint):
    if checkpoint is None:
        raise ValueError('Pairwise comparison model requires --checkpoint')
    with np.load(Path(checkpoint), allow_pickle=False) as saved:
        mean = saved['mean']
        scale = saved['scale']
        weights = saved['weights']
    features = make_inputs(bank, arrays)
    if features.shape[-1] != len(weights) or len(weights) != len(FEATURE_NAMES) + 1024:
        raise ValueError('Checkpoint feature count does not match method')
    scores = ((features - mean) / scale) @ weights
    return [int(i) for i in scores.argmax(axis=1)]
