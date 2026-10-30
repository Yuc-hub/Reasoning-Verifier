"""Linear candidate scorer trained only on supplied fixed embeddings."""

from pathlib import Path

import numpy as np


def predict(bank, arrays, checkpoint):
    if checkpoint is None:
        raise ValueError('Embedding model requires --checkpoint')
    with np.load(Path(checkpoint), allow_pickle=False) as saved:
        mean = saved['mean']
        scale = saved['scale']
        weights = saved['weights']
    embeddings = arrays['embeddings']
    if embeddings.shape != (len(bank), 4, len(weights)):
        raise ValueError('Embedding shape does not match checkpoint')
    scores = ((embeddings - mean) / scale) @ weights
    return [int(i) for i in scores.argmax(axis=1)]
