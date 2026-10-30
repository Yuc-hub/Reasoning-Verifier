"""Fuse frozen candidate scores from three already-trained methods."""

import json
from pathlib import Path

import numpy as np

from methods.text_features import make_features
from methods.pairwise_comparison import make_inputs


def _linear_scores(features, path):
    with np.load(path, allow_pickle=False) as saved:
        mean, scale, weights = (saved[key] for key in ('mean', 'scale', 'weights'))
    if features.shape[-1] != len(weights):
        raise ValueError(f'Feature/checkpoint mismatch: {path}')
    return ((features - mean) / scale) @ weights


def component_scores(bank, arrays, checkpoint, active=None):
    root = Path(checkpoint)
    text = make_features(bank)
    embeddings = arrays['embeddings']
    if embeddings.shape != (len(bank), 4, 1024):
        raise ValueError('Expected [N, 4, 1024] embeddings')
    active = set(active or ('pairwise', 'text', 'embedding'))
    result = {}
    if 'pairwise' in active:
        pair = np.concatenate([text, embeddings.astype(np.float64)], axis=-1)
        result['pairwise'] = _linear_scores(pair, root / 'pairwise.npz')
    if 'text' in active:
        result['text'] = _linear_scores(text, root / 'text.npz')
    if 'embedding' in active:
        result['embedding'] = _linear_scores(embeddings, root / 'embedding.npz')
    return result


def normalize_group(scores):
    centered = scores - scores.mean(axis=1, keepdims=True)
    scale = np.sqrt(np.mean(centered ** 2, axis=1, keepdims=True))
    return centered / np.maximum(scale, 1e-8)


def fused_scores(components, weights):
    return sum(weights[name] * normalize_group(components[name])
               for name in ('pairwise', 'text', 'embedding') if weights[name] != 0)


def predict(bank, arrays, checkpoint):
    if checkpoint is None:
        raise ValueError('Score fusion requires --checkpoint directory')
    root = Path(checkpoint)
    config = json.loads((root / 'config.json').read_text())
    weights = config['weights']
    if set(weights) != {'pairwise', 'text', 'embedding'}:
        raise ValueError('Fusion config must contain exactly three weights')
    active = [name for name, weight in weights.items() if weight != 0]
    if not active:
        raise ValueError('At least one fusion weight must be nonzero')
    scores = fused_scores(component_scores(bank, arrays, root, active), weights)
    return [int(i) for i in scores.argmax(axis=1)]
