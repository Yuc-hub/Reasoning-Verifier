"""Group-choice model with candidate numeric-divergence features."""

from fractions import Fraction
from pathlib import Path

import numpy as np

from methods.arithmetic_consistency import _equations
from methods.pairwise_comparison import make_inputs
from methods.text_features import NUMBERS


EXTRA_NAMES = (
    'mean_numeric_prefix', 'max_numeric_prefix',
    'mean_result_prefix', 'max_result_prefix',
    'mean_numeric_jaccard', 'mean_result_jaccard',
    'first_divergent_number_from_question',
    'peer_first_divergent_number_from_question',
)


def _canonical(token):
    try:
        return str(Fraction(token.replace(',', '')))
    except (ValueError, ZeroDivisionError):
        return token


def _prefix_length(first, second):
    count = 0
    for a, b in zip(first, second):
        if a != b:
            break
        count += 1
    return count


def _jaccard(first, second):
    a, b = set(first), set(second)
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def divergence_features(bank):
    groups = []
    for row in bank:
        question_numbers = {_canonical(t) for t in NUMBERS.findall(row['question'])}
        number_sequences = [
            [_canonical(t) for t in NUMBERS.findall(c['text'])]
            for c in row['candidates']
        ]
        result_sequences = [
            [str(value) for _, value in _equations(c['text'])]
            for c in row['candidates']
        ]
        group = []
        for j in range(4):
            comparisons = []
            for k in range(4):
                if j == k:
                    continue
                a, b = number_sequences[j], number_sequences[k]
                ra, rb = result_sequences[j], result_sequences[k]
                common = _prefix_length(a, b)
                result_common = _prefix_length(ra, rb)
                numeric_prefix = common / max(1, min(len(a), len(b)))
                result_prefix = result_common / max(1, min(len(ra), len(rb)))
                own_divergence = float(
                    common < len(a) and a[common] in question_numbers)
                peer_divergence = float(
                    common < len(b) and b[common] in question_numbers)
                comparisons.append((
                    numeric_prefix, result_prefix,
                    _jaccard(a, b), _jaccard(ra, rb),
                    own_divergence, peer_divergence,
                ))
            pair = np.asarray(comparisons)
            group.append([
                pair[:, 0].mean(), pair[:, 0].max(),
                pair[:, 1].mean(), pair[:, 1].max(),
                pair[:, 2].mean(), pair[:, 3].mean(),
                pair[:, 4].mean(), pair[:, 5].mean(),
            ])
        groups.append(group)
    return np.asarray(groups, dtype=np.float64)


def make_divergence_inputs(bank, arrays):
    return np.concatenate([make_inputs(bank, arrays),
                           divergence_features(bank)], axis=-1)


def predict(bank, arrays, checkpoint):
    if checkpoint is None:
        raise ValueError('Divergence model requires --checkpoint')
    with np.load(Path(checkpoint), allow_pickle=False) as saved:
        mean, scale, weights = (saved[key] for key in ('mean', 'scale', 'weights'))
    features = make_divergence_inputs(bank, arrays)
    if features.shape[-1] != len(weights):
        raise ValueError('Divergence feature/checkpoint mismatch')
    scores = ((features - mean) / scale) @ weights
    return [int(i) for i in scores.argmax(axis=1)]
