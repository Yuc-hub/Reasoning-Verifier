"""Group-choice reranker with all four lightweight reasoning feature families."""

from fractions import Fraction
from pathlib import Path

import numpy as np

from common import parse_answer
from methods.arithmetic_consistency import _equations, arithmetic_features
from methods.condition_coverage import condition_features
from methods.divergence_localization import divergence_features
from methods.pairwise_comparison import make_inputs


LINK_NAMES = (
    'last_three_result_match_fraction',
    'last_result_conflicts_with_final',
    'last_verified_result_matches_final',
    'verified_final_result_exists',
)


def final_link_features(bank):
    groups = []
    for row in bank:
        group = []
        for candidate in row['candidates']:
            answer = parse_answer(candidate['text'])
            final = Fraction(answer) if answer is not None else None
            equations = _equations(candidate['text'])
            tail = equations[-3:]
            verified = [result for valid, result in equations if valid]
            match_fraction = (
                sum(result == final for _, result in tail) / len(tail)
                if final is not None and tail else 0.0
            )
            last_conflicts = float(
                final is not None and bool(equations)
                and equations[-1][1] != final
            )
            last_verified_match = float(
                final is not None and bool(verified)
                and verified[-1] == final
            )
            verified_final_exists = float(
                final is not None and final in verified
            )
            group.append([
                match_fraction, last_conflicts,
                last_verified_match, verified_final_exists,
            ])
        groups.append(group)
    return np.asarray(groups, dtype=np.float64)


def make_combined_inputs(bank, arrays):
    return np.concatenate([
        make_inputs(bank, arrays),
        arithmetic_features(bank),
        condition_features(bank),
        divergence_features(bank),
        final_link_features(bank),
    ], axis=-1)


def predict(bank, arrays, checkpoint):
    if checkpoint is None:
        raise ValueError('Combined reasoning model requires --checkpoint')
    with np.load(Path(checkpoint), allow_pickle=False) as saved:
        mean, scale, weights = (saved[key] for key in ('mean', 'scale', 'weights'))
    features = make_combined_inputs(bank, arrays)
    if features.shape[-1] != len(weights):
        raise ValueError('Combined feature/checkpoint mismatch')
    scores = ((features - mean) / scale) @ weights
    return [int(i) for i in scores.argmax(axis=1)]
