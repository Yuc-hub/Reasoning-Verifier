"""Group-choice model augmented with conservative arithmetic-equation features."""

from fractions import Fraction
from pathlib import Path
import re

import numpy as np

from common import parse_answer
from methods.pairwise_comparison import make_inputs


NUMBER = r'[+-]?\d[\d,]*(?:\.\d+)?'
EQUATION = re.compile(
    rf'(?<![\w.])\$?({NUMBER})\s*([+*/×÷\-])\s*\$?({NUMBER})'
    rf'\s*=\s*\$?({NUMBER})(?![\w.])'
)
EXTRA_NAMES = (
    'log_checked_equations', 'log_wrong_equations', 'wrong_fraction',
    'has_correct_equation', 'last_equation_correct',
    'any_equation_result_matches_final', 'last_result_matches_final',
)


def _equations(text):
    normalized = (text.replace('\\times', '×').replace('\\cdot', '×')
                  .replace('\\div', '÷').replace('−', '-')
                  .replace('\\boxed{', ' ').replace('}', ' ')
                  .replace('**', ''))
    results = []
    for match in EQUATION.finditer(normalized):
        left, operator, right, stated = match.groups()
        try:
            a = Fraction(left.replace(',', ''))
            b = Fraction(right.replace(',', ''))
            c = Fraction(stated.replace(',', ''))
            if operator == '+':
                actual = a + b
            elif operator == '-':
                actual = a - b
            elif operator in ('*', '×'):
                actual = a * b
            else:
                if b == 0:
                    continue
                actual = a / b
            # Exact arithmetic, with a small absolute tolerance for displayed
            # rounded decimal calculations. This is only a model feature.
            correct = abs(actual - c) <= Fraction(1, 1000)
            results.append((correct, c))
        except (ValueError, ZeroDivisionError):
            continue
    return results


def arithmetic_features(bank):
    groups = []
    for row in bank:
        candidates = []
        for candidate in row['candidates']:
            equations = _equations(candidate['text'])
            n = len(equations)
            wrong = sum(not correct for correct, _ in equations)
            answer = parse_answer(candidate['text'])
            final = Fraction(answer) if answer is not None else None
            candidates.append([
                np.log1p(n), np.log1p(wrong), wrong / max(n, 1),
                float(any(correct for correct, _ in equations)),
                float(equations[-1][0]) if n else 0.0,
                float(final is not None and any(
                    value == final for _, value in equations)),
                float(final is not None and n > 0 and equations[-1][1] == final),
            ])
        groups.append(candidates)
    return np.asarray(groups, dtype=np.float64)


def make_arithmetic_inputs(bank, arrays):
    return np.concatenate([make_inputs(bank, arrays),
                           arithmetic_features(bank)], axis=-1)


def predict(bank, arrays, checkpoint):
    if checkpoint is None:
        raise ValueError('Arithmetic model requires --checkpoint')
    with np.load(Path(checkpoint), allow_pickle=False) as saved:
        mean, scale, weights = (saved[k] for k in ('mean', 'scale', 'weights'))
    features = make_arithmetic_inputs(bank, arrays)
    if features.shape[-1] != len(weights):
        raise ValueError('Arithmetic feature/checkpoint mismatch')
    scores = ((features - mean) / scale) @ weights
    return [int(i) for i in scores.argmax(axis=1)]
