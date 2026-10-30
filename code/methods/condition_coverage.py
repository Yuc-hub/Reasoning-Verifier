"""Group choice with question-condition coverage and calculation-use features."""

from fractions import Fraction
from pathlib import Path
import re

import numpy as np

from methods.pairwise_comparison import make_inputs


NUMBER = re.compile(r'\d[\d,]*(?:\.\d+)?(?:/\d+)?')
WORD = re.compile(r"[A-Za-z]+")
MATH_GAP = re.compile(r'^[ \t$\\(){}]*[+\-*/×÷=][ \t$\\(){}]*$')
MATH_OPERATOR = re.compile(r'[+\-*/×÷=]')
UNIT_GROUPS = {
    'money': ('dollar', 'dollars', 'cent', 'cents', 'usd'),
    'time': ('second', 'seconds', 'minute', 'minutes', 'hour', 'hours',
             'day', 'days', 'week', 'weeks', 'month', 'months',
             'year', 'years', 'am', 'pm'),
    'distance': ('mile', 'miles', 'km', 'kilometer', 'kilometers',
                 'meter', 'meters', 'feet', 'foot', 'inch', 'inches'),
    'mass': ('ounce', 'ounces', 'pound', 'pounds', 'gram', 'grams',
             'kilogram', 'kilograms', 'kg', 'oz', 'lb', 'lbs'),
    'volume': ('gallon', 'gallons', 'liter', 'liters', 'litre', 'litres'),
}
CONSTRAINTS = {
    'at_least': ('at least', 'minimum'),
    'at_most': ('at most', 'maximum', 'no more than'),
    'each': ('each', 'every', 'per'),
    'remaining': ('remaining', 'left over', 'left'),
    'half': ('half', 'halfway'),
    'double': ('twice', 'double', 'two times'),
    'before': ('before', 'earlier'),
    'after': ('after', 'later'),
    'including': ('including', 'include'),
    'excluding': ('excluding', 'exclude', 'without'),
    'discount': ('discount', 'off'),
    'tax': ('tax', 'taxes'),
    'profit': ('profit', 'gain'),
}
EXTRA_NAMES = (
    'question_number_mention_fraction',
    'question_number_calculation_fraction',
    'mentioned_but_not_calculated_fraction',
    'unit_group_mention_fraction',
    'unit_group_near_calculation_fraction',
    'time_unit_mention',
    'time_unit_near_calculation',
    'constraint_group_mention_fraction',
    'operationalized_half_or_double',
    'missing_condition_fraction',
)


def _canonical(token):
    try:
        return str(Fraction(token.replace(',', '')))
    except (ValueError, ZeroDivisionError):
        return token


def _normalize(text):
    return (text.replace('\\times', '×').replace('\\cdot', '×')
            .replace('\\div', '÷').replace('−', '-'))


def _number_usage(text):
    matches = list(NUMBER.finditer(text))
    mentioned = {_canonical(match.group()) for match in matches}
    calculated = set()
    for first, second in zip(matches, matches[1:]):
        if MATH_GAP.fullmatch(text[first.end():second.start()]):
            calculated.add(_canonical(first.group()))
            calculated.add(_canonical(second.group()))
    return mentioned, calculated


def _words(text):
    return {match.group().lower() for match in WORD.finditer(text)}


def _has_variant(text, variants):
    return any(re.search(r'(?<![A-Za-z])' + re.escape(variant)
                         + r'(?![A-Za-z])', text, re.I)
               for variant in variants)


def _near_calculation(text, variants):
    for variant in variants:
        pattern = re.compile(r'(?<![A-Za-z])' + re.escape(variant)
                             + r'(?![A-Za-z])', re.I)
        for match in pattern.finditer(text):
            window = text[max(0, match.start() - 25):match.end() + 25]
            if MATH_OPERATOR.search(window) and len(NUMBER.findall(window)) >= 2:
                return True
    return False


def condition_features(bank):
    groups = []
    for row in bank:
        question = _normalize(row['question'])
        q_numbers = {_canonical(token) for token in NUMBER.findall(question)}
        q_units = [name for name, variants in UNIT_GROUPS.items()
                   if _has_variant(question, variants)]
        q_constraints = [name for name, variants in CONSTRAINTS.items()
                         if _has_variant(question, variants)]
        group = []
        for candidate in row['candidates']:
            text = _normalize(candidate['text'])
            mentioned, calculated = _number_usage(text)
            num_mentioned = len(q_numbers & mentioned) / max(1, len(q_numbers))
            num_calculated = len(q_numbers & calculated) / max(1, len(q_numbers))
            unit_mentioned = sum(_has_variant(text, UNIT_GROUPS[name])
                                 for name in q_units) / max(1, len(q_units))
            unit_used = sum(_near_calculation(text, UNIT_GROUPS[name])
                            for name in q_units) / max(1, len(q_units))
            time_present = 'time' in q_units
            time_mention = float(time_present and
                                 _has_variant(text, UNIT_GROUPS['time']))
            time_used = float(time_present and
                              _near_calculation(text, UNIT_GROUPS['time']))
            constraint_mention = sum(
                _has_variant(text, CONSTRAINTS[name]) for name in q_constraints
            ) / max(1, len(q_constraints))
            operational = 0.0
            if 'half' in q_constraints:
                operational = float(bool(re.search(r'(?:/|÷)\s*2\b|'
                                                   r'(?:×|\*)\s*0\.5\b', text)))
            elif 'double' in q_constraints:
                operational = float(bool(re.search(r'(?:×|\*)\s*2\b|'
                                                   r'2\s*(?:×|\*)', text)))
            missing = (len(q_numbers - mentioned) +
                       sum(not _has_variant(text, CONSTRAINTS[name])
                           for name in q_constraints)) / max(
                               1, len(q_numbers) + len(q_constraints))
            group.append([
                num_mentioned, num_calculated,
                max(0.0, num_mentioned - num_calculated),
                unit_mentioned, unit_used, time_mention, time_used,
                constraint_mention, operational, missing,
            ])
        groups.append(group)
    return np.asarray(groups, dtype=np.float64)


def make_condition_inputs(bank, arrays):
    return np.concatenate([make_inputs(bank, arrays),
                           condition_features(bank)], axis=-1)


def predict(bank, arrays, checkpoint):
    if checkpoint is None:
        raise ValueError('Condition-coverage model requires --checkpoint')
    with np.load(Path(checkpoint), allow_pickle=False) as saved:
        mean, scale, weights = (saved[k] for k in ('mean', 'scale', 'weights'))
    features = make_condition_inputs(bank, arrays)
    if features.shape[-1] != len(weights):
        raise ValueError('Condition feature/checkpoint mismatch')
    scores = ((features - mean) / scale) @ weights
    return [int(i) for i in scores.argmax(axis=1)]
