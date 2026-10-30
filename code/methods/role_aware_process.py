"""Process verifier with question-target and quantity-role alignment cues."""

from fractions import Fraction
from pathlib import Path
import re

import numpy as np

from methods.process_verifier import _steps, make_process_inputs


TARGET = re.compile(r'\b(?:how many|how much|what (?:is|was|will|would)|find)\b', re.I)
WORDS = re.compile(r"[A-Za-z]+")
PERCENT = re.compile(r'(\d+(?:\.\d+)?)\s*%')
MONEY = re.compile(r'\$\s*(\d[\d,]*(?:\.\d+)?)')
STOP = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'will', 'would',
        'does', 'did', 'do', 'if', 'of', 'in', 'on', 'to', 'for', 'by',
        'with', 'and', 'or', 'they', 'them', 'their', 'he', 'she', 'it',
        'his', 'her', 'you', 'many', 'much', 'what', 'how', 'total'}
ROLES = {
    'paid': ('paid', 'pay', 'spend', 'spent', 'payment'),
    'original': ('original', 'before discount', 'initial price'),
    'remaining': ('remaining', 'left', 'left over'),
    'used': ('used', 'spent', 'consumed'),
    'profit': ('profit', 'earned', 'gain'),
    'difference': ('difference', 'more than', 'less than'),
    'price': ('price', 'cost', 'charge'),
    'count': ('number of', 'how many', 'count'),
}
EXTRA_NAMES = (
    'target_keyword_recall_in_conclusion',
    'target_conclusion_keyword_jaccard',
    'target_role_mention_fraction',
    'target_role_conflict',
    'question_percent_factor_in_verified_step',
    'question_money_value_in_verified_step',
    'question_percent_or_money_missing_fraction',
    'conclusion_mentions_target_unit',
)
UNITS = ('dollar', 'dollars', 'cent', 'cents', 'hour', 'hours', 'minute',
         'minutes', 'mile', 'miles', 'km', 'ounce', 'ounces', 'pound',
         'pounds', 'book', 'books', 'point', 'points', 'item', 'items')


def _keywords(text):
    return {word.lower() for word in WORDS.findall(text)
            if len(word) >= 3 and word.lower() not in STOP}


def _has_role(text, role):
    return any(re.search(r'(?<![A-Za-z])' + re.escape(phrase)
                         + r'(?![A-Za-z])', text, re.I)
               for phrase in ROLES[role])


def _target_text(question):
    matches = list(TARGET.finditer(question))
    if matches:
        return question[matches[-1].start():]
    return question.split('.')[-1]


def _conclusion_text(text):
    lower = text.lower()
    positions = [lower.rfind(marker) for marker in
                 ('final answer', '####', '\\boxed{', 'answer:')]
    marker = max(positions)
    before = text[:marker] if marker >= 0 else text
    return before[-350:]


def role_features(bank):
    groups = []
    for row in bank:
        target = _target_text(row['question'])
        target_words = _keywords(target)
        target_roles = [name for name in ROLES if _has_role(target, name)]
        target_units = {unit for unit in UNITS
                        if re.search(r'\b' + re.escape(unit) + r'\b', target, re.I)}
        percent_factors = set()
        for match in PERCENT.finditer(row['question']):
            percent_factors.add(Fraction(match.group(1)) / 100)
        money_values = set()
        for match in MONEY.finditer(row['question']):
            money_values.add(Fraction(match.group(1).replace(',', '')))
        group = []
        for candidate in row['candidates']:
            conclusion = _conclusion_text(candidate['text'])
            conclusion_words = _keywords(conclusion)
            shared = target_words & conclusion_words
            recall = len(shared) / max(1, len(target_words))
            jaccard = len(shared) / max(1, len(target_words | conclusion_words))
            role_recall = sum(_has_role(conclusion, role)
                              for role in target_roles) / max(1, len(target_roles))
            conflict = float(
                ('paid' in target_roles and not _has_role(conclusion, 'paid')
                 and _has_role(conclusion, 'original')) or
                ('remaining' in target_roles and not _has_role(conclusion, 'remaining')
                 and _has_role(conclusion, 'used'))
            )
            verified_literals = set()
            for valid, _, literals in _steps(candidate['text']):
                if valid:
                    verified_literals.update(literals)
            percent_used = len(percent_factors & verified_literals) / max(
                1, len(percent_factors))
            money_used = len(money_values & verified_literals) / max(
                1, len(money_values))
            missing_source = (
                len(percent_factors - verified_literals)
                + len(money_values - verified_literals)
            ) / max(1, len(percent_factors) + len(money_values))
            unit_mentioned = float(bool(target_units) and any(
                re.search(r'\b' + re.escape(unit) + r'\b', conclusion, re.I)
                for unit in target_units))
            group.append([
                recall, jaccard, role_recall, conflict,
                percent_used, money_used, missing_source, unit_mentioned,
            ])
        groups.append(group)
    return np.asarray(groups, dtype=np.float64)


def make_role_inputs(bank, arrays):
    return np.concatenate([make_process_inputs(bank, arrays),
                           role_features(bank)], axis=-1)


def predict(bank, arrays, checkpoint):
    if checkpoint is None:
        raise ValueError('Role-aware process model requires --checkpoint')
    with np.load(Path(checkpoint), allow_pickle=False) as saved:
        mean, scale, weights = (saved[k] for k in ('mean', 'scale', 'weights'))
    features = make_role_inputs(bank, arrays)
    if features.shape[-1] != len(weights):
        raise ValueError('Role feature/checkpoint mismatch')
    scores = ((features - mean) / scale) @ weights
    return [int(i) for i in scores.argmax(axis=1)]
