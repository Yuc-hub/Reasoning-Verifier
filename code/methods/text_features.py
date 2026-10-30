"""Small, interpretable text-only candidate reranker."""

from fractions import Fraction
from pathlib import Path
import re

import numpy as np

from common import parse_answer


NUMBERS = re.compile(r'[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?')
WORDS = re.compile(r"[A-Za-z]+")
FEATURE_NAMES = (
    'answer_agreement', 'answer_unique', 'answer_missing',
    'answer_log_abs', 'answer_integer', 'answer_negative',
    'text_log_chars', 'text_log_words', 'text_log_numbers',
    'text_log_lines', 'text_log_equals', 'text_log_operators',
    'has_boxed', 'has_final_answer', 'has_hash_answer',
    'question_number_recall', 'question_word_recall',
    'chars_relative', 'numbers_relative', 'question_overlap_relative',
)


def _one(row):
    question = row['question']
    q_numbers = set(NUMBERS.findall(question))
    q_words = {w.lower() for w in WORDS.findall(question) if len(w) >= 4}
    texts = [c['text'] for c in row['candidates']]
    answers = [parse_answer(t) for t in texts]
    raw = []
    for text, answer in zip(texts, answers):
        numbers = set(NUMBERS.findall(text))
        words = {w.lower() for w in WORDS.findall(text) if len(w) >= 4}
        count = answers.count(answer) if answer is not None else 0
        value = Fraction(answer) if answer is not None else None
        raw.append([
            float(count), float(count == 1), float(answer is None),
            np.log1p(abs(float(value))) if value is not None else 0.0,
            float(value.denominator == 1) if value is not None else 0.0,
            float(value < 0) if value is not None else 0.0,
            np.log1p(len(text)), np.log1p(len(WORDS.findall(text))),
            np.log1p(len(NUMBERS.findall(text))), np.log1p(text.count('\n')),
            np.log1p(text.count('=')),
            np.log1p(sum(text.count(op) for op in '+-*/×÷')),
            float('\\boxed{' in text),
            float('final answer' in text.lower()), float('####' in text),
            len(q_numbers & numbers) / max(1, len(q_numbers)),
            len(q_words & words) / max(1, len(q_words)),
        ])
    base = np.asarray(raw, dtype=np.float64)
    relative = base[:, [6, 8, 15]] - base[:, [6, 8, 15]].mean(axis=0)
    return np.concatenate([base, relative], axis=1)


def make_features(bank):
    """Return float64 features of shape [N, 4, len(FEATURE_NAMES)]."""
    return np.stack([_one(row) for row in bank])


def predict(bank, arrays, checkpoint):
    if checkpoint is None:
        raise ValueError('Text feature model requires --checkpoint')
    path = Path(checkpoint)
    with np.load(path, allow_pickle=False) as saved:
        mean, scale, weights = (saved[k] for k in ('mean', 'scale', 'weights'))
    features = make_features(bank)
    if features.shape[-1] != len(weights):
        raise ValueError('Checkpoint feature count does not match method')
    scores = ((features - mean) / scale) @ weights
    return [int(i) for i in scores.argmax(axis=1)]
