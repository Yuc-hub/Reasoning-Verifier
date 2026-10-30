"""Combined group-choice model with a conservative calculation-chain verifier."""

import ast
from fractions import Fraction
from pathlib import Path
import re

import numpy as np

from common import parse_answer
from methods.combined_reasoning import make_combined_inputs
from methods.condition_coverage import NUMBER


NUMERIC_SPAN = re.compile(r'[(-]?\d[0-9., \t+\-*/()]*')
PERCENT = re.compile(r'(\d+(?:\.\d+)?)\s*%')
FRAC = re.compile(r'\\frac\{(\d+(?:\.\d+)?)\}\{(\d+(?:\.\d+)?)\}')
FEATURE_NAMES = (
    'log_verified_steps', 'log_invalid_steps', 'verified_step_fraction',
    'final_chain_depth_fraction', 'final_chain_question_number_coverage',
    'final_has_verified_derivation', 'verified_result_reuse_fraction',
    'orphan_verified_result_fraction', 'final_equals_question_number',
)


def _normalize(text):
    text = FRAC.sub(r'(\1/\2)', text)
    text = PERCENT.sub(r'(\1/100)', text)
    text = (text.replace('\\times', '*').replace('\\cdot', '*')
            .replace('\\div', '/').replace('×', '*').replace('÷', '/')
            .replace('−', '-').replace('\\boxed{', ' ')
            .replace('**', '').replace('$', '').replace('}', ' '))
    return re.sub(r'(?<=\d),(?=\d{3}(?:\D|$))', '', text)


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return Fraction(str(node.value))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _eval(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp):
        a, b = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Add):
            return a + b
        if isinstance(node.op, ast.Sub):
            return a - b
        if isinstance(node.op, ast.Mult):
            return a * b
        if isinstance(node.op, ast.Div) and b != 0:
            return a / b
    raise ValueError('Unsupported numeric expression')


def _expression(segment, side):
    spans = NUMERIC_SPAN.findall(segment)
    if not spans:
        return None
    raw = spans[-1] if side == 'left' else spans[0]
    raw = raw.strip(' +\t')
    if not raw:
        return None
    try:
        tree = ast.parse(raw, mode='eval')
        value = _eval(tree)
        literals = [Fraction(str(node.value)) for node in ast.walk(tree)
                    if isinstance(node, ast.Constant)
                    and type(node.value) in (int, float)]
        return value, literals, raw
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError):
        return None


def _steps(text):
    steps = []
    normalized = _normalize(text)
    for line in normalized.splitlines():
        parts = line.split('=')
        if len(parts) < 2:
            continue
        for left, right in zip(parts, parts[1:]):
            lhs = _expression(left, 'left')
            rhs = _expression(right, 'right')
            if lhs is None or rhs is None:
                continue
            # Require an operation, not a bare restatement such as "5 = 5".
            if not any(op in lhs[2] for op in '+-*/'):
                continue
            valid = abs(lhs[0] - rhs[0]) <= Fraction(1, 1000)
            steps.append((valid, rhs[0], lhs[1]))
    return steps


def process_features(bank):
    groups = []
    for row in bank:
        q_numbers = set()
        for match in NUMBER.finditer(row['question']):
            try:
                q_numbers.add(Fraction(match.group().replace(',', '')))
            except (ValueError, ZeroDivisionError):
                pass
        group = []
        for candidate in row['candidates']:
            steps = _steps(candidate['text'])
            n = len(steps)
            valid_count = sum(valid for valid, _, _ in steps)
            invalid_count = n - valid_count
            answer = parse_answer(candidate['text'])
            final = Fraction(answer) if answer is not None else None
            depths = []
            origins = []
            reused = set()
            for i, (valid, result, literals) in enumerate(steps):
                if not valid:
                    depths.append(0)
                    origins.append(set())
                    continue
                parents = [j for j in range(i) if steps[j][0]
                           and steps[j][1] in literals]
                for parent in parents:
                    reused.add(parent)
                depths.append(1 + max((depths[j] for j in parents), default=0))
                source = set(literals) & q_numbers
                for parent in parents:
                    source.update(origins[parent])
                origins.append(source)
            final_nodes = [i for i, (valid, result, _) in enumerate(steps)
                           if valid and final is not None and result == final]
            best = max(final_nodes, key=lambda i: depths[i]) if final_nodes else None
            final_depth = depths[best] / max(1, valid_count) if best is not None else 0.0
            question_coverage = (
                len(origins[best]) / len(q_numbers)
                if best is not None and q_numbers else 0.0
            )
            orphan = sum(valid and i not in reused and i not in final_nodes
                         for i, (valid, _, _) in enumerate(steps))
            group.append([
                np.log1p(valid_count), np.log1p(invalid_count),
                valid_count / max(1, n), final_depth,
                question_coverage, float(best is not None),
                len(reused) / max(1, valid_count),
                orphan / max(1, valid_count),
                float(final is not None and final in q_numbers),
            ])
        groups.append(group)
    return np.asarray(groups, dtype=np.float64)


def make_process_inputs(bank, arrays):
    return np.concatenate([make_combined_inputs(bank, arrays),
                           process_features(bank)], axis=-1)


def predict(bank, arrays, checkpoint):
    if checkpoint is None:
        raise ValueError('Process verifier requires --checkpoint')
    with np.load(Path(checkpoint), allow_pickle=False) as saved:
        mean, scale, weights = (saved[key] for key in ('mean', 'scale', 'weights'))
    features = make_process_inputs(bank, arrays)
    if features.shape[-1] != len(weights):
        raise ValueError('Process feature/checkpoint mismatch')
    scores = ((features - mean) / scale) @ weights
    return [int(i) for i in scores.argmax(axis=1)]
