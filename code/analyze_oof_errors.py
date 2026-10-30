"""Five-fold out-of-fold error analysis for the combined reasoning model."""

import argparse
import json
from pathlib import Path

import numpy as np

from common import DATA, load_inputs, parse_answer, read_jsonl
from methods.arithmetic_consistency import arithmetic_features
from methods.combined_reasoning import final_link_features, make_combined_inputs
from methods.condition_coverage import condition_features
from methods.process_verifier import make_process_inputs
from methods.role_aware_process import make_role_inputs
from train_hard_negative_combined import hard_loss_and_gradient
from train_group_choice import loss_and_gradient


def fit(features, correct, method, regularization=25.0, epochs=300,
        learning_rate=0.02):
    weights = np.zeros(features.shape[-1], dtype=np.float64)
    first = np.zeros_like(weights)
    second = np.zeros_like(weights)
    for epoch in range(1, epochs + 1):
        if method == 'hard_negative_combined':
            _, gradient = hard_loss_and_gradient(
                features, correct, weights, 0.5, 1.0, regularization)
        else:
            _, gradient = loss_and_gradient(features, correct, weights,
                                            regularization)
        first = 0.9 * first + 0.1 * gradient
        second = 0.999 * second + 0.001 * gradient ** 2
        weights -= learning_rate * (first / (1 - 0.9 ** epoch)) / (
            np.sqrt(second / (1 - 0.999 ** epoch)) + 1e-8)
    return weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path,
                        default=Path('runs/oof_analysis/summary.json'))
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--method', choices=('combined_reasoning', 'process_verifier',
                                             'hard_negative_combined',
                                             'role_aware_process'),
                        default='combined_reasoning')
    args = parser.parse_args()
    if args.folds < 2:
        parser.error('--folds must be at least two')

    bank, arrays = load_inputs('train')
    references = read_jsonl(DATA / 'gsm8k_train.jsonl')
    if len(bank) != len(references) or any(
        row['id'] != ref['id'] or row['question'] != ref['question']
        for row, ref in zip(bank, references)
    ):
        raise ValueError('Training references are not aligned')
    if args.method in ('combined_reasoning', 'hard_negative_combined'):
        features = make_combined_inputs(bank, arrays)
    elif args.method == 'process_verifier':
        features = make_process_inputs(bank, arrays)
    else:
        features = make_role_inputs(bank, arrays)
    arithmetic = arithmetic_features(bank)
    conditions = condition_features(bank)
    links = final_link_features(bank)
    parsed = [[parse_answer(c['text']) for c in row['candidates']]
              for row in bank]
    correct = np.asarray([
        [answer == parse_answer(ref['answer']) for answer in answers]
        for answers, ref in zip(parsed, references)
    ], dtype=bool)

    rng = np.random.default_rng(args.seed)
    permuted = rng.permutation(len(bank))
    folds = np.empty(len(bank), dtype=np.int64)
    folds[permuted] = np.arange(len(bank)) % args.folds
    scores = np.empty((len(bank), 4), dtype=np.float64)
    for fold in range(args.folds):
        training = folds != fold
        heldout = ~training
        x = features[training]
        flat = x.reshape(-1, x.shape[-1])
        mean = flat.mean(axis=0)
        scale = flat.std(axis=0)
        scale[scale < 1e-8] = 1.0
        standardized = (x - mean) / scale
        weights = fit(standardized, correct[training], args.method)
        scores[heldout] = ((features[heldout] - mean) / scale) @ weights
        print(f'fold {fold + 1}/{args.folds}: train={int(training.sum())}, '
              f'heldout={int(heldout.sum())}', flush=True)

    selected = scores.argmax(axis=1)
    errors = []
    categories = {
        'selected_has_wrong_simple_equation': 0,
        'selected_omits_question_number_from_calculation': 0,
        'selected_last_equation_conflicts_with_final': 0,
        'selected_wrong_answer_shared_by_peer': 0,
        'none_of_these_signals': 0,
    }
    hard_count = hard_correct = 0
    for i, (row, answers) in enumerate(zip(bank, parsed)):
        hard = len({a for a in answers if a is not None}) >= 2
        picked = int(selected[i])
        good = bool(correct[i, picked])
        if hard:
            hard_count += 1
            hard_correct += good
        if good or not hard:
            continue
        flags = {
            'selected_has_wrong_simple_equation': bool(arithmetic[i, picked, 1] > 0),
            'selected_omits_question_number_from_calculation': bool(
                conditions[i, picked, 0] > conditions[i, picked, 1]),
            'selected_last_equation_conflicts_with_final': bool(links[i, picked, 1] > 0),
            'selected_wrong_answer_shared_by_peer': bool(
                answers[picked] is not None and answers.count(answers[picked]) > 1),
        }
        if not any(flags.values()):
            categories['none_of_these_signals'] += 1
        for name, value in flags.items():
            categories[name] += int(value)
        errors.append({
            'id': row['id'], 'fold': int(folds[i]),
            'selected': picked,
            'correct_indices': [j for j in range(4) if correct[i, j]],
            'selected_answer': answers[picked],
            'answers': answers,
            'score_margin_over_best_correct': float(
                scores[i, picked] - max(scores[i, correct[i]])),
            'signals': flags,
        })
    result = {
        'method': args.method, 'seed': args.seed,
        'folds': args.folds, 'train_questions': len(bank),
        'oof_correct': int(correct[np.arange(len(bank)), selected].sum()),
        'hard_questions': hard_count, 'hard_correct': hard_correct,
        'hard_errors': len(errors), 'nonexclusive_error_signals': categories,
        'errors': errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'errors'}, indent=2))


if __name__ == '__main__':
    main()
