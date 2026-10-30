"""Fit Bradley-Terry pairwise logistic loss on training-only candidate pairs."""

import argparse
from pathlib import Path

import numpy as np

from common import DATA, load_inputs, parse_answer, read_jsonl
from methods.pairwise_comparison import make_inputs


def objective(design, weights, regularization):
    margins = design @ weights
    return float(np.logaddexp(0, -margins).sum()
                 + 0.5 * regularization * (weights @ weights))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path,
                        default=Path('runs/bradley_terry/model.npz'))
    parser.add_argument('--regularization', type=float, default=25.0)
    parser.add_argument('--max-iterations', type=int, default=20)
    args = parser.parse_args()
    if args.regularization <= 0 or args.max_iterations < 1:
        parser.error('Regularization and iteration limit must be positive')

    bank, arrays = load_inputs('train')
    references = read_jsonl(DATA / 'gsm8k_train.jsonl')
    if len(bank) != len(references) or any(
        row['id'] != reference['id'] or row['question'] != reference['question']
        for row, reference in zip(bank, references)
    ):
        raise ValueError('Training references are not aligned')
    features = make_inputs(bank, arrays)
    flat = features.reshape(-1, features.shape[-1])
    mean = flat.mean(axis=0)
    scale = flat.std(axis=0)
    scale[scale < 1e-8] = 1.0
    features = (features - mean) / scale

    # The exact same oriented correct-minus-incorrect pairs as the existing
    # pairwise ridge experiment. Questions with multiple correct candidates
    # contribute every good/bad pair; all-correct groups add no pairs.
    differences = []
    for row, reference, group in zip(bank, references, features):
        gold = parse_answer(reference['answer'])
        good = [j for j, c in enumerate(row['candidates'])
                if parse_answer(c['text']) == gold]
        bad = [j for j in range(4) if j not in good]
        differences.extend(group[j] - group[k] for j in good for k in bad)
    design = np.asarray(differences, dtype=np.float64)
    weights = np.zeros(design.shape[1], dtype=np.float64)

    # Newton optimization of sum log(1 + exp(-D w)) + lambda/2 * ||w||^2.
    # Every accepted step must reduce the exact objective.
    iterations = 0
    for iteration in range(args.max_iterations):
        margins = design @ weights
        negative_probability = 1 / (1 + np.exp(np.clip(margins, -40, 40)))
        curvature = negative_probability * (1 - negative_probability)
        gradient = -(design.T @ negative_probability) + args.regularization * weights
        hessian = design.T @ (curvature[:, None] * design)
        hessian.flat[::len(hessian) + 1] += args.regularization
        step = np.linalg.solve(hessian, gradient)
        old_loss = objective(design, weights, args.regularization)
        rate = 1.0
        while rate > 1 / 1024:
            candidate = weights - rate * step
            if objective(design, candidate, args.regularization) <= old_loss:
                break
            rate *= 0.5
        if rate <= 1 / 1024:
            raise RuntimeError('Newton line search failed to reduce objective')
        weights = candidate
        iterations = iteration + 1
        if np.max(np.abs(rate * step)) < 1e-7:
            break

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, mean=mean.astype(np.float32),
             scale=scale.astype(np.float32), weights=weights.astype(np.float32))
    print(f'trained on {len(bank)} questions and {len(design)} pairs; '
          f'{iterations} Newton iterations; loss {objective(design, weights, args.regularization):.6f}; '
          f'checkpoint: {args.output}')


if __name__ == '__main__':
    main()
