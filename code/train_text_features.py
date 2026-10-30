"""Fit a regularized logistic candidate scorer on training data only."""

import argparse
from pathlib import Path

import numpy as np

from common import DATA, load_inputs, parse_answer, read_jsonl
from methods.text_features import FEATURE_NAMES, make_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('runs/text_features/model.npz'))
    parser.add_argument('--regularization', type=float, default=1.0)
    args = parser.parse_args()
    if args.regularization <= 0:
        parser.error('--regularization must be positive')

    bank, _ = load_inputs('train')
    references = read_jsonl(DATA / 'gsm8k_train.jsonl')
    if any(b['id'] != r['id'] for b, r in zip(bank, references)):
        raise ValueError('Training references are not aligned')
    x = make_features(bank).reshape(-1, len(FEATURE_NAMES))
    y = np.asarray([
        parse_answer(c['text']) == parse_answer(reference['answer'])
        for row, reference in zip(bank, references)
        for c in row['candidates']
    ], dtype=np.float64)
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    x = (x - mean) / scale

    # Logistic loss with L2 penalty, solved by Newton steps. The offset is
    # trained but omitted from the checkpoint: it is identical for all four
    # candidates and cannot affect their ranking.
    design = np.column_stack([np.ones(len(x)), x])
    weights = np.zeros(design.shape[1])
    penalty = np.diag([0.0] + [args.regularization] * len(FEATURE_NAMES))
    for _ in range(30):
        z = np.clip(design @ weights, -40, 40)
        probability = 1 / (1 + np.exp(-z))
        curvature = np.maximum(probability * (1 - probability), 1e-8)
        gradient = design.T @ (probability - y) + penalty @ weights
        hessian = design.T @ (curvature[:, None] * design) + penalty
        step = np.linalg.solve(hessian, gradient)
        weights -= step
        if np.max(np.abs(step)) < 1e-7:
            break

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, mean=mean, scale=scale, weights=weights[1:])
    print(f'trained on {len(bank)} questions; {int(y.sum())} correct candidates; '
          f'{len(FEATURE_NAMES)} features; checkpoint: {args.output}')


if __name__ == '__main__':
    main()
