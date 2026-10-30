"""Train combined group choice with a hard-wrong-candidate margin term."""

import argparse
from pathlib import Path

import numpy as np

from common import DATA, load_inputs, parse_answer, read_jsonl
from methods.combined_reasoning import make_combined_inputs
from train_group_choice import loss_and_gradient


def hard_loss_and_gradient(features, correct, weights, alpha, margin,
                           regularization):
    base_loss, base_gradient = loss_and_gradient(
        features, correct, weights, regularization)
    scores = features @ weights
    informative = correct.any(axis=1) & (~correct).any(axis=1)
    x = features[informative]
    c = correct[informative]
    s = scores[informative]
    rows = np.arange(len(x))
    good_index = np.where(c, s, -np.inf).argmax(axis=1)
    bad_index = np.where(~c, s, -np.inf).argmax(axis=1)
    gap = margin + s[rows, bad_index] - s[rows, good_index]
    probability = 1 / (1 + np.exp(-np.clip(gap, -40, 40)))
    extra_loss = alpha * np.logaddexp(0, gap).sum()
    delta = x[rows, bad_index] - x[rows, good_index]
    extra_gradient = alpha * np.einsum('nd,n->d', delta, probability)
    return base_loss + float(extra_loss), base_gradient + extra_gradient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path,
                        default=Path('runs/hard_negative_combined/model.npz'))
    parser.add_argument('--regularization', type=float, default=25.0)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--learning-rate', type=float, default=0.02)
    parser.add_argument('--hard-weight', type=float, default=0.5)
    parser.add_argument('--margin', type=float, default=1.0)
    args = parser.parse_args()
    if min(args.regularization, args.epochs, args.learning_rate,
           args.hard_weight, args.margin) <= 0:
        parser.error('All training parameters must be positive')

    bank, arrays = load_inputs('train')
    references = read_jsonl(DATA / 'gsm8k_train.jsonl')
    if len(bank) != len(references) or any(
        row['id'] != ref['id'] or row['question'] != ref['question']
        for row, ref in zip(bank, references)
    ):
        raise ValueError('Training references are not aligned')
    features = make_combined_inputs(bank, arrays)
    flat = features.reshape(-1, features.shape[-1])
    mean = flat.mean(axis=0)
    scale = flat.std(axis=0)
    scale[scale < 1e-8] = 1.0
    features = (features - mean) / scale
    correct = np.asarray([
        [parse_answer(c['text']) == parse_answer(ref['answer'])
         for c in row['candidates']]
        for row, ref in zip(bank, references)
    ], dtype=bool)
    if not correct.any(axis=1).all():
        raise ValueError('Every training group must have a correct candidate')

    weights = np.zeros(features.shape[-1], dtype=np.float64)
    first = np.zeros_like(weights)
    second = np.zeros_like(weights)
    initial_loss, _ = hard_loss_and_gradient(
        features, correct, weights, args.hard_weight, args.margin,
        args.regularization)
    for epoch in range(1, args.epochs + 1):
        _, gradient = hard_loss_and_gradient(
            features, correct, weights, args.hard_weight, args.margin,
            args.regularization)
        first = 0.9 * first + 0.1 * gradient
        second = 0.999 * second + 0.001 * gradient ** 2
        weights -= args.learning_rate * (first / (1 - 0.9 ** epoch)) / (
            np.sqrt(second / (1 - 0.999 ** epoch)) + 1e-8)
    final_loss, _ = hard_loss_and_gradient(
        features, correct, weights, args.hard_weight, args.margin,
        args.regularization)
    if not np.isfinite(final_loss) or final_loss >= initial_loss:
        raise RuntimeError('Training did not reduce the objective')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, mean=mean.astype(np.float32),
             scale=scale.astype(np.float32), weights=weights.astype(np.float32))
    print(f'trained on {len(bank)} groups; '
          f'loss {initial_loss:.3f} -> {final_loss:.3f}; '
          f'checkpoint: {args.output}')


if __name__ == '__main__':
    main()
