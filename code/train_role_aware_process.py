"""Train the process verifier with target-role and quantity-source cues."""

import argparse
from pathlib import Path

import numpy as np

from common import DATA, load_inputs, parse_answer, read_jsonl
from methods.role_aware_process import make_role_inputs
from train_group_choice import loss_and_gradient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path,
                        default=Path('runs/role_aware_process/model.npz'))
    parser.add_argument('--regularization', type=float, default=25.0)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--learning-rate', type=float, default=0.02)
    args = parser.parse_args()
    if args.regularization <= 0 or args.epochs < 1 or args.learning_rate <= 0:
        parser.error('Regularization, epochs and learning rate must be positive')
    bank, arrays = load_inputs('train')
    references = read_jsonl(DATA / 'gsm8k_train.jsonl')
    if len(bank) != len(references) or any(
        row['id'] != ref['id'] or row['question'] != ref['question']
        for row, ref in zip(bank, references)
    ):
        raise ValueError('Training references are not aligned')
    features = make_role_inputs(bank, arrays)
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
    initial_loss, _ = loss_and_gradient(features, correct, weights,
                                        args.regularization)
    for epoch in range(1, args.epochs + 1):
        _, gradient = loss_and_gradient(features, correct, weights,
                                        args.regularization)
        first = 0.9 * first + 0.1 * gradient
        second = 0.999 * second + 0.001 * gradient ** 2
        weights -= args.learning_rate * (first / (1 - 0.9 ** epoch)) / (
            np.sqrt(second / (1 - 0.999 ** epoch)) + 1e-8)
    final_loss, _ = loss_and_gradient(features, correct, weights,
                                      args.regularization)
    if not np.isfinite(final_loss) or final_loss >= initial_loss:
        raise RuntimeError('Training did not reduce the objective')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, mean=mean.astype(np.float32),
             scale=scale.astype(np.float32), weights=weights.astype(np.float32))
    print(f'trained on {len(bank)} groups; {features.shape[-1]} features; '
          f'loss {initial_loss:.3f} -> {final_loss:.3f}; '
          f'checkpoint: {args.output}')


if __name__ == '__main__':
    main()
