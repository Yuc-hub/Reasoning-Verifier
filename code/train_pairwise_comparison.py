"""Fit candidate ranking from correct-versus-incorrect pairs per question."""

import argparse
from pathlib import Path

import numpy as np

from common import DATA, load_inputs, parse_answer, read_jsonl
from methods.pairwise_comparison import make_inputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path,
                        default=Path('runs/pairwise_comparison/model.npz'))
    parser.add_argument('--ridge', type=float, default=100.0,
                        help='L2 penalty for the summed pairwise squared error')
    args = parser.parse_args()
    if args.ridge <= 0:
        parser.error('--ridge must be positive')

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

    # Every row is a within-question correct-minus-incorrect difference.
    # Questions with all four correct provide no ordering signal and are
    # omitted. Multiple correct candidates each contribute valid pairs.
    differences = []
    for row, reference, group in zip(bank, references, features):
        gold = parse_answer(reference['answer'])
        good = [j for j, c in enumerate(row['candidates'])
                if parse_answer(c['text']) == gold]
        bad = [j for j in range(4) if j not in good]
        differences.extend(group[j] - group[k] for j in good for k in bad)
    differences = np.asarray(differences, dtype=np.float64)

    # Fit margin 1 by regularized least squares: min ||D w - 1||² + λ||w||².
    gram = differences.T @ differences
    gram.flat[::len(gram) + 1] += args.ridge
    weights = np.linalg.solve(gram, differences.sum(axis=0))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, mean=mean.astype(np.float32),
             scale=scale.astype(np.float32), weights=weights.astype(np.float32))
    print(f'trained on {len(bank)} questions and {len(differences)} '
          f'correct/incorrect pairs; checkpoint: {args.output}')


if __name__ == '__main__':
    main()
