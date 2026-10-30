"""Fit a ridge-regression scorer on training embeddings and answer labels."""

import argparse
from pathlib import Path

import numpy as np

from common import DATA, load_inputs, parse_answer, read_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('runs/embedding_linear/model.npz'))
    parser.add_argument('--ridge', type=float, default=100.0,
                        help='L2 penalty in the summed squared-error objective')
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
    x = arrays['embeddings'].reshape(-1, 1024).astype(np.float64)
    y = np.asarray([
        parse_answer(candidate['text']) == parse_answer(reference['answer'])
        for row, reference in zip(bank, references)
        for candidate in row['candidates']
    ], dtype=np.float64)

    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    x = (x - mean) / scale
    y_centered = y - y.mean()
    gram = x.T @ x
    gram.flat[::len(gram) + 1] += args.ridge
    weights = np.linalg.solve(gram, x.T @ y_centered)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, mean=mean.astype(np.float32),
             scale=scale.astype(np.float32), weights=weights.astype(np.float32))
    print(f'trained on {len(bank)} questions; {int(y.sum())} correct candidates; '
          f'{x.shape[1]} embedding dimensions; checkpoint: {args.output}')


if __name__ == '__main__':
    main()
