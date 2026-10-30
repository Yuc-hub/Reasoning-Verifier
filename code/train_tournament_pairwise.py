"""Train pair-dependent preference margins from training-only candidate pairs."""

import argparse
from pathlib import Path

import numpy as np

from common import DATA, load_inputs, parse_answer, read_jsonl
from methods.pairwise_comparison import make_inputs
from methods.tournament_pairwise import PAIRS, PAIR_DIM, pair_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path,
                        default=Path('runs/tournament_pairwise/model.npz'))
    parser.add_argument('--ridge', type=float, default=100.0)
    args = parser.parse_args()
    if args.ridge <= 0:
        parser.error('--ridge must be positive')

    bank, arrays = load_inputs('train')
    references = read_jsonl(DATA / 'gsm8k_train.jsonl')
    if len(bank) != len(references) or any(
        row['id'] != ref['id'] or row['question'] != ref['question']
        for row, ref in zip(bank, references)
    ):
        raise ValueError('Training references are not aligned')
    base = make_inputs(bank, arrays)
    flat = base.reshape(-1, base.shape[-1])
    mean = flat.mean(axis=0)
    scale = flat.std(axis=0)
    scale[scale < 1e-8] = 1.0
    pairs = pair_features((base - mean) / scale)
    labels = np.asarray([
        [parse_answer(c['text']) == parse_answer(ref['answer'])
         for c in row['candidates']]
        for row, ref in zip(bank, references)
    ], dtype=bool)
    pair_rows = []
    for p, (j, k) in enumerate(PAIRS):
        informative = labels[:, j] != labels[:, k]
        direction = np.where(labels[informative, j], 1.0, -1.0)
        pair_rows.append(pairs[informative, p] * direction[:, None])
    design = np.concatenate(pair_rows, axis=0)
    if design.shape[1] != PAIR_DIM:
        raise ValueError('Unexpected pair feature dimension')
    gram = design.T @ design
    gram.flat[::len(gram) + 1] += args.ridge
    weights = np.linalg.solve(gram, design.sum(axis=0))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, mean=mean.astype(np.float32),
             scale=scale.astype(np.float32), weights=weights.astype(np.float32))
    print(f'trained on {len(bank)} questions and {len(design)} informative pairs; '
          f'{PAIR_DIM} pair features; checkpoint: {args.output}')


if __name__ == '__main__':
    main()
