"""Select a small prespecified fusion grid on validation, then freeze assets."""

import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from common import DATA, load_inputs, parse_answer, read_jsonl
from methods.score_fusion import component_scores, fused_scores


# Keep this grid small to limit validation-set overfitting. The first entry is
# the existing pairwise method, serving as a no-fusion control.
WEIGHT_GRID = (
    (1.0, 0.0, 0.0),
    (1.0, 0.25, 0.0),
    (1.0, 0.5, 0.0),
    (1.0, 1.0, 0.0),
    (1.0, 0.0, 0.25),
    (1.0, 0.0, 0.5),
    (1.0, 0.25, 0.25),
    (1.0, 0.5, 0.25),
    (1.0, 0.25, 0.5),
    (1.0, 0.5, 0.5),
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pairwise', type=Path,
                        default=Path('runs/pairwise_comparison/model.npz'))
    parser.add_argument('--text', type=Path,
                        default=Path('runs/text_features/model.npz'))
    parser.add_argument('--embedding', type=Path,
                        default=Path('runs/embedding_linear/model.npz'))
    parser.add_argument('--output', type=Path,
                        default=Path('runs/score_fusion/model_bundle'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for name in ('pairwise', 'text', 'embedding'):
        source = getattr(args, name)
        target = args.output / f'{name}.npz'
        if source.resolve() != target.resolve():
            shutil.copyfile(source, target)

    bank, arrays = load_inputs('validation')
    references = read_jsonl(DATA / 'gsm8k_validation.jsonl')
    if len(bank) != len(references) or any(
        row['id'] != reference['id'] or row['question'] != reference['question']
        for row, reference in zip(bank, references)
    ):
        raise ValueError('Validation references are not aligned')
    correct = np.asarray([
        [parse_answer(c['text']) == parse_answer(r['answer'])
         for c in row['candidates']]
        for row, r in zip(bank, references)
    ], dtype=bool)
    components = component_scores(bank, arrays, args.output)
    results = []
    rows = np.arange(len(bank))
    for pairwise, text, embedding in WEIGHT_GRID:
        weights = {'pairwise': pairwise, 'text': text, 'embedding': embedding}
        choices = fused_scores(components, weights).argmax(axis=1)
        count = int(correct[rows, choices].sum())
        results.append({'weights': weights, 'correct': count,
                        'accuracy': 100 * count / len(bank)})
    # On a tie, retain the earlier and simpler grid entry.
    best = max(results, key=lambda item: item['correct'])
    (args.output / 'config.json').write_text(json.dumps({
        'weights': best['weights'], 'selection_split': 'validation',
        'grid_size': len(WEIGHT_GRID),
    }, indent=2) + '\n')
    (args.output.parent / 'validation-grid.json').write_text(
        json.dumps(results, indent=2) + '\n')
    for name, weight in best['weights'].items():
        if weight == 0:
            (args.output / f'{name}.npz').unlink()
    print(json.dumps({'best': best, 'grid_size': len(results)}, indent=2))


if __name__ == '__main__':
    main()
