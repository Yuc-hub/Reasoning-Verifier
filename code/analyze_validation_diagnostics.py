"""Aggregate validation-only confidence and selector-disagreement diagnostics."""
import argparse
import json
from pathlib import Path

import numpy as np

from common import DATA, load_inputs, parse_answer, read_jsonl
from methods.answer_agreement import predict as vote_predict
from methods.embedding_linear import predict as embedding_predict
from methods.pairwise_comparison import predict as pairwise_predict
from methods.role_aware_process import make_role_inputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('results/validation-diagnostics.json'))
    args = parser.parse_args()
    bank, arrays = load_inputs('validation')
    references = read_jsonl(DATA / 'gsm8k_validation.jsonl')
    assert all(b['id'] == r['id'] for b, r in zip(bank, references))
    correct = np.asarray([[parse_answer(c['text']) == parse_answer(r['answer'])
                           for c in b['candidates']]
                          for b, r in zip(bank, references)], dtype=bool)
    with np.load('checkpoints/role_aware_process/model.npz', allow_pickle=False) as f:
        mean, scale, weights = (f[k] for k in ('mean', 'scale', 'weights'))
    scores = ((make_role_inputs(bank, arrays) - mean) / scale) @ weights
    order = scores.argsort(axis=1)
    chosen = order[:, -1]
    margin = scores[np.arange(len(bank)), order[:, -1]] - scores[np.arange(len(bank)), order[:, -2]]
    success = correct[np.arange(len(bank)), chosen]
    answers = [[parse_answer(c['text']) for c in b['candidates']] for b in bank]
    hard = np.asarray([len({v for v in row if v is not None}) >= 2 for row in answers])

    def margin_quartiles(mask):
        selected = np.flatnonzero(mask)
        quartiles = []
        for ids in np.array_split(selected[np.argsort(margin[selected], kind='stable')], 4):
            quartiles.append({'n': len(ids), 'correct': int(success[ids].sum()),
                              'accuracy_percent': float(100 * success[ids].mean()),
                              'margin_min': float(margin[ids].min()),
                              'margin_max': float(margin[ids].max())})
        return quartiles
    choices = {
        'answer_agreement': np.asarray(vote_predict(bank, arrays, None)),
        'embedding_linear': np.asarray(embedding_predict(bank, arrays, Path('runs/embedding_linear/model.npz'))),
        'pairwise_comparison': np.asarray(pairwise_predict(bank, arrays, Path('runs/pairwise_comparison/model.npz'))),
        'role_aware_process': chosen,
    }
    selected_answers = {name: np.asarray([answers[i][j] for i, j in enumerate(indexes)])
                        for name, indexes in choices.items()}
    unanimous = np.logical_and.reduce([v == selected_answers['role_aware_process']
                                       for v in selected_answers.values()])
    disagreement = {}
    for name, other in choices.items():
        if name == 'role_aware_process':
            continue
        same = selected_answers[name] == selected_answers['role_aware_process']
        disagreement[name] = {
            'different_choices': int((~same).sum()),
            'role_correct_when_different': int(success[~same].sum()),
            'other_correct_when_different': int(correct[np.arange(len(bank))[~same], other[~same]].sum()),
        }
    result = {
        'split': 'validation', 'n': len(bank), 'role_correct': int(success.sum()),
        'score_margin_note': 'Raw linear top-two score gap; not a calibrated probability.',
        'margin_quartiles_low_to_high': margin_quartiles(np.ones(len(bank), dtype=bool)),
        'hard_groups': {'n': int(hard.sum()), 'correct': int(success[hard].sum()),
                        'margin_quartiles_low_to_high': margin_quartiles(hard)},
        'all_four_selectors_same_final_answer': {'n': int(unanimous.sum()),
                                                   'correct': int(success[unanimous].sum())},
        'at_least_one_selector_different_final_answer': {'n': int((~unanimous).sum()),
                                                           'correct': int(success[~unanimous].sum())},
        'pairwise_disagreements': disagreement,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
