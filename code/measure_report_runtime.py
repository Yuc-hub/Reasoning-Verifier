"""Measure fresh-process official test prediction plus independent scoring."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

METHODS = [
    ('answer_agreement', 'methods/answer_agreement.py', None),
    ('text_features', 'methods/text_features.py', 'runs/text_features/model.npz'),
    ('embedding_linear', 'methods/embedding_linear.py', 'runs/embedding_linear/model.npz'),
    ('pairwise_comparison', 'methods/pairwise_comparison.py', 'runs/pairwise_comparison/model.npz'),
    ('group_choice', 'methods/group_choice.py', 'runs/group_choice/model.npz'),
    ('combined_reasoning', 'methods/combined_reasoning.py', 'runs/combined_reasoning/model.npz'),
    ('process_verifier', 'methods/process_verifier.py', 'runs/process_verifier/model.npz'),
    ('role_aware_process', 'submission.py', 'checkpoints/role_aware_process/model.npz'),
]


def main():
    env = os.environ.copy()
    env.update({k: '4' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')})
    out = []
    folder = Path('runs/report_runtime')
    folder.mkdir(parents=True, exist_ok=True)
    for name, method, checkpoint in METHODS:
        pred = folder / f'{name}-test-predictions.json'
        score = folder / f'{name}-test-score.json'
        command = [sys.executable, 'predict.py', '--method', method, '--split', 'test', '--output', str(pred)]
        if checkpoint:
            command += ['--checkpoint', checkpoint]
        start = time.perf_counter()
        subprocess.run(command, env=env, check=True, capture_output=True, text=True)
        prediction_wall = time.perf_counter() - start
        start = time.perf_counter()
        subprocess.run([sys.executable, 'score_predictions.py', '--split', 'test',
                        '--predictions', str(pred), '--output', str(score)],
                       env=env, check=True, capture_output=True, text=True)
        scoring_wall = time.perf_counter() - start
        official = json.loads(score.read_text())
        out.append({'method': name, 'prediction_wall_seconds': prediction_wall,
                    'scoring_wall_seconds': scoring_wall,
                    'total_wall_seconds': prediction_wall + scoring_wall,
                    'correct': official['correct'], 'n': official['n']})
        print(out[-1], flush=True)
    Path('results/runtime-comparison.json').write_text(json.dumps({
        'machine': 'local Apple Silicon arm64',
        'protocol': 'one fresh prediction process plus one independent scoring process per method; four thread environment variables; Python executable from mp1',
        'runs': out,
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
