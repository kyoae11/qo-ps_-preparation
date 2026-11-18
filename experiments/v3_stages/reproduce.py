"""Post-evaluation checks: standalone inference, selection resume, one seed replay.

Writes only to reproducibility/. No model choice or test-driven tuning occurs.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--gpu', default='0')
    args = ap.parse_args()
    root = Path(args.root).resolve()
    assert args.gpu == '0', 'This replay is authorized for physical GPU0 only'
    verification = json.loads((root / 'evaluation/independent_verification.json').read_text())
    assert verification['frozen_artifacts_unchanged'] and verification['source_data_unchanged']
    selected = json.loads((root / 'selection.json').read_text())
    out = root / 'reproducibility'
    out.mkdir(exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
               MKL_NUM_THREADS='4', CUBLAS_WORKSPACE_CONFIG=':4096:8')

    # The independent inference process opens inputs and weights, never labels.
    predictions = out / 'standalone_predictions_seed11.csv'
    subprocess.run([sys.executable, str(root / 'code/predict.py'), '--root', str(root),
                    '--inputs', str(root / 'cache/test_x.pkl'), '--out', str(predictions),
                    '--seed', '11'], env=env, check=True)
    predicted = pd.read_csv(predictions)
    differences = {}
    for stage, column in [('network_time', 'network'), ('ready_time', 'ready'),
                          ('execution_time', 'execution'), ('completion_time', 'completion')]:
        saved = pd.read_csv(root / 'evaluation/test' / f'selected_pipeline_s11_{stage}.csv.gz')
        assert saved[['run_id', 'invocation_id']].equals(predicted[['run_id', 'invocation_id']])
        a, b = saved.pred.to_numpy(), predicted[f'pred_{column}_seconds'].to_numpy()
        differences[stage] = float(np.max(np.abs(a - b)))
        assert np.allclose(a, b, rtol=1e-6, atol=1e-4), (stage, differences[stage])

    # Re-select in a separate directory after ablations exist; preserve the original.
    resume = out / 'selection_resume'
    resume.mkdir(exist_ok=True)
    for name in ['results', 'cache']:
        link = resume / name
        if not link.exists():
            link.symlink_to(root / name, target_is_directory=True)
    shutil.copy2(root / 'search_jobs.json', resume / 'search_jobs.json')
    with (resume / 'selection.log').open('w') as log:
        subprocess.run([sys.executable, str(root / 'code/select.py'), '--root', str(resume)],
                       env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    assert json.loads((resume / 'selection.json').read_text()) == selected

    original = root / 'results' / selected['network']['folders']['11']
    config = json.loads((original / 'result.json').read_text())['config']
    replay = out / 'network_seed11_retrain'
    if not (replay / 'result.json').exists():
        rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,uuid',
                                        '--format=csv,noheader'], text=True).splitlines()
        gpu = next(row.split(',') for row in rows if row.split(',')[0].strip() == args.gpu)
        assert '5880' in gpu[1]
        active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                          '--format=csv,noheader'], text=True)
        assert gpu[2].strip() not in active, 'GPU0 has another compute process; replay was not started'
        replay.mkdir(exist_ok=True)
        command = [sys.executable, str(root / 'code/train.py'), '--cache', str(root / 'cache'),
                   '--out', str(replay), '--gpu']
        for key in ['stage', 'kind', 'variant', 'width', 'lr', 'seed', 'epochs', 'batch', 'architecture']:
            command.extend(['--' + key, str(config[key])])
        with (replay / 'console.log').open('w') as log:
            subprocess.run(command, env=dict(env, CUDA_VISIBLE_DEVICES=args.gpu),
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    first = pd.read_csv(original / 'validation_predictions.csv.gz')
    second = pd.read_csv(replay / 'validation_predictions.csv.gz')
    assert first[['run_id', 'invocation_id']].equals(second[['run_id', 'invocation_id']])
    delta = float(np.max(np.abs(first.pred - second.pred)))
    assert np.allclose(first.pred, second.pred, rtol=0, atol=1e-6), delta
    a = torch.load(original / 'model.pt', map_location='cpu', weights_only=True)
    b = torch.load(replay / 'model.pt', map_location='cpu', weights_only=True)
    weights_equal = a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
    report = dict(standalone_calls=len(predicted), standalone_cpu_max_abs_difference_seconds=differences,
                  standalone_tolerance=dict(rtol=1e-6, atol_seconds=1e-4),
                  selection_resume_identical=True, original_trial=original.name,
                  replay_seed=11, replay_validation_max_abs_difference_seconds=delta,
                  replay_weights_bitwise_equal=weights_equal,
                  replay_validation_predictions_bitwise_equal=bool(np.array_equal(first.pred.to_numpy(), second.pred.to_numpy())),
                  result_use='Reproducibility check only; original selection, weights and evaluation unchanged')
    (out / 'reproducibility_checks.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
