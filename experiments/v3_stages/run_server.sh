#!/usr/bin/env bash
set -euo pipefail
ROOT=${QOPS_EXPERIMENT_ROOT:-/data/qops_lxy_20261118}
GPUS=${1:?Provide one or two freshly checked RTX5880 GPU indices; GPU2 is prohibited}
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
"$ROOT/env/bin/python" - "$GPUS" <<'PY'
import subprocess,sys
ids=sys.argv[1].split(',')
assert len(ids) in (1,2) and len(set(ids))==len(ids)
assert '2' not in ids,'RTX6000D is excluded by the user'
g=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,uuid','--format=csv,noheader'],text=True)
active=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
rows={a.strip():(b.strip(),c.strip()) for a,b,c in (line.split(',') for line in g.strip().splitlines())}
for i in ids:
 name,uuid=rows[i]
 assert '5880' in name,'Only RTX5880 permitted'
 assert uuid not in active,f'GPU{i} already has a compute process; do not start'
print('Selected GPUs have no compute processes:',ids)
PY
cd "$ROOT"
"$ROOT/env/bin/python" code/validate.py --root data --cache cache --manifest data_source_hashes.json --out pretraining_checks.json
"$ROOT/env/bin/python" code/orchestrate.py --root "$ROOT" --gpus "$GPUS" --workers-per-gpu 2 --phase search
"$ROOT/env/bin/python" code/select.py --root "$ROOT" > selection.log
"$ROOT/env/bin/python" code/orchestrate.py --root "$ROOT" --gpus "$GPUS" --workers-per-gpu 2 --phase ablation
CUDA_VISIBLE_DEVICES="${GPUS%%,*}" "$ROOT/env/bin/python" code/evaluate.py --root "$ROOT" > evaluation.log
"$ROOT/env/bin/python" code/verify_results.py --root "$ROOT" > verification.log
