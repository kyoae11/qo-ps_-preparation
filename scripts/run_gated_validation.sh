#!/usr/bin/env bash
set -euo pipefail

gpu="${1:-0}"
epochs="${2:-50}"
result_dir="outputs/results/first_comparison"

export PYTHONPATH="$PWD/src:$PWD"
mkdir -p outputs/logs/gated_multiseed "$result_dir"

python tools/check_entity_predictability.py

for seed in 0 1 2 3 4; do
  for qos_type in rt tp; do
    model="entity_gated"
    result_file="$result_dir/${model}_${qos_type}_md0.1_seed${seed}.json"
    if [[ -s "$result_file" ]]; then
      echo "[skip] $model $qos_type seed=$seed"
      continue
    fi
    log_file="outputs/logs/gated_multiseed/${model}_${qos_type}_seed${seed}.log"
    echo "[start] $(date '+%F %T') $model $qos_type seed=$seed gpu=$gpu"
    CUDA_VISIBLE_DEVICES="$gpu" python -m experiments.run_comparison \
      --model "$model" --type "$qos_type" --md 0.10 --seed "$seed" \
      --epochs "$epochs" --cuda 0 2>&1 | tee "$log_file"
    echo "[done] $(date '+%F %T') $model $qos_type seed=$seed"
  done
done

python tools/summarize_comparison.py
echo "[complete] $(date '+%F %T') gated multiseed validation finished"
