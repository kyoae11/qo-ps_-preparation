#!/usr/bin/env bash
set -euo pipefail

gpu="${1:-0}"
epochs="${2:-50}"

export PYTHONPATH="$PWD/src:$PWD"
python -m qops_preparation.simulation --mode full

for qos_type in rt tp; do
  for model in baseline entity_concat; do
    CUDA_VISIBLE_DEVICES="$gpu" python -m experiments.run_comparison \
      --model "$model" \
      --type "$qos_type" \
      --md 0.10 \
      --seed 0 \
      --epochs "$epochs" \
      --cuda 0
  done
done

python tools/summarize_comparison.py
