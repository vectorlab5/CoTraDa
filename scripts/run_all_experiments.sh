#!/usr/bin/env bash
set -euo pipefail

python scripts/train.py --config configs/experiment_main.yaml

LATEST_EXP="$(ls -td experiments/main_comparison_* | head -n 1)"
python scripts/evaluate.py \
  --checkpoint "${LATEST_EXP}/checkpoints/best.pt" \
  --episodes 10 \
  --output "${LATEST_EXP}/results/eval_metrics.json"

echo "All experiments finished: ${LATEST_EXP}"

