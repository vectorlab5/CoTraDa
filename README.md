# CooperativeTrafficData

This repository contains a full implementation of a freshness-aware V2X data pipeline with four coupled stages: admission gating, adaptive compression, reliability-weighted routing, and delivery feedback.

## What this code does

The code runs slot-based traffic data collection experiments where many agents compete for limited communication links. At each slot, the model decides:
- which agents should transmit now,
- how much each selected feature vector should be compressed,
- which multi-hop route to use,
- and how to update reliability scores from delivery outcomes.

The training loop optimizes these decisions with PPO while routing is solved with warm-started dual updates.

## Project layout

- `configs/`: experiment settings and default hyperparameters
- `src/data/`: scenario definitions and V2X simulator
- `src/models/`: CooperativeTrafficData model and slot-level decision logic
- `src/training/`: PPO update and training loop
- `src/evaluation/`: evaluation runner and metric export
- `scripts/`: command-line entrypoints
- `figures/scripts/`: plotting utilities

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train

```bash
python scripts/train.py --config configs/experiment_main.yaml
```

## Evaluate

```bash
python scripts/evaluate.py \
  --checkpoint experiments/<run_id>/checkpoints/best.pt \
  --episodes 10 \
  --output experiments/<run_id>/results/eval_metrics.json
```

## Reproduce end-to-end

```bash
bash scripts/run_all_experiments.sh
```

## Notes

- Random seeds are fixed through config for reproducibility.
- All experiment outputs are written under `experiments/`.
- `figures/scripts/` can be used to regenerate the plots from exported results.
