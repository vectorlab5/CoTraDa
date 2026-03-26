#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.training import Trainer
from src.utils import load_config, setup_logging, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CooperativeTrafficData")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment yaml")
    parser.add_argument("--base-config", type=str, default="configs/default.yaml", help="Path to default yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.base_config)
    set_seed(int(cfg["experiment"]["seed"]))

    exp_name = f"{cfg['experiment']['name']}_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir = Path(cfg["experiment"]["output_dir"]) / exp_name
    setup_logging(out_dir / "logs" / "train.log")

    trainer = Trainer(cfg, out_dir)
    metrics_path = trainer.train()
    print(f"Training complete. Metrics: {metrics_path}")


if __name__ == "__main__":
    main()

