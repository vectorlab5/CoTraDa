#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.evaluation import Evaluator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CooperativeTrafficData")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--episodes", type=int, default=10, help="Number of eval episodes")
    parser.add_argument("--output", type=str, default="experiments/eval/results.json", help="Path to output json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluator = Evaluator(Path(args.checkpoint))
    results = evaluator.run(episodes=args.episodes)
    Evaluator.save(results, Path(args.output))
    print("Evaluation complete:")
    for k, v in results.items():
        print(f"  {k}: {v:.6f}")


if __name__ == "__main__":
    main()

