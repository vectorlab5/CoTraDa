from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from src.data.simulator import V2XEnvironment
from src.models.cooperative_traffic_data import CooperativeTrafficData


class Evaluator:
    def __init__(self, checkpoint_path: Path):
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        self.cfg = ckpt["config"]
        self.model = CooperativeTrafficData(self.cfg)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.env = V2XEnvironment(self.cfg)

    def run(self, episodes: int = 10) -> dict:
        metrics = {
            "reward": [],
            "weighted_aoi": [],
            "data_utility": [],
            "delivery_ratio": [],
            "throughput_mbps": [],
            "avg_latency_ms": [],
            "slot_time_ms": [],
        }
        for _ in range(episodes):
            slot_state = self.env.reset()
            self.model.reset_state()
            for _ in range(int(self.cfg["experiment"]["slots_per_episode"])):
                with torch.no_grad():
                    actions, _, info = self.model.forward_slot(slot_state, self.env.agent_relay)
                slot_state, env_metrics = self.env.step(actions)
                metrics["reward"].append(info["reward"])
                metrics["weighted_aoi"].append(info["weighted_aoi"])
                metrics["data_utility"].append(info["data_utility"])
                metrics["slot_time_ms"].append(info["slot_time_ms"])
                metrics["delivery_ratio"].append(env_metrics["delivery_ratio"])
                metrics["throughput_mbps"].append(env_metrics["throughput_mbps"])
                metrics["avg_latency_ms"].append(env_metrics["avg_latency_ms"])

        return {k: float(np.mean(v)) for k, v in metrics.items()}

    @staticmethod
    def save(results: dict, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(results, indent=2), encoding="utf-8")

