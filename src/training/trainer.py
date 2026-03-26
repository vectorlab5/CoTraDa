from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch

from src.data.simulator import V2XEnvironment
from src.models.cooperative_traffic_data import CooperativeTrafficData, RolloutStep
from src.training.ppo import PPOConfig, ppo_update


class Trainer:
    def __init__(self, config: dict, output_dir: Path):
        self.cfg = config
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)
        self.env = V2XEnvironment(config)
        self.model = CooperativeTrafficData(config)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(config["policy"]["lr"]))
        self.ppo_cfg = PPOConfig(
            gamma=float(config["policy"]["gamma"]),
            gae_lambda=float(config["policy"]["gae_lambda"]),
            clip_eps=float(config["policy"]["clip_eps"]),
            ppo_epochs=int(config["policy"]["ppo_epochs"]),
            minibatch_size=int(config["policy"]["minibatch_size"]),
        )
        self.best_reward = -1e9

    def _run_episode(self) -> dict:
        slot_state = self.env.reset()
        self.model.reset_state()
        rollouts: list[RolloutStep] = []
        episode_metrics = {
            "reward": [],
            "weighted_aoi": [],
            "data_utility": [],
            "delivery_ratio": [],
            "throughput_mbps": [],
            "avg_latency_ms": [],
            "slot_time_ms": [],
        }
        for _ in range(int(self.cfg["experiment"]["slots_per_episode"])):
            actions, step_rollouts, info = self.model.forward_slot(slot_state, self.env.agent_relay)
            slot_state, env_metrics = self.env.step(actions)
            rollouts.extend(step_rollouts)
            episode_metrics["reward"].append(info["reward"])
            episode_metrics["weighted_aoi"].append(info["weighted_aoi"])
            episode_metrics["data_utility"].append(info["data_utility"])
            episode_metrics["slot_time_ms"].append(info["slot_time_ms"])
            episode_metrics["delivery_ratio"].append(env_metrics["delivery_ratio"])
            episode_metrics["throughput_mbps"].append(env_metrics["throughput_mbps"])
            episode_metrics["avg_latency_ms"].append(env_metrics["avg_latency_ms"])

        losses = ppo_update(self.model, self.optimizer, rollouts, self.ppo_cfg)
        out = {k: float(np.mean(v)) for k, v in episode_metrics.items()}
        out.update(losses)
        return out

    def train(self) -> Path:
        results = []
        n_episodes = int(self.cfg["experiment"]["episodes"])
        for ep in range(1, n_episodes + 1):
            metrics = self._run_episode()
            results.append({"episode": ep, **metrics})
            if metrics["reward"] > self.best_reward:
                self.best_reward = metrics["reward"]
                self._save_checkpoint(self.output_dir / "checkpoints" / "best.pt", ep, metrics)
            if ep % 10 == 0 or ep == 1:
                self.logger.info(
                    "ep=%d reward=%.4f utility=%.4f w_aoi=%.4f del=%.3f slot=%.2fms",
                    ep,
                    metrics["reward"],
                    metrics["data_utility"],
                    metrics["weighted_aoi"],
                    metrics["delivery_ratio"],
                    metrics["slot_time_ms"],
                )
        self._save_checkpoint(self.output_dir / "checkpoints" / "last.pt", n_episodes, results[-1])
        results_path = self.output_dir / "results" / "train_metrics.json"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        return results_path

    def _save_checkpoint(self, path: Path, episode: int, metrics: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "episode": episode,
                "metrics": metrics,
                "config": self.cfg,
                "state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )

