from pathlib import Path

import torch

from src.data import V2XEnvironment
from src.models import CooperativeTrafficData
from src.utils import load_config


def test_single_slot_forward():
    cfg = load_config(
        Path("configs/experiment_main.yaml"),
        Path("configs/default.yaml"),
    )
    cfg["experiment"]["episodes"] = 1
    cfg["experiment"]["slots_per_episode"] = 2
    env = V2XEnvironment(cfg)
    model = CooperativeTrafficData(cfg)
    slot = env.reset()
    actions, rollouts, info = model.forward_slot(slot, env.agent_relay)
    assert actions["admitted"].shape[0] == env.spec.num_agents
    assert actions["rho"].shape[0] == env.spec.num_agents
    assert len(rollouts) == env.spec.num_agents
    assert "data_utility" in info
    assert "weighted_aoi" in info

