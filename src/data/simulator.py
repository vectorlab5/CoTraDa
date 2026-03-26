from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import networkx as nx
import numpy as np

from src.data.scenario import SCENARIOS


@dataclass
class SlotState:
    features: np.ndarray
    aoi: np.ndarray
    compute_budget: np.ndarray
    link_capacity: Dict[tuple[int, int], float]
    link_reliability: Dict[tuple[int, int], float]


class V2XEnvironment:
    def __init__(self, config: dict):
        scenario_name = config["experiment"]["scenario"]
        self.spec = SCENARIOS[scenario_name]
        self.cfg = config
        self.rng = np.random.default_rng(config["experiment"]["seed"])
        self.feature_dim = int(config["model"]["feature_dim"])
        self.slot_ms = 100.0
        self.deadline_slots = int(config["simulator"]["deadline_slots"])
        self.base_failure = float(config["simulator"]["base_failure_probability"])
        self.noise_std = float(config["simulator"]["feature_noise_std"])
        self.rsu_count = int(config["simulator"]["rsu_count"])
        self._build_graph()
        self.reset()

    def _build_graph(self) -> None:
        self.graph = nx.Graph()
        self.fusion = "fusion"
        relay_nodes = [f"relay_{i}" for i in range(self.spec.num_relays)]
        for relay in relay_nodes:
            self.graph.add_node(relay)
        self.graph.add_node(self.fusion)
        for relay in relay_nodes:
            self.graph.add_edge(relay, self.fusion)
        for i, src in enumerate(relay_nodes):
            for dst in relay_nodes[i + 1 :]:
                if self.rng.uniform() < 0.4:
                    self.graph.add_edge(src, dst)
        self.relay_nodes = relay_nodes

    def reset(self) -> SlotState:
        self.t = 0
        self.aoi = np.ones(self.spec.num_agents, dtype=np.float32)
        self.last_delivery_slot = np.zeros(self.spec.num_agents, dtype=np.int32)
        self.agent_relay = self.rng.integers(0, self.spec.num_relays, size=self.spec.num_agents)
        self.base_features = self.rng.normal(0.0, 1.0, size=(self.spec.num_agents, self.feature_dim)).astype(np.float32)
        return self.current_state()

    def _sample_link_states(self) -> tuple[Dict[tuple[int, int], float], Dict[tuple[int, int], float]]:
        capacities: Dict[tuple[int, int], float] = {}
        reliabilities: Dict[tuple[int, int], float] = {}
        for u, v in self.graph.edges():
            edge = tuple(sorted((u, v)))
            cap = self.rng.uniform(40.0, 120.0)
            rel = np.clip(1.0 - self.base_failure - self.rng.uniform(0.0, 0.2), 0.05, 0.99)
            capacities[edge] = float(cap)
            reliabilities[edge] = float(rel)
        return capacities, reliabilities

    def current_state(self) -> SlotState:
        capacities, reliabilities = self._sample_link_states()
        features = self.base_features + self.rng.normal(0.0, self.noise_std, size=self.base_features.shape)
        compute_budget = self.rng.uniform(8.0, 20.0, size=self.spec.num_agents).astype(np.float32)
        return SlotState(
            features=features.astype(np.float32),
            aoi=self.aoi.copy(),
            compute_budget=compute_budget,
            link_capacity=capacities,
            link_reliability=reliabilities,
        )

    def step(self, actions: dict) -> tuple[SlotState, dict]:
        self.t += 1
        admitted = actions["admitted"]
        selected_rho = actions["rho"]
        latencies = actions["latency_slots"]
        delivery_success = actions["delivery_success"]

        throughput_bits = 0.0
        for idx in range(self.spec.num_agents):
            self.aoi[idx] += 1.0
            if admitted[idx]:
                payload_dim = self.feature_dim * (1.0 - selected_rho[idx])
                throughput_bits += payload_dim * 32.0
                if delivery_success[idx]:
                    self.aoi[idx] = float(max(1, latencies[idx]))
                    self.last_delivery_slot[idx] = self.t

        metrics = {
            "throughput_mbps": throughput_bits / 1e6 / (self.slot_ms / 1000.0),
            "delivery_ratio": float(delivery_success.sum() / max(1, admitted.sum())),
            "avg_latency_ms": float(np.mean(latencies[admitted == 1]) * self.slot_ms if admitted.sum() else 0.0),
        }
        return self.current_state(), metrics

