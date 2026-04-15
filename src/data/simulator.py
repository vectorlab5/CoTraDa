from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import networkx as nx
import numpy as np

from src.data.scenario import SCENARIOS
from src.data.pneuma_replayer import PneumaReplayer


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
        
        if scenario_name == "pneuma_athens":
            # Assume data path is in config or same dir
            data_path = config["simulator"].get("pneuma_path", "data/pneuma_athens.csv")
            self.replayer = PneumaReplayer(data_path, self.spec.area_m)
        else:
            self.replayer = None
            
        self._build_graph()
        self.reset()

    def _build_graph(self) -> None:
        self.graph = nx.Graph()
        self.fusion = "fusion"
        self.relay_nodes = [f"relay_{i}" for i in range(self.spec.num_relays)]
        
        # Place relays in a grid across the defined area
        grid_size = int(np.ceil(np.sqrt(len(self.relay_nodes))))
        self.relay_coords = {}
        for i, relay in enumerate(self.relay_nodes):
            xi = (i % grid_size) * (self.spec.area_m[0] / max(1, grid_size-1))
            yi = (i // grid_size) * (self.spec.area_m[1] / max(1, grid_size-1))
            self.relay_coords[relay] = np.array([xi, yi], dtype=np.float32)
            self.graph.add_node(relay)
        
        self.graph.add_node(self.fusion)
        # Place the fusion center in the middle of the field
        self.fusion_coords = np.array([self.spec.area_m[0]/2, self.spec.area_m[1]/2], dtype=np.float32)

        # Build the network topology: connect relays to fusion and near neighbors
        for relay in self.relay_nodes:
            self.graph.add_edge(relay, self.fusion)
        for i, src in enumerate(self.relay_nodes):
            for dst in self.relay_nodes[i + 1 :]:
                dist = np.linalg.norm(self.relay_coords[src] - self.relay_coords[dst])
                if dist < self.spec.area_m[0] / 2:
                    self.graph.add_edge(src, dst)

    def reset(self) -> SlotState:
        self.t = 0
        self.aoi = np.ones(self.spec.num_agents, dtype=np.float32)
        self.last_delivery_slot = np.zeros(self.spec.num_agents, dtype=np.int32)
        
        if self.replayer:
            # Use real-world coordinates from the replayer (e.g., pNEUMA)
            self.agent_coords, _ = self.replayer.get_step(0, self.spec.num_agents)
            self.agent_vel = np.zeros((self.spec.num_agents, 2), dtype=np.float32)
        else:
            # Set up agents with random positions and velocities
            self.agent_coords = self.rng.uniform(0, self.spec.area_m[0], size=(self.spec.num_agents, 2)).astype(np.float32)
            self.agent_vel = self.rng.uniform(self.spec.speed_kmh[0]/3.6, self.spec.speed_kmh[1]/3.6, size=(self.spec.num_agents, 2)).astype(np.float32)
            
        self.agent_relay = np.zeros(self.spec.num_agents, dtype=np.int32)
        self._update_agent_relay_mapping()
        
        self.base_features = self.rng.normal(0.0, 1.0, size=(self.spec.num_agents, self.feature_dim)).astype(np.float32)
        return self.current_state()

    def _update_agent_relay_mapping(self) -> None:
        """Assign each agent to the nearest relay."""
        for i in range(self.spec.num_agents):
            dists = [np.linalg.norm(self.agent_coords[i] - self.relay_coords[r]) for r in self.relay_nodes]
            self.agent_relay[i] = np.argmin(dists)

    def _sample_link_states(self) -> tuple[Dict[tuple[str, str], float], Dict[tuple[str, str], float]]:
        capacities: Dict[tuple[str, str], float] = {}
        reliabilities: Dict[tuple[str, str], float] = {}
        
        # Model the differences between sub-6GHz and mmWave links
        for u, v in self.graph.edges():
            edge = tuple(sorted((u, v)))
            if u == self.fusion or v == self.fusion:
                dist = 0.0 # High-speed backbone
                base_cap = 500.0
                rel = 0.99
            else:
                dist = np.linalg.norm(self.relay_coords[u] - self.relay_coords[v])
                # mmWave scenario: high throughput but sensitive to blockage at distance
                if dist < 100:
                    base_cap = 400.0 
                    rel = np.clip(1.0 - self.base_failure - self.rng.uniform(0.1, 0.3), 0.05, 0.95)
                else:
                    base_cap = 100.0 # Conventional sub-6GHz
                    rel = np.clip(1.0 - self.base_failure - self.rng.uniform(0.0, 0.1), 0.5, 0.99)
            
            capacities[edge] = float(base_cap * self.rng.uniform(0.8, 1.2))
            reliabilities[edge] = float(rel)
        return capacities, reliabilities

    def current_state(self) -> SlotState:
        # Move vehicles based on their velocity
        if self.replayer:
            self.agent_coords, _ = self.replayer.get_step(self.t, self.spec.num_agents)
        else:
            self.agent_coords += self.agent_vel * (self.slot_ms / 1000.0)
            # Boundary check: bounce off walls
            for i in range(2):
                mask_min = self.agent_coords[:, i] < 0
                mask_max = self.agent_coords[:, i] > self.spec.area_m[i]
                self.agent_vel[mask_min | mask_max, i] *= -1.0
                self.agent_coords[:, i] = np.clip(self.agent_coords[:, i], 0, self.spec.area_m[i])
        
        self._update_agent_relay_mapping()
        capacities, reliabilities = self._sample_link_states()
        
        # Evolve vehicle features over time
        self.base_features += self.rng.normal(0.0, 0.05, size=self.base_features.shape)
        features = self.base_features + self.rng.normal(0.0, self.noise_std, size=self.base_features.shape)
        
        # Variance in local compute resources
        compute_budget = self.rng.uniform(8.0, 25.0, size=self.spec.num_agents).astype(np.float32)
        
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
            self.aoi[idx] += 1.0 # Standard AoI increment
            if admitted[idx]:
                payload_dim = self.feature_dim * (1.0 - selected_rho[idx])
                throughput_bits += payload_dim * 32.0 # 32-bit floats
                if delivery_success[idx]:
                    # Update freshness: AoI is reset based on current path delay
                    self.aoi[idx] = float(latencies[idx])
                    self.last_delivery_slot[idx] = self.t

        metrics = {
            "throughput_mbps": throughput_bits / 1e6 / (self.slot_ms / 1000.0),
            "delivery_ratio": float(delivery_success.sum() / max(1, admitted.sum())),
            "avg_latency_ms": float(np.mean(latencies[admitted == 1]) * self.slot_ms if admitted.sum() else 0.0),
            "weighted_aoi": float(np.mean(self.aoi)),
        }
        return self.current_state(), metrics

