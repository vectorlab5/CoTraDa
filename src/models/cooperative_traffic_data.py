from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import torch
import torch.nn as nn

from src.data.simulator import SlotState


@dataclass
class RolloutStep:
    obs: torch.Tensor
    action: torch.Tensor
    logprob: torch.Tensor
    value: torch.Tensor
    reward: float
    done: bool


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: list[int], out_dim: int):
        super().__init__()
        layers: list[nn.Module] = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        layers.append(nn.Linear(last, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CooperativeTrafficData(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        model_cfg = config["model"]
        policy_cfg = config["policy"]
        self.feature_dim = int(model_cfg["feature_dim"])
        self.alpha = float(model_cfg["alpha"])
        self.beta_cost = float(model_cfg["beta_cost"])
        self.lambda_rho = float(model_cfg["lambda_rho"])
        self.lambda_pi = float(model_cfg["lambda_pi"])
        self.lambda_aoi = float(model_cfg["lambda_aoi"])
        self.context_momentum = float(model_cfg["context_momentum"])
        self.trust_forgetting = float(model_cfg["trust_forgetting"])
        self.dual_step_size = float(model_cfg["dual_step_size"])
        self.dual_iterations = int(model_cfg["dual_iterations"])
        self.max_hops = int(model_cfg["max_hops"])
        self.threshold_payload_bits = float(model_cfg["threshold_payload_bits"])
        self.rho_candidates = [float(x) for x in model_cfg["rho_candidates"]]
        self.w_q = nn.Parameter(torch.eye(self.feature_dim))
        self.q_network = MLP(self.feature_dim * 2 + 2, policy_cfg["q_hidden_dims"], 1)
        self.actor = MLP(self.feature_dim * 2 + 2, policy_cfg["hidden_dims"], len(self.rho_candidates))
        self.critic = MLP(self.feature_dim * 2 + 2, policy_cfg["hidden_dims"], 1)
        self.projectors = nn.ModuleDict()
        for rho in self.rho_candidates:
            bottleneck = max(1, int(self.feature_dim * (1.0 - rho)))
            self.projectors[self._rho_key(rho)] = nn.Sequential(
                nn.Linear(self.feature_dim, bottleneck),
                nn.ReLU(),
                nn.Linear(bottleneck, self.feature_dim),
            )
        self.reset_state()

    @staticmethod
    def _rho_key(rho: float) -> str:
        return f"rho_{rho:.2f}".replace(".", "_")

    def reset_state(self) -> None:
        self.global_context: np.ndarray | None = None
        self.trust_scores: np.ndarray | None = None
        self.dual_vars: Dict[tuple[str, str], float] = {}

    def _ensure_state(self, num_agents: int) -> None:
        if self.global_context is None:
            self.global_context = np.zeros(self.feature_dim, dtype=np.float32)
        if self.trust_scores is None:
            self.trust_scores = np.ones(num_agents, dtype=np.float32) * 0.7

    def _build_obs(self, slot: SlotState) -> np.ndarray:
        self._ensure_state(slot.features.shape[0])
        ctx = np.repeat(self.global_context[None, :], slot.features.shape[0], axis=0)
        return np.concatenate(
            [slot.features, ctx, slot.aoi[:, None], slot.compute_budget[:, None]],
            axis=1,
        ).astype(np.float32)

    def _admission(self, slot: SlotState, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        obs_t = torch.from_numpy(obs)
        content_logits = self.q_network(obs_t).squeeze(-1).detach().numpy()
        urgency = 1.0 - np.exp(-self.alpha * slot.aoi)
        expected_rho = 0.5
        comp_cost = expected_rho * (self.feature_dim / np.maximum(slot.compute_budget, 1e-3))
        q_values = urgency * self.trust_scores * (1.0 / (1.0 + np.exp(-content_logits))) - self.beta_cost * comp_cost
        est_capacity_agents = max(1.0, sum(slot.link_capacity.values()) / self.threshold_payload_bits)
        threshold = float(np.sum(q_values) / est_capacity_agents)
        admitted = (q_values >= threshold).astype(np.int32)
        return admitted, q_values, np.array([threshold], dtype=np.float32)

    def _select_compression(self, obs: np.ndarray, admitted: np.ndarray) -> tuple[np.ndarray, np.ndarray, torch.Tensor, torch.Tensor]:
        obs_t = torch.from_numpy(obs)
        logits = self.actor(obs_t)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        logprob = dist.log_prob(action)
        value = self.critic(obs_t).squeeze(-1)
        action_np = action.detach().numpy()
        rho = np.array([self.rho_candidates[i] for i in action_np], dtype=np.float32)
        rho[admitted == 0] = 0.0
        return rho, action_np, logprob, value

    def _routing(
        self,
        slot: SlotState,
        admitted: np.ndarray,
        rho: np.ndarray,
        agent_relay: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        g = nx.Graph()
        for edge, cap in slot.link_capacity.items():
            rel = slot.link_reliability[edge]
            u, v = edge
            dual = self.dual_vars.get(edge, 0.0)
            g.add_edge(u, v, cap=cap, rel=rel, dual=dual)
        payload = self.feature_dim * (1.0 - rho)
        delivery_success = np.zeros_like(admitted)
        latency_slots = np.ones_like(admitted)
        path_utils = np.zeros_like(rho, dtype=np.float32)
        provenance = np.zeros_like(rho, dtype=np.float32)

        start = time.perf_counter()
        for _ in range(self.dual_iterations):
            edge_load = {edge: 0.0 for edge in slot.link_capacity}
            for agent_idx in np.where(admitted == 1)[0]:
                src = f"relay_{agent_relay[agent_idx]}"
                for u, v, data in g.edges(data=True):
                    edge = tuple(sorted((u, v)))
                    trust_hop = 0.8 if "relay" in u and "relay" in v else 1.0
                    raw_cost = data["dual"] * payload[agent_idx] - data["rel"] * np.log(1.0 + data["cap"]) + self.lambda_pi * (
                        1.0 - trust_hop
                    )
                    # Dijkstra requires non-negative edge weights.
                    data["weight"] = float(max(1e-6, raw_cost + 10.0))
                try:
                    path = nx.shortest_path(g, source=src, target="fusion", weight="weight")
                except nx.NetworkXNoPath:
                    continue
                path_edges = [tuple(sorted((path[i], path[i + 1]))) for i in range(len(path) - 1)]
                cap_min = min(slot.link_capacity[e] for e in path_edges)
                rel_prod = np.prod([slot.link_reliability[e] for e in path_edges])
                delivery_success[agent_idx] = int(rel_prod > 0.4 and payload[agent_idx] <= cap_min)
                latency_slots[agent_idx] = min(self.max_hops, len(path_edges))
                path_utils[agent_idx] = sum(slot.link_reliability[e] * np.log(1.0 + slot.link_capacity[e]) for e in path_edges)
                provenance[agent_idx] = sum((1.0 - 0.8) for _ in path_edges)
                for e in path_edges:
                    edge_load[e] += payload[agent_idx]

            for edge in slot.link_capacity:
                mu = self.dual_vars.get(edge, 0.0)
                mu = max(0.0, mu + self.dual_step_size * (edge_load[edge] - slot.link_capacity[edge]))
                self.dual_vars[edge] = mu
                if g.has_edge(*edge):
                    g.edges[edge]["dual"] = mu
        slot_ms = (time.perf_counter() - start) * 1000.0
        return delivery_success, latency_slots, path_utils, provenance, np.array([slot_ms], dtype=np.float32)

    def _update_feedback(
        self,
        slot: SlotState,
        admitted: np.ndarray,
        delivery_success: np.ndarray,
        rho: np.ndarray,
    ) -> None:
        delivered_idx = np.where((admitted == 1) & (delivery_success == 1))[0]
        if delivered_idx.size > 0:
            compressed = []
            for idx in delivered_idx:
                key = self._rho_key(float(rho[idx]))
                if key not in self.projectors:
                    key = self._rho_key(0.0)
                x = torch.from_numpy(slot.features[idx]).float()
                with torch.no_grad():
                    compressed.append(self.projectors[key](x).numpy())
            compressed_arr = np.stack(compressed, axis=0)
            weights = np.exp(-self.alpha * slot.aoi[delivered_idx])
            weights = weights / np.clip(weights.sum(), 1e-6, None)
            target_ctx = (compressed_arr * weights[:, None]).sum(axis=0)
            self.global_context = (1.0 - self.context_momentum) * self.global_context + self.context_momentum * target_ctx
        consistency = np.ones_like(delivery_success, dtype=np.float32)
        self.trust_scores = (1.0 - self.trust_forgetting) * self.trust_scores + self.trust_forgetting * (delivery_success * consistency)

    def forward_slot(self, slot: SlotState, agent_relay: np.ndarray) -> tuple[dict, List[RolloutStep], dict]:
        obs = self._build_obs(slot)
        admitted, q_values, threshold = self._admission(slot, obs)
        rho, action_idx, logprob, value = self._select_compression(obs, admitted)
        delivery_success, latency_slots, path_utils, provenance, slot_time_ms = self._routing(slot, admitted, rho, agent_relay)

        freshness = np.exp(-self.alpha * slot.aoi)
        utility = freshness * admitted * path_utils - self.lambda_rho * rho - self.lambda_pi * provenance
        reward = float(np.mean(utility) - self.lambda_aoi * np.mean(np.maximum(0.0, slot.aoi - 5.0)))
        self._update_feedback(slot, admitted, delivery_success, rho)

        rollouts = []
        obs_t = torch.from_numpy(obs)
        for i in range(obs.shape[0]):
            rollouts.append(
                RolloutStep(
                    obs=obs_t[i],
                    action=torch.tensor(action_idx[i], dtype=torch.long),
                    logprob=logprob[i].detach(),
                    value=value[i].detach(),
                    reward=reward,
                    done=False,
                )
            )

        actions = {
            "admitted": admitted,
            "rho": rho,
            "delivery_success": delivery_success,
            "latency_slots": latency_slots,
        }
        info = {
            "reward": reward,
            "weighted_aoi": float(np.mean(slot.aoi * freshness)),
            "data_utility": float(np.mean(utility)),
            "slot_time_ms": float(slot_time_ms[0]),
            "admitted_count": int(admitted.sum()),
            "threshold": float(threshold[0]),
            "avg_q": float(np.mean(q_values)),
        }
        return actions, rollouts, info

