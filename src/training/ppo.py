from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

from src.models.cooperative_traffic_data import RolloutStep, CooperativeTrafficData


@dataclass
class PPOConfig:
    gamma: float
    gae_lambda: float
    clip_eps: float
    ppo_epochs: int
    minibatch_size: int


def _compute_gae(rollouts: List[RolloutStep], cfg: PPOConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes Generalized Advantage Estimation (GAE) to help stabilize training.
    Works backwards through the rollout to calculate multi-step returns.
    """
    rewards = torch.tensor([r.reward for r in rollouts], dtype=torch.float32)
    values = torch.stack([r.value for r in rollouts]).float()
    dones = torch.tensor([r.done for r in rollouts], dtype=torch.float32)

    advantages = torch.zeros_like(rewards)
    gae = 0.0
    next_value = 0.0
    for t in reversed(range(len(rollouts))):
        # Lookahead error (delta): r + gamma * V(s') - V(s)
        delta = rewards[t] + cfg.gamma * next_value * (1.0 - dones[t]) - values[t]
        # Decay GAE based on both gamma and lambda
        gae = delta + cfg.gamma * cfg.gae_lambda * (1.0 - dones[t]) * gae
        advantages[t] = gae
        next_value = values[t]
    
    returns = advantages + values
    # Standardize advantages so they have zero mean and unit variance
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)
    return returns.detach(), advantages.detach()


def ppo_update(
    model: CooperativeTrafficData,
    optimizer: torch.optim.Optimizer,
    rollouts: List[RolloutStep],
    cfg: PPOConfig,
) -> dict:
    """
    Standard PPO actor-critic update.
    Iterates through shuffled minibatches to update the policy and value networks.
    """
    returns, advantages = _compute_gae(rollouts, cfg)
    obs = torch.stack([r.obs for r in rollouts]).float()
    actions_adm = torch.stack([r.action_adm for r in rollouts]).float()
    actions_rho = torch.stack([r.action_rho for r in rollouts]).long()
    old_logprobs = torch.stack([r.logprob for r in rollouts]).float()

    idx = np.arange(len(rollouts))
    losses = []
    for _ in range(cfg.ppo_epochs):
        np.random.shuffle(idx)
        for start in range(0, len(idx), cfg.minibatch_size):
            mb = idx[start : start + cfg.minibatch_size]
            mb_obs = obs[mb]
            mb_actions_adm = actions_adm[mb]
            mb_actions_rho = actions_rho[mb]
            mb_old_logprobs = old_logprobs[mb]
            mb_returns = returns[mb]
            mb_advantages = advantages[mb]

            # Re-evaluate the policy with new parameters
            logits = model.actor(mb_obs)
            adm_logits = logits[:, 0]
            rho_logits = logits[:, 1:]
            
            adm_dist = torch.distributions.Bernoulli(logits=adm_logits)
            rho_dist = torch.distributions.Categorical(logits=rho_logits)
            
            new_adm_logprob = adm_dist.log_prob(mb_actions_adm)
            new_rho_logprob = rho_dist.log_prob(mb_actions_rho)
            new_logprobs = new_adm_logprob + new_rho_logprob
            
            # PPO actor loss (clipping avoids too large updates)
            ratio = torch.exp(new_logprobs - mb_old_logprobs)
            clipped = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * mb_advantages
            actor_loss = -torch.min(ratio * mb_advantages, clipped).mean()
            
            # Value function loss (Mean Squared Error)
            value = model.critic(mb_obs).squeeze(-1)
            critic_loss = F.mse_loss(value, mb_returns)
            
            # Entropy bonus to encourage exploration
            entropy = (adm_dist.entropy() + rho_dist.entropy()).mean()
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping keeps the update stable
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(float(loss.item()))

    return {"ppo_loss": float(np.mean(losses) if losses else 0.0)}

