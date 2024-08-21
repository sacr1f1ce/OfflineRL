import copy
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from const import EXP_ADV_MAX
from dataset import TensorBatch
from utils import soft_update
from loss import asymmetric_l2_loss


class ImplicitQLearning:
    def __init__(
        self,
        max_action: float,
        actor: nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        actor_lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        q_network: nn.Module,
        q_optimizer: torch.optim.Optimizer,
        v_network: nn.Module,
        v_optimizer: torch.optim.Optimizer,
        iql_tau: float = 0.9,
        beta: float = 10.0,
        gamma: float = 0.99,
        tau: float = 0.005,
        device: str = "cpu",
    ):
        self.max_action = max_action
        self.qf = q_network
        self.q_target = copy.deepcopy(self.qf).requires_grad_(False).to(device)
        self.vf = v_network
        self.actor = actor
        self.v_optimizer = v_optimizer
        self.q_optimizer = q_optimizer
        self.actor_optimizer = actor_optimizer
        self.actor_lr_scheduler = actor_lr_scheduler
        self.iql_tau = iql_tau
        self.beta = beta
        self.gamma = gamma
        self.tau = tau
        self.device = device

    def _update_v(self, observations, actions, log_dict) -> torch.Tensor:
        # Update value function
        with torch.no_grad():
            target_q = self.q_target(observations, actions)

        v = self.vf(observations)
        adv = target_q - v
        v_loss = asymmetric_l2_loss(adv, self.iql_tau)
        log_dict["value_loss"] = v_loss.item()
        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()
        return adv

    def _update_q(
        self,
        next_v: torch.Tensor,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        terminals: torch.Tensor,
        log_dict: Dict,
    ):
        targets = rewards + (1.0 - terminals.float()) * self.gamma * next_v.detach()
        qs = self.qf.both(observations, actions)
        q_loss = sum(F.mse_loss(q, targets) for q in qs) / len(qs)
        log_dict["q_loss"] = q_loss.item()
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # Update target Q network
        soft_update(self.q_target, self.qf, self.tau)

    def _update_policy(
        self,
        adv: torch.Tensor,
        observations: torch.Tensor,
        actions: torch.Tensor,
        log_dict: Dict,
    ):
        exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=EXP_ADV_MAX)
        policy_out = self.actor(observations)
        if isinstance(policy_out, torch.distributions.Distribution):
            bc_losses = -policy_out.log_prob(actions).sum(-1, keepdim=False)
        elif torch.is_tensor(policy_out):
            if policy_out.shape != actions.shape:
                raise RuntimeError("Actions shape missmatch")
            bc_losses = torch.sum((policy_out - actions) ** 2, dim=1)
        else:
            raise NotImplementedError
        policy_loss = torch.mean(exp_adv * bc_losses)
        log_dict["actor_loss"] = policy_loss.item()
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
        self.actor_lr_scheduler.step()

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        (
            observations,
            actions,
            rewards,
            next_observations,
            dones,
        ) = batch
        log_dict = {}

        with torch.no_grad():
            next_v = self.vf(next_observations)
        # Update value function
        adv = self._update_v(observations, actions, log_dict)
        rewards = rewards.squeeze(dim=-1)
        dones = dones.squeeze(dim=-1)
        # Update Q function
        self._update_q(next_v, observations, actions, rewards, dones, log_dict)
        # Update actor
        self._update_policy(adv, observations, actions, log_dict)

        return log_dict

    def state_dict(self) -> Dict[str, Any]:
        return {
            "qf": self.qf.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "vf": self.vf.state_dict(),
            "v_optimizer": self.v_optimizer.state_dict(),
            "actor": self.actor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "actor_lr_scheduler": self.actor_lr_scheduler.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.qf.load_state_dict(state_dict["qf"])
        self.q_optimizer.load_state_dict(state_dict["q_optimizer"])
        self.q_target = copy.deepcopy(self.qf)

        self.vf.load_state_dict(state_dict["vf"])
        self.v_optimizer.load_state_dict(state_dict["v_optimizer"])

        self.actor.load_state_dict(state_dict["actor"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.actor_lr_scheduler.load_state_dict(state_dict["actor_lr_scheduler"])
