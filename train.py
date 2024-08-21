import os
# the following lines are needed for arm macs
os.environ['CC'] = '/opt/homebrew/opt/llvm/bin/clang'
os.environ['CXX'] = '/opt/homebrew/opt/llvm/bin/clang'
os.environ['CPPFLAGS'] = "-I/opt/homebrew/opt/llvm/include"
os.environ['LDFLAGS'] = "-L/opt/homebrew/opt/llvm/lib"
os.environ['WANDB_SILENT'] = "true"

import contextlib
import uuid
from typing import Dict
from dataclasses import asdict

import minari
import gymnasium as gym
import pyrallis
import torch
import wandb
import numpy as np
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import trange

from dataset import ReplayBuffer
from models import TwinQ, ValueFunction, DeterministicPolicy, GaussianPolicy
from iql import ImplicitQLearning
from config import TrainConfig
from utils import (
    qlearning_dataset,
    modify_reward,
    compute_mean_std,
    normalize_states,
    wrap_env,
    set_seed
)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def evaluate(
    env: gym.Env, actor: nn.Module, num_episodes: int, seed: int, device: str
) -> np.ndarray:
    def compute_distance_to_goal(state: Dict[str, torch.Tensor]) -> float:
        return ((state['desired_goal'] - state['achieved_goal'])**2).sum()

    actor.eval()
    episode_rewards = []
    episode_l2_distance = []
    for i in range(num_episodes):
        done = False
        state, info = env.reset(seed=seed + i)
        episode_reward = 0.0
        while not done:
            action = actor.act(state['observation'], device)
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
        episode_rewards.append(episode_reward)
        episode_l2_distance.append(compute_distance_to_goal(state))

    actor.train()
    return np.asarray(episode_rewards), np.asarray(episode_l2_distance)


@pyrallis.wrap()  # config_path='./config.yaml')
def train(config: TrainConfig):
    wandb.init(
        config=asdict(config),
        project=config.project,
        name=config.name,
        id=str(uuid.uuid4()),
        save_code=True,
    )
    minari.download_dataset(config.dataset_id)
    dataset = minari.load_dataset(config.dataset_id)

    eval_env = dataset.recover_environment(eval_env=True)
    state_dim = eval_env.observation_space['observation'].shape[0]
    action_dim = eval_env.action_space.shape[0]
    max_action = float(eval_env.action_space.high[0])

    qdataset = qlearning_dataset(dataset)
    if config.normalize_reward:
        modify_reward(qdataset, config.dataset_id)

    if config.normalize_state:
        state_mean, state_std = compute_mean_std(qdataset["observations"], eps=1e-3)
    else:
        state_mean, state_std = 0, 1

    qdataset["observations"] = normalize_states(
        qdataset["observations"], state_mean, state_std
    )
    qdataset["next_observations"] = normalize_states(
        qdataset["next_observations"], state_mean, state_std
    )

    eval_env = wrap_env(eval_env, state_mean=state_mean, state_std=state_std)
    replay_buffer = ReplayBuffer(
        state_dim,
        action_dim,
        config.buffer_size,
        DEVICE,
    )
    replay_buffer.load_dataset(qdataset)

    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    # Set seeds
    set_seed(config.train_seed)

    q_network = TwinQ(state_dim, action_dim).to(DEVICE)
    v_network = ValueFunction(state_dim).to(DEVICE)
    if config.iql_deterministic:
        actor = DeterministicPolicy(
            state_dim, action_dim, max_action, dropout=config.actor_dropout
        ).to(DEVICE)
    else:
        actor = GaussianPolicy(
            state_dim, action_dim, max_action, dropout=config.actor_dropout
        ).to(DEVICE)

    v_optimizer = torch.optim.Adam(v_network.parameters(), lr=config.vf_lr)
    q_optimizer = torch.optim.Adam(q_network.parameters(), lr=config.qf_lr)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.actor_lr)
    actor_lr_scheduler = CosineAnnealingLR(actor_optimizer, config.update_steps)

    trainer = ImplicitQLearning(
        max_action=max_action,
        actor=actor,
        actor_optimizer=actor_optimizer,
        actor_lr_scheduler=actor_lr_scheduler,
        q_network=q_network,
        q_optimizer=q_optimizer,
        v_network=v_network,
        v_optimizer=v_optimizer,
        iql_tau=config.iql_tau,
        beta=config.beta,
        gamma=config.gamma,
        tau=config.tau,
        device=DEVICE,
    )

    for step in trange(config.update_steps):
        batch = [b.to(DEVICE) for b in replay_buffer.sample(config.batch_size)]
        log_dict = trainer.train(batch)

        wandb.log(log_dict, step=step)

        if (step + 1) % config.eval_every == 0:
            eval_scores, eval_distances = evaluate(
                env=eval_env,
                actor=actor,
                num_episodes=config.eval_episodes,
                seed=config.eval_seed,
                device=DEVICE,
            )

            wandb.log({"evaluation_return_scores": eval_scores.mean()}, step=step)
            wandb.log({"evaluation_return_distances": eval_distances.mean()}, step=step)
            # optional normalized score logging, only if dataset has reference scores
            with contextlib.suppress(ValueError):
                normalized_score = (
                    minari.get_normalized_score(dataset, eval_scores).mean() * 100
                )
                wandb.log({"normalized_score": normalized_score}, step=step)

            if config.checkpoints_path is not None:
                torch.save(
                    trainer.state_dict(),
                    os.path.join(config.checkpoints_path, f"checkpoint_{step}.pt"),
                )


if __name__ == "__main__":
    train()
