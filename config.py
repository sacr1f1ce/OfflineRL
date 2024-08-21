import uuid
import os
from typing import Optional
from dataclasses import dataclass


@dataclass
class TrainConfig:
    # wandb params
    project: str = "IQL"
    name: str = "test_run"
    # model params
    gamma: float = 0.99  # Discount factor
    tau: float = 0.005  # Target network update rate
    beta: float = 10.0  # Inverse temperature. Small beta -> BC, big beta -> maximizing Q
    iql_tau: float = 0.9  # Coefficient for asymmetric loss
    iql_deterministic: bool = False  # Use deterministic actor
    vf_lr: float = 3e-4  # V function learning rate
    qf_lr: float = 3e-4  # Critic learning rate
    actor_lr: float = 3e-4  # Actor learning rate
    actor_dropout: Optional[float] = None  # Adroit uses dropout for policy network
    # training params
    dataset_id: str = "antmaze-medium-diverse-v1"  # Minari remote dataset name
    update_steps: int = int(1e6)  # Total training networks updates
    buffer_size: int = int(1e7)  # Replay buffer size
    batch_size: int = 256  # Batch size for all networks
    normalize_state: bool = True  # Normalize states
    normalize_reward: bool = True  # Normalize reward
    # evaluation params
    eval_every: int = 50  # How often (time steps) we evaluate
    eval_episodes: int = 100  # How many episodes run during evaluation
    # general params
    train_seed: int = 0
    eval_seed: int = 0
    checkpoints_path: Optional[str] = None  # Save path

    def __post_init__(self):
        self.name = f"{self.name}-{self.dataset_id}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)
