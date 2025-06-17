import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import TYPE_CHECKING

from rlforge.functional import BaselineFactory, BaselineType

if TYPE_CHECKING:
    from ..experiments import Experiment

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

class Agent:
    def __init__(self, name, learning_rate=0.01, gamma=0.99, baseline_type=None):
        self.name = name
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.experiment = None
        self.env = None
        self.policy_network = None
        self.optimizer = None
        if isinstance(baseline_type, BaselineType):
            self.baseline_type = baseline_type
            self.baseline_args = {}
        elif isinstance(baseline_type, tuple):
            self.baseline_type = BaselineType(baseline_type[0])
            self.baseline_args = baseline_type[1] if len(baseline_type) > 1 else {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

    def build_policy_network(self):
        from ..policies import PolicyNetwork
        if self.env is None:
            raise ValueError("Environment is not set. Call set_experiment() first.")
        if self.env.observation_space is None or self.env.action_space is None:
            raise ValueError("Environment's observation_space or action_space is not initialized.")
        if not hasattr(self.env.observation_space, "shape") or self.env.observation_space.shape is None:
            raise ValueError("Observation space does not have a valid 'shape' attribute.")
 

        
        network = PolicyNetwork(self.env.observation_space.shape[0], self.env.action_space.n)
        return network.to(self.device)

    def set_policy_network(self, policy_network, optimizer=None):
        self.policy_network = policy_network.to(self.device)
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = optim.AdamW(self.policy_network.parameters(), lr=self.learning_rate, weight_decay=1e-5)

 

    def set_experiment(self, experiment: 'Experiment', reset_policy_network=True):
        from ..experiments import Experiment
        if not isinstance(experiment, Experiment):
            raise ValueError("experiment must be an instance of Experiment")
        self.experiment = experiment
        self.env = experiment.env

        # Build policy network after environment is set
        if self.policy_network is None or reset_policy_network:
            self.policy_network = self.build_policy_network()
            self.optimizer = optim.Adam(self.policy_network.parameters(), lr=self.learning_rate)
        self.baseline = BaselineFactory.create_baseline(self, self.baseline_type, **self.baseline_args)

    def select_action(self, state):
        import torch.nn.functional as F
        """Select an action based on the current state using the policy network."""        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_logits = self.policy_network(state_tensor)
        dist = torch.distributions.Categorical(F.softmax(action_logits, dim=-1))
        action = dist.sample()
        return action, dist.log_prob(action)


    def evaluate(self, episodes=10, max_steps=200):
        """Evaluate the agent for a number of episodes and return average metrics."""
        if self.experiment is None:
            raise RuntimeError("Experiment is not set. Please set an Experiment using set_experiment() before evaluation.")
        
        self.policy_network.eval()  # Set to evaluation mode
        total_rewards = []
        episode_lengths = []
        
        for _ in range(episodes):
            episode_data = self.experiment.run_episode(agent=self, max_steps=max_steps, render=False)
            total_rewards.append(episode_data.total_reward)
            episode_lengths.append(len(episode_data))
        
        self.policy_network.train()  # Set back to training mode
        
        avg_reward = np.mean(total_rewards)
        avg_length = np.mean(episode_lengths)
        
        return avg_reward, avg_length

    def train_online(self, episodes=1000, max_steps=200, render=False, wandb_logging=False):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def __str__(self):
        return self.name
    def __repr__(self):
        return f"Agent(name={self.name}, learning_rate={self.learning_rate}, gamma={self.gamma})"
