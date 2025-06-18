import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import TYPE_CHECKING

from rlforge.functional import BaselineFactory, BaselineType, SchedulerType

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
        self.baseline = None
        self.optimizer = None
        self.scheduler = None
        if isinstance(baseline_type, BaselineType):
            print(f"Using baseline type: {baseline_type}")
            self.baseline_type = baseline_type
            self.baseline_args = {}
        elif isinstance(baseline_type, tuple):
            self.baseline_type = BaselineType(baseline_type[0])
            self.baseline_args = baseline_type[1] if len(baseline_type) > 1 else {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

    def _initialize_scheduler(self, scheduler_type: SchedulerType, scheduler_kwargs: dict):
        """Initializes the learning rate scheduler."""
        from rlforge.functional import SchedulerFactory
        self.scheduler = SchedulerFactory.create_scheduler(self.optimizer, scheduler_type, **scheduler_kwargs)

    def build_policy_network(self):
        from ..policies import PolicyNetwork
        try:
             self.check_env()
        except ValueError as e:
            raise ValueError(f"Environment is not set or not properly initialized: {e}")
        network = PolicyNetwork(self.env.observation_space.shape[0], self.env.action_space.n)
        return network.to(self.device)

    def build_baseline(self):
        """Builds the baseline network based on the specified baseline type.""" 
        if self.baseline_type is None:
            return BaselineFactory.create_baseline(self, BaselineType.NONE)
        else:
            try:
                self.check_env()
            except ValueError as e:
               raise ValueError(f"Environment is not set or not properly initialized: {e}")
            return BaselineFactory.create_baseline(
                self,  
                self.baseline_type,  
                **self.baseline_args
            )
        
    def set_policy_network(self, policy_network, optimizer=None):
        self.policy_network = policy_network.to(self.device)
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = optim.Adam(self.policy_network.parameters(), lr=self.learning_rate)

 
    def check_env(self):
        """Check if the environment is set and properly initialized."""
        if self.env is None:
            raise ValueError("Environment is not set. Please set an environment using set_experiment() before training.")
        if not isinstance(self.env, gym.Env):
            raise ValueError("The environment must be an instance of gym.Env or a compatible environment.")
        if not hasattr(self.env, 'observation_space') or not hasattr(self.env, 'action_space'):
            raise ValueError("The environment must have observation_space and action_space attributes.")
        
    def set_experiment(self, experiment: 'Experiment', reset_policy_network=True, reset_baseline_network=True):
        from ..experiments import Experiment
        if not isinstance(experiment, Experiment):
            raise ValueError("experiment must be an instance of Experiment")
        self.experiment = experiment
        self.env = experiment.env

        if self.policy_network is None or reset_policy_network:
            self.policy_network = self.build_policy_network()
            self.optimizer = optim.Adam(self.policy_network.parameters(), lr=self.learning_rate)
        if self.baseline is None or reset_baseline_network:
            self.baseline = self.build_baseline()

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
    
    def save_video(self, max_steps=200):
        """
        Renders a video of the agent's performance for one episode.
        The experiment's `env_renderer` must be set up for rendering,
        e.g., with `gym.wrappers.RecordVideo`.

        Args:
            max_steps (int): Maximum steps per episode.
        """
        if self.experiment is None:
            raise RuntimeError("Experiment is not set. Please set an Experiment using set_experiment() before rendering.")
        
        self.experiment.save_video(agent=self, max_steps=max_steps)

    def render(self, max_steps=200):
        """
        Renders the agent's performance for one episode.
        The experiment's `env_renderer` must be set up for rendering.

        Args:
            max_steps (int): Maximum steps per episode.
        """
        if self.experiment is None:
            raise RuntimeError("Experiment is not set. Please set an Experiment using set_experiment() before rendering.")
        
        self.experiment.run_episode(agent=self, max_steps=max_steps, render=True)

    def load_state_dict(self, state_dict, val_to_beat=None):
        """Load the state dictionary into the agent's policy network."""
        if self.policy_network is None:
            raise RuntimeError("Policy network is not set. Please set a policy network before loading state dict.")
        self.policy_network.load_state_dict(state_dict)
        if val_to_beat is not None:
            self.evaluation_best = val_to_beat

    def load_baseline_state_dict(self, state_dict):
        """Load the state dictionary into the agent's baseline network."""
        if self.baseline is None:
            raise RuntimeError("Baseline network is not set. Please set a baseline network before loading state dict.")
        self.baseline.load_state_dict(state_dict)


    def __str__(self):
        return self.name
    def __repr__(self):
        return f"Agent(name={self.name}, learning_rate={self.learning_rate}, gamma={self.gamma})"
