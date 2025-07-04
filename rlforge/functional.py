from enum import Enum

import numpy as np
import wandb
from rlforge.data import EpisodeData, StepData
import torch
import torch.nn.functional as F
from torch import nn
from collections import defaultdict, deque
import time
from contextlib import contextmanager

ACTIVATION_FUNCTIONS = {
    "relu": nn.ReLU(),
    "tanh": nn.Tanh(),
    "sigmoid": nn.Sigmoid(),
    "leaky_relu": nn.LeakyReLU(negative_slope=0.01),
    "elu": nn.ELU(),
    "selu": nn.SELU(),
    "softmax": nn.Softmax(dim=-1),
}

class SchedulerType(Enum):
    """
    Types of learning rate schedulers.
    - NONE: No scheduler, learning rate remains constant.
    - CYCLIC: Cyclic learning rate scheduler.
    - STEP: Step learning rate scheduler that reduces the learning rate at specified intervals.
    - EXPONENTIAL: Exponential decay learning rate scheduler.
    - COSINE_ANNEALING: Cosine annealing learning rate scheduler.
    """
    NONE = "none"
    CYCLIC = "cyclic"
    STEP = "step"
    EXPONENTIAL = "exponential"
    COSINE_ANNEALING = "cosine_annealing"

class BaselineType(Enum):
    """Types of baselines for variance reduction in policy gradient methods.
    - NONE: No baseline, returns raw discounted returns.
    - EPISODIC_STANDARDIZATION: Standardizes returns within each episode.
    - RUNNING_STANDARDIZATION: Standardizes returns using running statistics across episodes.
    - VALUE_FUNCTION: Uses a neural network value function to estimate returns.
        Arguments for this baseline can be passed as a dictionary:
        {
            "learning_rate": float,  # Learning rate for the value function optimizer
            "hidden_dims": list,     # Hidden dimensions for the value function network
            "normalize_returns": bool,  # Whether to normalize returns
            "normalize_advantages": bool,  # Whether to normalize advantages
            "epsilon": float,        # Small value to avoid division by zero in normalization
            "init_gain": float,      # Gain for orthogonal initialization of weights
            "wandb_logging": bool     # Whether to log metrics to wandb
            "momentum": float,  # Momentum for running statistics 
            "activation": str,  # Activation function for the value network
            "max_grad_norm": float  # Maximum gradient norm for clipping
        }
    """

    NONE = "none"
    EPISODIC_STANDARDIZATION = "episodic_standardization"
    RUNNING_STANDARDIZATION = "running_standardization"
    VALUE_FUNCTION = "value_function"


class DebugTimer:
        """Utility class for accurate timing measurements during training."""
        
        def __init__(self, window_size=100):
            self.timings = defaultdict(deque)
            self.window_size = window_size
            self.current_timings = {}
        
        @contextmanager
        def time_block(self, name):
            """Context manager for timing code blocks."""
            start_time = time.perf_counter()
            try:
                yield
            finally:
                end_time = time.perf_counter()
                elapsed = end_time - start_time
                self.timings[name].append(elapsed)
                if len(self.timings[name]) > self.window_size:
                    self.timings[name].popleft()
                self.current_timings[name] = elapsed
        
        def get_stats(self, name):
            """Get timing statistics for a named block."""
            if name not in self.timings or not self.timings[name]:
                return None
            times = list(self.timings[name])
            return {
                'mean': sum(times) / len(times),
                'min': min(times),
                'max': max(times),
                'current': self.current_timings.get(name, 0),
                'count': len(times)
            }
        
        def get_all_stats(self):
            """Get timing statistics for all blocks."""
            return {name: self.get_stats(name) for name in self.timings.keys()}
        
        def print_summary(self):
            """Print a summary of all timing statistics."""
            print("\n" + "="*60)
            print("TIMING DEBUG SUMMARY")
            print("="*60)
            for name, stats in self.get_all_stats().items():
                if stats:
                    print(f"{name:25} | "
                        f"Mean: {stats['mean']*1000:6.2f}ms | "
                        f"Min: {stats['min']*1000:6.2f}ms | "
                        f"Max: {stats['max']*1000:6.2f}ms | "
                        f"Current: {stats['current']*1000:6.2f}ms")
            print("="*60)

class PolicyGradientUtils:
    """Utility functions for policy gradient algorithms."""
    
    @staticmethod
    def compute_discounted_returns(rewards, gamma=None, normalize=True):
        if isinstance(rewards, EpisodeData):
            rewards = rewards.get_rewards(return_pt=True)
        rewards = rewards.to(torch.float64)
        T = rewards.size(0)
        gamma_powers = gamma ** torch.arange(T, dtype=torch.float64, device=rewards.device)
        discounted = rewards * gamma_powers
        returns = torch.flip(torch.cumsum(torch.flip(discounted, [0]), 0), [0])
        if normalize:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns.float()


    def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
        """Compute the entropy of a probability distribution given logits."""
        probs = F.softmax(logits, dim=-1)
        probs = torch.clamp(probs, min=1e-8, max=1.0)
        log_probs = torch.log(probs + 1e-8)
        return -torch.sum(probs * log_probs, dim=-1).mean()

    import torch

    from torch.optim.lr_scheduler import _LRScheduler
    class BanditLR(_LRScheduler):
        """
        Exp3-style multi-armed-bandit learning-rate scheduler (LRRL) for PyTorch.
        • arms: list/tuple of candidate learning-rates (η₁ … η_K)
        • α:    bandit step-size (positive float)
        • δ:    exponential decay for past rewards in weights 0 < δ ≤ 1
        Call `update(avg_reward)` after each κ-interaction block,
        then `step()` as usual to refresh the optimizer’s lr.
        """

        def __init__(self, optimizer, arms, α=0.2, δ=0.99, last_epoch=-1):
            self.arms = torch.tensor(arms, dtype=torch.float32)
            self.α = α
            self.δ = δ
            self.w = torch.zeros(len(arms))            # Exp3 weights
            self._last_reward = None                   # store most recent feedback
            super().__init__(optimizer, last_epoch)

        # -------- Bandit interface -------------------------------------------------
        @torch.no_grad()
        def update(self, reward: float):
            """Pass the average return of the latest κ interactions."""
            if self._last_reward is None:
                self._last_reward = reward
            Δ = reward - self._last_reward            # improvement signal (f′ₙ)
            self._last_reward = reward
            self.w.mul_(self.δ).add_(self.α * Δ)      # Eq. (3) with decay

        # -------- _LRScheduler hooks ----------------------------------------------
        def get_lr(self):
            # sample an arm according to softmax(w)
            probs = torch.softmax(self.w, dim=0)
            k = torch.multinomial(probs, 1).item()
            return [float(self.arms[k]) for _ in self.optimizer.param_groups]

class BaselineStrategy:
    """Base class for baseline strategies."""
    
    def __init__(self, agent):
        self.agent = agent
        self.device = agent.device

    def load_state_dict(self, state_dict):
        """Load the state dictionary into the baseline strategy."""
        if hasattr(self, 'value_network'):
            self.value_network.load_state_dict(state_dict)
        else:
            raise NotImplementedError("This baseline strategy does not support loading state dicts.")
    def train_step(self, episode_data: EpisodeData):
        """Perform a training step for the baseline strategy."""
        pass

class NoBaseline(BaselineStrategy):
    """No baseline - returns raw discounted returns."""
    
    def __call__(self, episode_data: EpisodeData) -> torch.Tensor:
        rewards = episode_data.get_rewards(return_pt=True).to(self.device)
        return PolicyGradientUtils.compute_discounted_returns(rewards, self.agent.gamma, normalize=False)


class EpisodicStandardization(BaselineStrategy):
    """Standardize rewards within each episode."""

    def __call__(self, episode_data: EpisodeData, train: bool = True) -> torch.Tensor:
        rewards = torch.FloatTensor(episode_data.get_rewards()).squeeze(-1).to(self.device)
        returns = PolicyGradientUtils.compute_discounted_returns(rewards, self.agent.gamma)
        return returns     


class RunningStandardization(BaselineStrategy):
    """Standardize returns using running statistics across episodes."""
    
    def __init__(self, agent, momentum: float = 0.01):
        super().__init__(agent)
        self.running_mean = 0.0
        self.running_std = 1.0
        self.momentum = momentum
    
    def __call__(self, episode_data: EpisodeData, train: bool = True) -> torch.Tensor:
        rewards = episode_data.get_rewards(return_pt=True).to(self.device)
        returns = PolicyGradientUtils.compute_discounted_returns(rewards, self.agent.gamma,normalize=False)
        
        # Update running statistics
        episode_mean = returns.mean().item()
        episode_std = returns.std().item()
        
        self.running_mean += self.momentum * (episode_mean - self.running_mean)
        self.running_std += self.momentum * (episode_std - self.running_std)
        
        # Standardize returns
        standardized_returns = (returns - self.running_mean) / (self.running_std + 1e-8)
        return standardized_returns

class RunningCentering(BaselineStrategy):
    """Center returns using running statistics."""
    def __init__(self, agent, momentum: float = 0.01, epsilon: float = 1e-8):
        super().__init__(agent)
        self.running_mean = 0.0
        self.momentum = momentum
        self.epsilon = epsilon

    def __call__(self, episode_data: EpisodeData, train: bool = True) -> torch.Tensor:
        values = episode_data.get_rewards(return_pt=True).to(self.device) 
        episode_mean = values.mean().item()
        
        # Update running mean
        self.running_mean += self.momentum * (episode_mean - self.running_mean)
        
        # Center values
        return values - self.running_mean + self.epsilon

class RunningAdvantageStandardization(BaselineStrategy):
    """Standardize advantages using running statistics."""

    def __init__(self, momentum: float = 0.01, epsilon: float = 1e-8):
        self.running_mean = 0.0
        self.running_std = 1.0
        self.momentum = momentum
        self.epsilon = epsilon
    
    def __call__(self, advantages: torch.Tensor, train: bool = True) -> torch.Tensor:
        episode_mean = advantages.mean().item()
        episode_std = advantages.std().item()
        
        self.running_mean += self.momentum * (episode_mean - self.running_mean)
        self.running_std += self.momentum * (episode_std - self.running_std)
        
        return (advantages - self.running_mean) / (self.running_std + self.epsilon)

class RunningAdvantageCentering:
    """Center advantages using running statistics."""
    
    def __init__(self, momentum: float = 0.01, epsilon: float = 1e-8):
        self.running_mean = 0.0
        self.momentum = momentum
        self.epsilon = epsilon

    def __call__(self, advantages: torch.Tensor, train: bool = True) -> torch.Tensor:
        episode_mean = advantages.mean().item()
        
        # Update running mean
        self.running_mean += self.momentum * (episode_mean - self.running_mean)
        
        # Center advantages
        return advantages - self.running_mean + self.epsilon
    
class ValueFunctionBaseline(BaselineStrategy):
    """Neural network value function baseline with advantage estimation."""
    
    def __init__(
        self,
        agent,
        learning_rate: float = 1e-2, 
        hidden_dims: list = [128, 128],
        epsilon: float = 1e-8,
        init_gain: float = 1.0,
        wandb_logging: bool = True,
        max_grad_norm: float = 0  
    ):
        super().__init__(agent)
        
        obs_dim = agent.env.observation_space.shape[0]
        if wandb_logging:
            try:
                current_run = wandb.run
                # Prepare a dictionary with value-function specific configurations
                vf_config = {
                    "vf_learning_rate": learning_rate,
                    "vf_hidden_dims": hidden_dims,
                    "vf_epsilon": epsilon,
                    "vf_init_gain": init_gain
                }

                if current_run is None:
                    # No active run, initialize a new one if agent provides project/run names
                    # Check if agent has the necessary attributes for wandb initialization
                    if hasattr(agent, 'wandb_project') and agent.wandb_project and \
                       hasattr(agent, 'wandb_run') and agent.wandb_run:
                        wandb.init(
                            project=agent.wandb_project,
                            name=agent.wandb_run,
                            config=vf_config,
                            reinit=False # Avoid issues if init is called multiple times
                        )
                    else:
                        print("Warning: wandb_project and/or wandb_run not specified in agent. "
                              "Cannot initialize new wandb run for ValueFunctionBaseline. "
                              "If a run is already active, its config will be updated.")
                        if wandb.run: # Check again in case an outer scope initialized it just now
                             wandb.config.update(vf_config, allow_val_change=True)

                else:
                    # Active run exists, update its config
                    wandb.config.update(vf_config, allow_val_change=True)
            
            except Exception as e:
                # Catch any exception during wandb interaction and print a warning
                print(f"Warning: Wandb logging for ValueFunctionBaseline encountered an error: {e}")
                
        # Create a simple feedforward neural network for value estimation
        layers = []
        in_dim = obs_dim
        for size in hidden_dims:
            layers.append(nn.Linear(in_dim, size))
            layers.append(nn.ReLU())
            in_dim = size
        layers.append(nn.Linear(in_dim, 1))
        self.value_network = nn.Sequential(*layers).to(self.device)

        # Initialize weights orthogonally
        self._init_weights(init_gain)        
        self.optimizer = torch.optim.Adam(self.value_network.parameters(), lr=learning_rate)        
        self.epsilon = epsilon
        self.max_grad_norm = max_grad_norm
        self.rewards_normalizer = EpisodicStandardization(agent)

    def _init_weights(self, gain: float = 1.0):
        """Initialize network weights orthogonally."""
        for module in self.value_network.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=gain)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def __call__(self, episode_data: EpisodeData) -> torch.Tensor:   
        with torch.no_grad():
            states = torch.FloatTensor(np.array(episode_data.get_states())).squeeze(-1).to(self.device)
            target_returns = self.rewards_normalizer(episode_data)
            values = self.value_network(states)        
        advantages = target_returns - values.detach()        
        advantages = advantages - advantages.mean()  
        advantages = advantages / (advantages.std() + self.epsilon)  
        return advantages

    def train_step(self, episode_data: EpisodeData):
        """Perform a training step for the value function baseline."""
        if not isinstance(episode_data, EpisodeData):
            raise ValueError("episode_data must be an instance of EpisodeData")
        states = torch.FloatTensor(np.array(episode_data.get_states())).squeeze(-1).to(self.device)
        values = self.value_network(states).squeeze(-1)  # Ensure values are 1D tensor
        target_returns = self.rewards_normalizer(episode_data)
        loss = F.mse_loss(values, target_returns,reduction='mean')  
        self.optimizer.zero_grad()
        loss.backward()        
        torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), self.max_grad_norm)
        self.optimizer.step()        
        # Log metrics to wandb if available
        try:
            if wandb.run is not None:
                wandb.log({
                    "value_function/loss": loss.item(),
                    "value_function/mean_value": values.mean().item(),
                    "value_function/value_std": values.std().item(),
                    "value_function/target_returns_mean": target_returns.mean().item(),
                    "value_function/target_returns_std": target_returns.std().item(),
                })
        except Exception as e:
            print(f"Warning: Wandb logging for ValueFunctionBaseline encountered an error: {e}")
        
        return self.value_network

class BaselineFactory:
    """Factory for creating baseline strategies."""
    
    _baseline_classes = {
        BaselineType.NONE: NoBaseline,
        BaselineType.EPISODIC_STANDARDIZATION: EpisodicStandardization,
        BaselineType.RUNNING_STANDARDIZATION: RunningStandardization,
        BaselineType.VALUE_FUNCTION: ValueFunctionBaseline,
    }
    
    @classmethod
    def create_baseline(cls, agent, baseline_type: BaselineType, **kwargs):
        """Create a baseline strategy instance."""
        if baseline_type not in cls._baseline_classes:
            raise ValueError(f"Unknown baseline type: {baseline_type}")
        
        baseline_class = cls._baseline_classes[baseline_type]
        return baseline_class(agent, **kwargs)


# -------------------------- Q-learning utilities ---------------------------

class ReplayBuffer:
    """Simple experience replay buffer for value-based methods."""

    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity

    def push(self, step: StepData):
        """Add a transition to the buffer."""
        self.buffer.append(step)

    def push_episode(self, episode: EpisodeData):
        """Add all transitions from an episode."""
        for step in episode.steps:
            self.push(step)

    def sample(self, batch_size: int, device=None):
        import random
        batch = random.sample(self.buffer, batch_size)
        states = torch.stack([torch.as_tensor(s.state, dtype=torch.float32) for s in batch]).to(device)
        actions = torch.as_tensor([s.action for s in batch], dtype=torch.long, device=device)
        rewards = torch.as_tensor([s.reward for s in batch], dtype=torch.float32, device=device)
        next_states = torch.stack([torch.as_tensor(s.next_state, dtype=torch.float32) for s in batch]).to(device)
        dones = torch.as_tensor([s.done for s in batch], dtype=torch.float32, device=device)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class TDUtils:
    pass

class SchedulerFactory:
    """Factory for creating learning rate schedulers."""
    
    @classmethod
    def create_scheduler(cls,optimizer, scheduler_type: SchedulerType, **kwargs):
        """Create a learning rate scheduler based on the specified type."""
        if scheduler_type == SchedulerType.NONE:
            return None
        
        if kwargs == {} or kwargs is None:
            kwargs = cls.get_default_kwargs(scheduler_type)

        scheduler_class = {
            SchedulerType.NONE: torch.optim.lr_scheduler.LambdaLR,
            SchedulerType.CYCLIC: torch.optim.lr_scheduler.CyclicLR,
            SchedulerType.STEP: torch.optim.lr_scheduler.StepLR,
            SchedulerType.EXPONENTIAL: torch.optim.lr_scheduler.ExponentialLR,
            SchedulerType.COSINE_ANNEALING: torch.optim.lr_scheduler.CosineAnnealingLR,
        }.get(scheduler_type)
        
        if scheduler_class is None:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
        
        return scheduler_class(optimizer, **kwargs)
    
    @staticmethod
    def get_default_kwargs(scheduler_type: SchedulerType):  
        """Get default keyword arguments for the specified scheduler type."""
        defaults = {
            SchedulerType.NONE: {"lambda": lambda epoch: 1.0},
            SchedulerType.CYCLIC: {"base_lr": 0.001, "max_lr": 0.01, "step_size_up": 2000, "mode": "triangular"},
            SchedulerType.STEP: {"step_size": 10000, "gamma": 0.1},
            SchedulerType.EXPONENTIAL: {"gamma": 0.99},
            SchedulerType.COSINE_ANNEALING: {"T_max": 10000, "eta_min": 0.001},
        }
        return defaults.get(scheduler_type, {})