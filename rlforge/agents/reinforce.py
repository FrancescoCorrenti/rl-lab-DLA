import random
import torch
import torch.nn.functional as F
import torch.nn as nn
import time
import os
import textwrap
from typing import Optional
from contextlib import nullcontext
from tqdm import tqdm

from .agent import Agent
from ..functional import BaselineType, SchedulerType, PolicyGradientUtils, DebugTimer
from ..policies import REINFORCEPolicy

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None


class REINFORCEAgent(Agent):
    """REINFORCE policy gradient agent."""

    def __init__(self, name, gamma=0.99, experiment=None,
                 entropy_weight=0.01, hidden_dims=[128], activation ="relu", 
                 baseline_type=BaselineType.NONE, debug_timing=False, 
                 aggregation="mean"):
        super().__init__(name=name, gamma=gamma,
                         baseline_type=baseline_type)

        self.entropy_weight = entropy_weight
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.debug_timing = debug_timing
        self.timer = DebugTimer() if debug_timing else None
        self._is_eval = False
        if experiment is not None:
            self.set_experiment(experiment)
        self.aggregation = self._get_aggregation_function(aggregation)

    def eval(self):
        self._is_eval = True
        self.policy_network.eval()
    
    def train(self):
        self._is_eval = False
        self.policy_network.train()

    def build_policy_network(self):
        if self.env is None:
            raise ValueError("Environment is not set. Please set an environment using set_experiment() before building the policy network.")
        
        obs_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.n
        
        if len(self.hidden_dims) == 1:
            return REINFORCEPolicy(obs_dim, action_dim, hidden_dim=self.hidden_dims[0], activation=self.activation).to(self.device)
        else:
            from ..policies import DeepPolicy
            return DeepPolicy(obs_dim, action_dim, hidden_dims=self.hidden_dims, activation=self.activation).to(self.device)

    def select_action(self, state):        # Convert to tensor and move to appropriate device
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)        
        with torch.no_grad() if self._is_eval else nullcontext():
            logits = self.policy_network(state_tensor)        
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)        
        action = dist.sample()       
        return action.item(), dist.log_prob(action), logits

    def _compute_policy_loss(self, episode_data, returns, episode=0):
        """Compute policy gradient loss with optional entropy regularization."""
        log_probs = torch.stack(episode_data.get_log_probs()).squeeze(-1)
        logits = episode_data.get_logits(return_pt=True).to(self.device)
       
        policy_loss = self.aggregation(-returns * log_probs)
        
        # Add entropy regularization if specified
        entropy_loss = torch.tensor(0.0, device=self.device)
        entropy_weight_actual = self.entropy_weight(episode) if callable(self.entropy_weight) else self.entropy_weight
        if entropy_weight_actual > 0:
            entropy = PolicyGradientUtils.compute_entropy(logits)
            entropy_loss = -entropy_weight_actual * entropy

        # Total loss
        loss = policy_loss + entropy_loss
        
        return loss, policy_loss.item(), entropy_loss.item()

    def _initialize_wandb(self, wandb_logging, wandb_project, wandb_run, wandb_config,
                         episodes, max_steps, eval_every, eval_episodes):
        """Initialize Weights & Biases logging if requested."""
        if not wandb_logging:
            return
        
        if not WANDB_AVAILABLE:
            print("Warning: wandb module not available. Wandb logging will be disabled.")
            return
        
        config = {
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "episodes": episodes,
            "max_steps": max_steps,
            "eval_every": eval_every,
            "eval_episodes": eval_episodes,
            "algorithm": "REINFORCE",
            "device": str(self.device),
            "entropy_weight": self.entropy_weight,
            "hidden_dims": self.hidden_dims,
            "baseline_type": str(self.baseline_type)
        }
        
        if wandb_config:
            config.update(wandb_config)

        wandb.init(project=wandb_project, name=wandb_run, config=config)
        self.wandb_project = wandb_project
        self.wandb_run = wandb_run

    def _log_gradient_metrics(self, wandb_logging, episode):
        """Log gradient statistics for monitoring training health."""
        if not wandb_logging:
            return
        
        grad_metrics = {}
        total_norm = 0.0
        param_count = 0
        
        for name, param in self.policy_network.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.detach().data.norm(2)
                grad_metrics[f"grad_norm/{name}"] = param_norm.item()
                total_norm += param_norm.item() ** 2
                param_count += 1
        
        total_norm = total_norm ** (1. / 2)
        grad_metrics["grad_norm/total"] = total_norm
        grad_metrics["grad_norm/avg_per_param"] = total_norm / max(param_count, 1)
        
        wandb.log(grad_metrics)

    def _perform_evaluation(self, wandb_logging, episode, eval_episodes, max_steps, pbar=None):
        self.eval()  # Set to evaluation mode

        avg_reward, avg_length = self.evaluate(episodes=eval_episodes, max_steps=max_steps)
        evaluation_result = {
            'episode': episode + 1,
            'avg_reward': avg_reward,
            'avg_length': avg_length
        }

        if wandb_logging:
            wandb.log({"eval/avg_reward": avg_reward, 
                       "eval/avg_length": avg_length,
                       "episode": episode + 1})

        # Update and maintain the postfix
        if not hasattr(self, 'eval_postfix'):
            self.eval_postfix = {}
            
        self.eval_postfix.update({
            "eval_avg_reward": f"{avg_reward:.2f}",
            "eval_avg_length": f"{avg_length:.1f}"
        })
        if pbar is not None:
            pbar.set_postfix(self.eval_postfix)

        self.train()
        return evaluation_result

    def _generate_model_description(self, episodes, max_steps):
        """Generate a pretty description of the model and training configuration."""
        
        network_str = str(self.policy_network)
        network_str = textwrap.indent(network_str, '    ')
        
        env_name = getattr(self.env, 'unwrapped', self.env).__class__.__name__
        obs_shape = self.env.observation_space.shape
        action_space = getattr(self.env.action_space, 'n', 'continuous')
        
        baseline_str = str(self.baseline_type) if self.baseline_type else "None"

        description = f"""
╔{'═' * 70}╗
║{' ' * 25}REINFORCE AGENT{' ' * 30}║
╠{'═' * 70}╣
║ {f"Agent: {self.name}":<68} ║
║ {f"Environment: {env_name}":<68} ║
║ {f"Observation Space: {obs_shape}":<68} ║
║ {f"Action Space: {action_space}":<68} ║
║ {f"Device: {self.device}":<68} ║
╠{'═' * 70}╣
║ {' ' * 26}TRAINING CONFIG{' ' * 27}║
╠{'═' * 70}╣
║ {f"Learning Rate: {self.learning_rate}":<68} ║
║ {f"Discount Factor (γ): {self.gamma}":<68} ║
║ {f"Episodes: {episodes}":<68} ║
║ {f"Max Steps per Episode: {max_steps}":<68} ║
║ {f"Entropy Weight: {self.entropy_weight}":<68} ║
║ {f"Baseline Type: {baseline_str}":<68} ║
╠{'═' * 70}╣
║ {' ' * 26}NETWORK ARCHITECTURE{' ' * 23}║
╠{'═' * 70}╣
{textwrap.indent(network_str, '║ ').rstrip()}
╚{'═' * 70}╝
"""
        return description

    def _run_episode(self, max_steps: int):
        """Run a single episode and return the collected data."""
        with self.timer.time_block("episode_execution") if self.timer else nullcontext():
            if self.experiment is None:
                raise RuntimeError("Experiment is not set. Please set an Experiment using set_experiment() before training.")
            
            episode_data = self.experiment.run_episode(agent=self, max_steps=max_steps)
            return episode_data

    def _perform_update_step(self, episode_data, max_grad_norm, episode = 0):
        """Process a single episode's data to compute losses and update the policy."""
        with self.timer.time_block("data_preparation") if self.timer else nullcontext():
            # Compute returns using baseline
            returns = self.baseline(episode_data) if self.baseline else PolicyGradientUtils.compute_discounted_returns(
                torch.FloatTensor(episode_data.get_rewards()).to(self.device), self.gamma
            )
        
        # Compute losses
        with self.timer.time_block("gradient_computation") if self.timer else nullcontext():
            self.optimizer.zero_grad()            
            loss, policy_loss, entropy_loss = self._compute_policy_loss(episode_data, returns, episode=episode)            
            # Backpropagate loss
            loss.backward()
            
            # Clip gradients if specified
            if max_grad_norm > 0:
                nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_grad_norm)
            
            # Update policy network
            self.optimizer.step()
            
            # Update baseline if needed
            if hasattr(self.baseline, 'train_step'):
                self.baseline.train_step(episode_data)
        
        if self.scheduler:
            self.scheduler.step()
        
        return {"total_loss": loss.item(), "policy_loss": policy_loss, "entropy_loss": entropy_loss}

    def _log_metrics(self, wandb_logging, episode, episode_data, losses):
        if not wandb_logging:
            return
            
        log_data = {
            "episode": episode + 1,
            "episode_reward": episode_data.total_reward,
            "episode_length": len(episode_data),
            "loss/policy": losses["policy_loss"],
            "learning_rate": self.optimizer.param_groups[0]['lr']
        }
        
        if "entropy_loss" in losses:
            log_data["loss/entropy"] = losses["entropy_loss"]
        
        if "total_loss" in losses:
            log_data["loss/total"] = losses["total_loss"]
            
        if self.debug_timing and self.timer:
            timing_stats = self.timer.get_all_stats()
            for name, stats in timing_stats.items():
                log_data[f"timing/{name}/mean"] = stats['mean']
                log_data[f"timing/{name}/max"] = stats['max']
        
        wandb.log(log_data)

    def _maybe_evaluate(self, wandb_logging, episode, eval_every, eval_episodes, max_steps,
                        best_eval_reward, episodes_without_improvement, reload_best_after_episodes,
                        save_best_model_path, early_stop=False, early_stop_path=None, pbar=None):
        evaluation_result = None
        is_solved = False
        
        if eval_every > 0 and (episode + 1) % eval_every == 0:
            evaluation_result = self._perform_evaluation(wandb_logging, episode, eval_episodes, max_steps, pbar)
            avg_reward = evaluation_result['avg_reward']
            
            # Check if we've beaten our previous best
            if avg_reward > best_eval_reward.value:
                print(f"\n🏆 New best model! Reward improved from {best_eval_reward.value:.2f} to {avg_reward:.2f}")
                best_eval_reward.value = avg_reward
                episodes_without_improvement.value = 0
                
                # Save the best model if a path is provided
                if save_best_model_path:
                    os.makedirs(os.path.dirname(save_best_model_path), exist_ok=True)
                    torch.save(self.policy_network.state_dict(), save_best_model_path)
                    print(f"🔄 Saved best model to {save_best_model_path}")
                
                # Check if environment is solved
                is_solved = self._check_if_solved(avg_reward,max_steps=max_steps)
                if is_solved and early_stop:
                    print(f"🎉 Environment solved! Stopping training.")
                    if early_stop_path and early_stop_path != save_best_model_path:
                        torch.save(self.policy_network.state_dict(), early_stop_path)
                        print(f"🔄 Saved solved model to {early_stop_path}")
            else:
                episodes_without_improvement.value += 1
                
                # Reload best model if we've gone too long without improvement
                if (reload_best_after_episodes > 0 and 
                    episodes_without_improvement.value >= reload_best_after_episodes and 
                    save_best_model_path and 
                    os.path.exists(save_best_model_path)):
                    
                    print(f"\n⚠️ No improvement for {episodes_without_improvement.value} episodes. Reloading best model...")
                    self.policy_network.load_state_dict(torch.load(save_best_model_path))
                    episodes_without_improvement.value = 0
        
        return evaluation_result, is_solved

    def _check_if_solved(self, avg_reward, solve_threshold=200.0, confirm_episodes=100, max_steps=None):
        """
        Check if the environment is considered solved based on the average reward.
        
        Args:
            avg_reward (float): The average reward from evaluation.
            solve_threshold (float): Threshold for considering the environment solved.
            confirm_episodes (int): Number of episodes to run for confirmation.
            max_steps (int, optional): Maximum steps per episode for evaluation.
            
        Returns:
            bool: True if the environment is solved, False otherwise.
        """
        if avg_reward < solve_threshold:
            return False
            
        print(f"\n🎉 Potential solution detected! Average reward: {avg_reward:.2f}")
        print(f"Confirming solution with {confirm_episodes} additional episodes...")
        
        confirm_reward, _ = self.evaluate(episodes=confirm_episodes, max_steps=max_steps)
        
        if confirm_reward >= solve_threshold:
            print(f"✅ Environment solved! Confirmed average reward: {confirm_reward:.2f}")
            return True
        else:
            print(f"❌ Solution not confirmed. Average reward: {confirm_reward:.2f}")
            return False

    def train_online(self, episodes=500, max_steps=200, render_every=-1, eval_every=100,
                    eval_episodes=10, wandb_logging=False, wandb_project="reinforce-training",
                    wandb_run=None, wandb_config=None, debug_timing=None,
                    save_best_model_path: Optional[str] = None, reload_best_after_episodes=-1,
                    log_gradients=True, log_gradients_every=10,
                    max_grad_norm=0, scheduler_type: SchedulerType = SchedulerType.NONE, learning_rate=1e-3,
                    early_stop=False, early_stop_path=None):
        """
        Train the agent using online policy gradient method.
        
        Args:
            episodes (int): Number of training episodes.
            max_steps (int): Maximum steps per episode. 
            render_every (int): Render every N episodes. Set to -1 to disable rendering.
            eval_every (int): Evaluate every N episodes.
            eval_episodes (int): Number of episodes to run for evaluation.
            wandb_logging (bool): Whether to log metrics to Weights & Biases.
            wandb_project (str): Wandb project name.
            wandb_run (str): Wandb run name. If None, a random name will be generated.
            wandb_config (dict): Additional configuration for wandb logging.
            debug_timing (bool): Whether to enable debug timing for performance profiling.
            save_best_model_path (Optional[str]): Path to save the best model. If None, no model will be saved.
            reload_best_after_episodes (int): Reload the best model if no improvement after this many episodes.
            log_gradients (bool): Whether to log gradient statistics.
            log_gradients_every (int): Log gradients every N episodes.
            max_grad_norm (float): Maximum gradient norm for clipping. Set to 0 to disable clipping.
            scheduler_type (SchedulerType or tuple): Type of learning rate scheduler to use. If a tuple, it should be (SchedulerType, kwargs).            
            learning_rate (float): Learning rate for the optimizer. If a scheduler is used, this will be the initial learning rate.
            early_stop (bool): Whether to stop training early if the environment is solved, based on property experiment.reward_threshold.
            early_stop_path (Optional[str]): Path to save the model if early stopping is triggered. Defaults to save_best_model_path.      
        """
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=self.learning_rate)

        if self.experiment is None:
            raise RuntimeError("Experiment is not set. Please set an Experiment using set_experiment() before training.")
        
        # Set up debug timing
        if debug_timing is not None:
            self.debug_timing = debug_timing
            if debug_timing and self.timer is None:
                self.timer = DebugTimer()

        if isinstance(scheduler_type, SchedulerType):
            if scheduler_type != SchedulerType.NONE:
                scheduler_kwargs = None
                self._initialize_scheduler(scheduler_type, scheduler_kwargs)
        elif isinstance(scheduler_type, tuple) and len(scheduler_type) == 2:
            scheduler_type, scheduler_kwargs = scheduler_type
            if isinstance(scheduler_type, SchedulerType):
                self._initialize_scheduler(scheduler_type, scheduler_kwargs)
            else:
                raise ValueError(f"Invalid scheduler type: {scheduler_type}. Must be a SchedulerType enum or a tuple of (SchedulerType, kwargs).")

        # Initialize wandb
        self._initialize_wandb(wandb_logging, wandb_project, wandb_run, wandb_config,
                             episodes, max_steps, eval_every, eval_episodes)
        
        # Print agent description
        print(self._generate_model_description(episodes, max_steps))
        
        # Track best evaluation reward
        class Value:
            def __init__(self, initial_value):
                self.value = initial_value
        
        best_eval_reward = Value(-float('inf'))
        episodes_without_improvement = Value(0)
        
        evaluation_results = []
        # Train loop
        with tqdm(range(episodes), desc=f"Training {self.name}") as pbar:
            for episode in pbar:
                # Run episode
                episode_data = self._run_episode(max_steps)

                # Perform update step
                losses = self._perform_update_step(episode_data, max_grad_norm, episode=episode)
                
                # Log metrics
                self._log_metrics(wandb_logging, episode, episode_data, losses)
                
                # Log gradients
                if log_gradients and (episode + 1) % log_gradients_every == 0:
                    self._log_gradient_metrics(wandb_logging, episode)
                
                # Render
                if render_every > 0 and (episode + 1) % render_every == 0:
                    self.experiment.run_episode(agent=self, max_steps=max_steps, render=True)
                
                # Evaluate
                evaluation_result, is_solved = self._maybe_evaluate(
                    wandb_logging, episode, eval_every, eval_episodes, max_steps,
                    best_eval_reward, episodes_without_improvement, reload_best_after_episodes,
                    save_best_model_path, early_stop, early_stop_path, pbar
                )
                self._is_solved = is_solved
                
                if evaluation_result:
                    evaluation_results.append(evaluation_result)
                
                # Early stopping if solved
                if self.has_solved and early_stop:
                    break
                
                # Update progress bar
                pbar.set_postfix({
                    "reward": f"{episode_data.total_reward:.2f}",
                    "length": len(episode_data),
                    "loss": f"{losses['total_loss']:.4f}"
                })
        
        # Final evaluation
        if eval_every > 0:
            final_eval = self._perform_evaluation(wandb_logging, episodes-1, eval_episodes, max_steps)
            print(f"\nFinal evaluation: {final_eval}")
        
        # Print timing summary
        if self.debug_timing and self.timer:
            self.timer.print_summary()
        
        # Close wandb
        if wandb_logging and WANDB_AVAILABLE:
            wandb.finish()
        
        if save_best_model_path and os.path.exists(save_best_model_path):
            print(f"\nLoading best model from {save_best_model_path}...")
            self.policy_network.load_state_dict(torch.load(save_best_model_path))
        
        self.evaluation_results = evaluation_results
        return evaluation_results, best_eval_reward.value
    
    def load_state_dict(self, state_dict, val_to_beat=None):
        """Load the state dictionary into the agent's policy network."""
        super().load_state_dict(state_dict, val_to_beat)
        # Additional REINFORCE-specific loading can be added here

    def _get_aggregation_function(self, aggregation):
        """Get the aggregation function based on the specified type."""
        if aggregation == "mean":
            return torch.mean
        elif aggregation == "sum":
            return torch.sum
        else:
            raise ValueError(f"Invalid aggregation type: {aggregation}. Must be one of 'mean', 'sum'.")