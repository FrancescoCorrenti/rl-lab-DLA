import random
import torch
import torch.nn.functional as F
import torch.nn as nn
import time
import os
from typing import Optional
from contextlib import contextmanager, nullcontext
from tqdm import tqdm

from .agent import Agent
from ..functional import ReplayBuffer, BaselineType, SchedulerType, TDUtils, DebugTimer
from ..policies import QNetwork, EpsilonScheduler

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None


class QLearningAgent(Agent):
    """Deep Q-Learning agent."""

    def __init__(self, name,  gamma=0.99, experiment=None,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.995,
                 buffer_capacity=10000, batch_size=64, target_update=10, debug_timing=False):
        super().__init__(name=name, gamma=gamma,
                         baseline_type=BaselineType.NONE)

        self.batch_size = batch_size
        self.target_update = target_update
        self.buffer = ReplayBuffer(buffer_capacity)
        self.epsilon_scheduler = EpsilonScheduler(epsilon_start, epsilon_end, epsilon_decay)
        self.target_network = None
        self.debug_timing = debug_timing
        self.timer = DebugTimer() if debug_timing else None
        self._is_eval = False 
        if experiment is not None:
            self.set_experiment(experiment)

    def eval(self):
        self._is_eval = True
        self.epsilon_scheduler.eval()
    
    def train(self):
        self._is_eval = False
        self.epsilon_scheduler.train()

    def build_policy_network(self):
        if self.env is None:
            raise ValueError("Environment is not set. Call set_experiment() first.")
        obs_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.n
        return QNetwork(obs_dim, action_dim).to(self.device)

    def set_experiment(self, experiment, reset_policy_network=True):
        super().set_experiment(experiment, reset_policy_network)
        # Create target network as a copy of the policy network
        self.target_network = self.build_policy_network()
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.target_network.eval()

    def select_action(self, state):        
        epsilon = self.epsilon_scheduler.get_epsilon()
        if random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                q_vals = self.policy_network(state_t)
            action = int(q_vals.argmax().item())
        return action, None, None

    def _optimize_model(self):
        if len(self.buffer) < self.batch_size:
            return None
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size, device=self.device)
        loss = compute_td_loss(self.policy_network, self.target_network, 
                               (states, actions, rewards, next_states, dones),
                               self.gamma, self.optimizer)
        return loss.item()

    def _initialize_wandb(self, wandb_logging, wandb_project, wandb_run, wandb_config,
                         episodes, max_steps, eval_every, eval_episodes):
        """Initialize Weights & Biases logging if requested."""
        if not wandb_logging:
            return
        
        if not WANDB_AVAILABLE:
            raise ImportError("wandb is not installed. Please install it with 'pip install wandb'")
        
        config = {
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "episodes": episodes,
            "max_steps": max_steps,
            "eval_every": eval_every,
            "eval_episodes": eval_episodes,
            "algorithm": "DQN",
            "device": str(self.device),
            "epsilon_start": getattr(self.epsilon_scheduler, 'epsilon_start', getattr(self.epsilon_scheduler, 'start', 1.0)),
            "epsilon_end": getattr(self.epsilon_scheduler, 'epsilon_end', getattr(self.epsilon_scheduler, 'end', 0.05)),
            "epsilon_decay": getattr(self.epsilon_scheduler, 'epsilon_decay', getattr(self.epsilon_scheduler, 'decay', 0.995)),
            "buffer_capacity": self.buffer.capacity,
            "batch_size": self.batch_size,
            "target_update": self.target_update
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
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
                
                grad_metrics[f"grad_norm/{name}"] = param_norm.item()
                grad_metrics[f"grad_mean/{name}"] = param.grad.data.mean().item()
                grad_metrics[f"grad_std/{name}"] = param.grad.data.std().item()
                grad_metrics[f"grad_max/{name}"] = param.grad.data.max().item()
                grad_metrics[f"grad_min/{name}"] = param.grad.data.min().item()
                
                if torch.isnan(param.grad).any():
                    grad_metrics[f"grad_has_nan/{name}"] = 1.0
                if torch.isinf(param.grad).any():
                    grad_metrics[f"grad_has_inf/{name}"] = 1.0
        
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
            wandb.log({
                "eval/avg_reward": avg_reward,
                "eval/avg_length": avg_length,
                "episode": episode + 1
            })

        # Aggiorna e mantieni il postfix
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


    def _generate_model_description(self, episodes, max_steps, buffer_capacity, batch_size):
        """Generate a pretty description of the model and training configuration."""
        import textwrap
        
        network_str = str(self.policy_network)
        network_str = textwrap.indent(network_str, '    ')
        
        env_name = getattr(self.env, 'unwrapped', self.env).__class__.__name__
        obs_shape = self.env.observation_space.shape
        action_space = getattr(self.env.action_space, 'n', 'continuous')

        # Safely get epsilon scheduler attributes
        epsilon_start = getattr(self.epsilon_scheduler, 'epsilon_start', getattr(self.epsilon_scheduler, 'start', 1.0))
        epsilon_end = getattr(self.epsilon_scheduler, 'epsilon_end', getattr(self.epsilon_scheduler, 'end', 0.05))
        epsilon_decay = getattr(self.epsilon_scheduler, 'epsilon_decay', getattr(self.epsilon_scheduler, 'decay', 0.995))

        description = f"""
╔{'═' * 70}╗
║{' ' * 28}DQN AGENT{' ' * 33}║
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
║ {f"Buffer Capacity: {buffer_capacity}":<68} ║
║ {f"Batch Size: {batch_size}":<68} ║
║ {f"Target Update Frequency: {self.target_update}":<68} ║
║ {f"Epsilon Start: {epsilon_start}":<68} ║
║ {f"Epsilon End: {epsilon_end}":<68} ║
║ {f"Epsilon Decay: {epsilon_decay}":<68} ║
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
            return self.experiment.run_episode(agent=self, max_steps=max_steps, render=False)

    def _process_batch(self, episode_data, max_grad_norm, num_updates):
        """Store transitions and update the model."""
        with self.timer.time_block("data_preparation") if self.timer else nullcontext():
            for step in episode_data.steps:
                self.buffer.push(step)
        losses = []
        with self.timer.time_block("gradient_computation") if self.timer else nullcontext():
            for _ in range(num_updates):
                loss = self._optimize_model()
                if loss is not None:
                    losses.append(loss)
        if self.scheduler:
            self.scheduler.step()
        return losses

    def _log_metrics(self, wandb_logging, episode, episode_data, losses):
        if not wandb_logging:
            return
        log_data = {
            "episode": episode + 1,
            "episode_reward": episode_data.total_reward,
            "episode_length": len(episode_data),
            "epsilon": self.epsilon_scheduler.get_epsilon(),
            "buffer_size": len(self.buffer),
            "learning_rate": self.optimizer.param_groups[0]['lr']
        }
        if losses:
            log_data.update({
                "loss": sum(losses) / len(losses),
                "loss_std": torch.tensor(losses).std().item() if len(losses) > 1 else 0
            })
        if self.debug_timing and self.timer:
            timing_stats = self.timer.get_all_stats()
            for name, stats in timing_stats.items():
                if stats:
                    log_data[f"timing_{name}_mean_ms"] = stats['mean'] * 1000
                    log_data[f"timing_{name}_current_ms"] = stats['current'] * 1000
        wandb.log(log_data)

    def _maybe_evaluate(self, wandb_logging, episode, eval_every, eval_episodes, max_steps,
                        best_eval_reward, episodes_without_improvement, reload_best_after_episodes,
                        save_best_model_path, early_stop=False, early_stop_path=None, pbar=None):
        evaluation_result = None
        is_solved = False
        if eval_every > 0 and (episode + 1) % eval_every == 0:
            eval_result = self._perform_evaluation(wandb_logging, episode, eval_episodes, max_steps, pbar=pbar)
            evaluation_result = eval_result
            
            # Check if environment is solved and handle early stopping
            if early_stop:
                is_solved, confirmed_reward = self._check_if_solved(
                    eval_result['avg_reward'], 
                    max_steps=max_steps
                )
                self._is_solved = is_solved

                if is_solved:
                    # Save the solved model

                    save_path = early_stop_path or save_best_model_path or f"solved_model_ep{episode+1}.pt"
                    dirname = os.path.dirname(save_path)
                    if dirname:
                        os.makedirs(dirname, exist_ok=True)
                    torch.save(self.policy_network.state_dict(), save_path)
                    tqdm.write(f"Solved model saved to {save_path} with confirmed avg reward: {confirmed_reward:.2f}")
                    
                    if wandb_logging:
                        wandb.log({
                            "environment_solved": True,
                            "solved_episode": episode + 1,
                            "solved_reward": confirmed_reward
                        })

            
            if save_best_model_path and eval_result['avg_reward'] > best_eval_reward:
                old_best_reward = best_eval_reward
                best_eval_reward = eval_result['avg_reward']
                episodes_without_improvement = 0

                tqdm.write(f"\n🏆 New best model! Reward improved from {old_best_reward:.2f} to {best_eval_reward:.2f}")

                dirname = os.path.dirname(save_best_model_path)
                if dirname:  
                    os.makedirs(dirname, exist_ok=True)
                torch.save(self.policy_network.state_dict(), save_best_model_path)
                tqdm.write(f"🔄 Saved best model to {save_best_model_path}")
            else:
                episodes_without_improvement += 1
                if (reload_best_after_episodes > 0 and
                        episodes_without_improvement >= reload_best_after_episodes and
                        save_best_model_path and os.path.exists(save_best_model_path)):
                    tqdm.write(f"No improvement for {episodes_without_improvement} evaluations. Reloading best model...")
                    self.policy_network.load_state_dict(torch.load(save_best_model_path, map_location=self.device))
                    episodes_without_improvement = 0
                    if wandb_logging:
                        wandb.log({
                            "model_reloaded": True,
                            "model_reload_episode": episode + 1,
                            "best_reward_at_reload": best_eval_reward
                        })
                    tqdm.write(f"Best model reloaded from {save_best_model_path} (reward: {best_eval_reward:.2f})")
            if self.debug_timing and self.timer:
                self.timer.print_summary()
                self.experiment.print_timing_summary()
        return evaluation_result, best_eval_reward, episodes_without_improvement, is_solved

    def _check_if_solved(self, avg_reward, confirm_episodes=100, max_steps=None):
        """
        Check if the environment is considered solved based on the average reward.
        
        Args:
            avg_reward (float): The average reward from evaluation.
            solve_threshold (float): Threshold for considering the environment solved.
            confirm_episodes (int): Number of episodes to run for confirmation.
            max_steps (int, optional): Maximum steps per episode for evaluation.
            
        Returns:
            bool: True if the environment is solved, False otherwise.
            float: The confirmed average reward if solved, None otherwise.
        """
        solve_threshold = self.experiment.reward_threshold
        if avg_reward < solve_threshold:
            return False, None
            
        print(f"\n🎉 Potential solution detected! Average reward: {avg_reward:.2f}")
        print(f"Confirming solution with {confirm_episodes} additional episodes...")
        
        # Run additional episodes to confirm the solution
        self.eval()
        confirmed_avg_reward, confirmed_avg_length = self.evaluate(
            episodes=confirm_episodes, 
            max_steps=max_steps
        )
        self.train()
        
        if confirmed_avg_reward >= solve_threshold:
            print(f"\n🏆 Environment SOLVED! Confirmed average reward: {confirmed_avg_reward:.2f}")
            return True, confirmed_avg_reward
        else:
            print(f"\n⚠️ Solution not confirmed. Average reward over {confirm_episodes} episodes: {confirmed_avg_reward:.2f}")
            return False, None

    def train_online(self, episodes=500, max_steps=200, render_every=-1, eval_every=100,
                    eval_episodes=10, wandb_logging=False, wandb_project="dqn-training",
                    wandb_run=None, wandb_config=None, debug_timing=None,
                    save_best_model_path: Optional[str] = None, reload_best_after_episodes=-1,
                    log_gradients=True, log_gradients_every=10, flush_experiment_every=-1,
                    max_grad_norm=1, scheduler_type: SchedulerType = SchedulerType.STEP,
                    learning_rate=1e-3, early_stop=False, early_stop_path=None):
        """Train the agent using Deep Q-Learning.
        
        Args:
            episodes (int): Number of training episodes.
            max_steps (int): Maximum steps per episode.
            render_every (int): Render every N episodes (-1 to disable).
            eval_every (int): Evaluate every N episodes.
            eval_episodes (int): Number of episodes for evaluation.
            wandb_logging (bool): Enable Weights & Biases logging.
            wandb_project (str): W&B project name.
            wandb_run (str): W&B run name.
            wandb_config (dict): Additional W&B configuration.
            debug_timing (bool): Enable timing debugging.
            save_best_model_path (str): Path to save the best model.
            reload_best_after_episodes (int): Number of evaluation periods without improvement 
                                             before reloading best model. Set to 0 to disable.
            log_gradients (bool): Enable gradient logging.
            log_gradients_every (int): Log gradients every N episodes.
            flush_experiment_every (int): Flush experiment memory every N episodes.
            max_grad_norm (float): Maximum gradient norm for clipping.
            scheduler_type (SchedulerType): Type of LR scheduler.
            scheduler_kwargs (dict, optional): Arguments for the scheduler.
            early_stop (bool): Stop training if environment is solved (avg reward > 200).
            early_stop_path (str): Path to save the solved model. If None, uses save_best_model_path.
        """
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=self.learning_rate)
        if self._is_eval:
            self.train()
        if self.experiment is None:
            raise RuntimeError("Experiment is not set. Please set an Experiment using set_experiment() before training.")

        if debug_timing is not None:
            self.debug_timing = debug_timing
            self.timer = DebugTimer() if debug_timing else None

        self.policy_network.train()
        print(self._generate_model_description(episodes, max_steps, self.buffer.capacity, self.batch_size))

        self._initialize_wandb(wandb_logging, wandb_project, wandb_run, wandb_config,
                               episodes, max_steps, eval_every, eval_episodes)

        if isinstance(scheduler_type, tuple):
            if isinstance(scheduler_type[1], dict):
                scheduler_type, scheduler_kwargs = scheduler_type
        elif isinstance(scheduler_type, SchedulerType):
            scheduler_kwargs = {}
        self._initialize_scheduler(scheduler_type, scheduler_kwargs)

        best_eval_reward = getattr(self, 'evaluation_best', float('-inf'))
        episodes_without_improvement = 0
        evaluation_results = []
        pbar = tqdm(range(episodes), desc=f"Training {self.name}")

        for episode in pbar:
            episode_start = time.perf_counter()

            if flush_experiment_every > 0 and (episode + 1) % flush_experiment_every == 0:
                self.experiment.flush_memory()
                tqdm.write(f"Flushed experiment memory at episode {episode + 1}.")

            episode_data = self._run_episode(max_steps)

            losses = self._process_batch(episode_data, max_grad_norm,
                                        min(len(episode_data.steps), 4))

            self.epsilon_scheduler.step()
            if (self.target_network and self.target_update > 0 and (episode + 1) % self.target_update == 0):
                self.target_network.load_state_dict(self.policy_network.state_dict())

            self._log_metrics(wandb_logging, episode, episode_data, losses)

            eval_result, best_eval_reward, episodes_without_improvement, is_solved = self._maybe_evaluate(
                wandb_logging, episode, eval_every, eval_episodes, max_steps,
                best_eval_reward, episodes_without_improvement,
                reload_best_after_episodes, save_best_model_path, early_stop, early_stop_path, pbar=pbar)
            
            if eval_result:
                evaluation_results.append(eval_result)
                
            # If environment is solved and early stopping is enabled, break the training loop
            if early_stop and is_solved:
                tqdm.write(f"\n🚀 Training stopped early at episode {episode+1} as environment is considered solved!")
                break

            avg_loss = sum(losses) / len(losses) if losses else 0
            pbar.set_postfix({
                "reward": f"{episode_data.total_reward:.2f}",
                "length": len(episode_data),
                "loss": f"{avg_loss:.4f}",
                "epsilon": f"{self.epsilon_scheduler.get_epsilon():.3f}"
            })

        pbar.close()

        if self.debug_timing and self.timer:
            print("\nFINAL TRAINING TIMING SUMMARY:")
            self.timer.print_summary()

        if wandb_logging:
            wandb.finish()

        self.evaluation_results = evaluation_results
        self.evaluation_best = best_eval_reward
        return evaluation_results, best_eval_reward

    def plot_training_results(self):
        """Plot training results using seaborn."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd
        
        if not hasattr(self, 'evaluation_results') or not self.evaluation_results:
            raise RuntimeError("No evaluation results found. Run train_online() first.")
        
        df = pd.DataFrame(self.evaluation_results)
        
        sns.set_style("whitegrid")
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot average reward
        sns.lineplot(data=df, x='episode', y='avg_reward', marker='o', 
                    linewidth=2.5, markersize=6, ax=axes[0])
        axes[0].set_title('Average Reward During Training', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Episode', fontsize=12)
        axes[0].set_ylabel('Average Reward', fontsize=12)
        axes[0].grid(True, alpha=0.3)
        
        # Plot average episode length
        sns.lineplot(data=df, x='episode', y='avg_length', marker='s', color='orange',
                    linewidth=2.5, markersize=6, ax=axes[1])
        axes[1].set_title('Average Episode Length During Training', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Episode', fontsize=12)
        axes[1].set_ylabel('Average Episode Length', fontsize=12)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def load_state_dict(self, state_dict, val_to_beat=None):
        if isinstance(state_dict, str):
            if not os.path.exists(state_dict):
                raise FileNotFoundError(f"State dict file '{state_dict}' does not exist.")
            state_dict = torch.load(state_dict, map_location=self.device)
        self.policy_network.load_state_dict(state_dict)
        self.target_network.load_state_dict(state_dict)
        if val_to_beat is not None:
            self.evaluation_best = val_to_beat

def compute_td_loss(policy_net: nn.Module, target_net: nn.Module, batch, gamma: float, optimizer=None):
        """Compute and optionally apply the TD loss for a batch of transitions."""
        
        states, actions, rewards, next_states, dones = batch
        q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = target_net(next_states).max(1)[0]
            targets = rewards + gamma * (1 - dones) * next_q

        loss = F.huber_loss(q_values, targets)

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return loss
