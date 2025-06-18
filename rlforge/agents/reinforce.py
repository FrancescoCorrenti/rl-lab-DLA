from sympy import beta
import torch
import time
import os
from typing import Optional # Added for type hinting
from contextlib import contextmanager, nullcontext
from collections import defaultdict, deque
from tqdm import tqdm

from rlforge.experiments import Experiment
from .agent import Agent
from ..functional import BaselineType, BaselineFactory, PolicyGradientUtils, SchedulerType

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None


class REINFORCEAgent(Agent):
    def __init__(self, name, learning_rate=0.01, gamma=0.99, experiment=None, 
                 debug_timing=False, baseline_type: BaselineType = BaselineType.NONE, policy_network=None):
        """Initialize the REINFORCE agent.
        Args:
            name (str): Name of the agent.
            learning_rate (float): Learning rate for the optimizer.
            gamma (float): Discount factor for future rewards.
            experiment (Experiment, optional): Experiment instance to set the environment.
            debug_timing (bool): If True, enables detailed timing for debugging.
            baseline_type (BaselineType): Type of baseline to use for advantage estimation. Can be a tuple with (BaselineType, args).
            policy_network (nn.Module, optional): Custom policy network to use.
        """
        super().__init__(name=name, learning_rate=learning_rate, gamma=gamma, baseline_type=baseline_type)
        self.baseline = None
        self.debug_timing = debug_timing
        self.timer = PolicyGradientUtils.DebugTimer() if debug_timing else None

        if experiment is not None:
            self.set_experiment(experiment, reset_policy_network=policy_network is None)
            if policy_network:
                self.set_policy_network(policy_network)
     
  
    def select_action(self, state):
        """Select an action based on the current state using the policy network.
        """
        import torch.nn.functional as F
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_logits = self.policy_network(state_tensor)
        dist = torch.distributions.Categorical(F.softmax(action_logits, dim=-1))
        action = dist.sample()
        return action.item(), dist.log_prob(action), action_logits

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
            "algorithm": "REINFORCE",
            "device": str(self.device)
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
                # Calculate gradient norm for this parameter
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
                
                # Log individual parameter gradient norms
                grad_metrics[f"grad_norm/{name}"] = param_norm.item()
                
                # Log gradient statistics
                grad_metrics[f"grad_mean/{name}"] = param.grad.data.mean().item()
                grad_metrics[f"grad_std/{name}"] = param.grad.data.std().item()
                grad_metrics[f"grad_max/{name}"] = param.grad.data.max().item()
                grad_metrics[f"grad_min/{name}"] = param.grad.data.min().item()
                
                # Check for potential issues
                if torch.isnan(param.grad).any():
                    grad_metrics[f"grad_has_nan/{name}"] = 1.0
                if torch.isinf(param.grad).any():
                    grad_metrics[f"grad_has_inf/{name}"] = 1.0
        
        # Calculate total gradient norm
        total_norm = total_norm ** (1. / 2)
        grad_metrics["grad_norm/total"] = total_norm
        grad_metrics["grad_norm/avg_per_param"] = total_norm / max(param_count, 1)
        
        # Log all gradient metrics
        wandb.log(grad_metrics)

    def _log_training_metrics(self, wandb_logging, episode, episode_data, loss, returns):
        """Log training metrics to wandb."""
        if wandb_logging:
            log_data = {
                "episode": episode + 1,
                "episode_reward": episode_data.total_reward,
                "episode_length": len(episode_data),
                "loss": loss.item(),
                "avg_return": returns.mean().item(),
                "std_return": returns.std().item()
            }
            
            # Add timing metrics if debug timing is enabled
            if self.debug_timing and self.timer:
                timing_stats = self.timer.get_all_stats()
                for name, stats in timing_stats.items():
                    if stats:
                        log_data[f"timing_{name}_mean_ms"] = stats['mean'] * 1000
                        log_data[f"timing_{name}_current_ms"] = stats['current'] * 1000
            
            wandb.log(log_data)

    def _perform_evaluation(self, wandb_logging, episode, eval_episodes, max_steps):
        """Perform evaluation and log results."""
        # Clear previous evaluation output if not the first evaluation
        if hasattr(self, '_last_eval_output') and self._last_eval_output:
            # Move cursor up by the number of lines used in last evaluation output
            print(f"\033[{self._last_eval_output}A", end="")
            # Clear those lines
            print("\033[J", end="")
        
        lines_used = 1  # Start count with the "Evaluating" line
        print(f"Evaluating at episode {episode+1}...")
        
        avg_reward, avg_length = self.evaluate(episodes=eval_episodes, max_steps=max_steps)
        evaluation_result = {
            'episode': episode + 1,
            'avg_reward': avg_reward,
            'avg_length': avg_length
        }
        
        if wandb_logging:
            wandb.log({
                "eval_episode": episode + 1,
                "eval_avg_reward": avg_reward,
                "eval_avg_length": avg_length
            })
        
        print(f"Evaluation results: Avg Reward: {avg_reward:.2f}, Avg Length: {avg_length:.2f}")
        lines_used += 1  # Count the "Evaluation results" line
        
        # Store the number of lines used for next evaluation
        self._last_eval_output = lines_used
        
        return evaluation_result

    def _generate_model_description(self, episodes, max_steps, batch_size, entropy_beta):
        """Generate a pretty description of the model and training configuration."""
        import textwrap
        from ..functional import ValueFunctionBaseline
        
        # Get policy network architecture as string
        network_str = str(self.policy_network)
        # Format network architecture with proper indentation
        network_str = textwrap.indent(network_str, '    ')
        
        # Get environment info
        env_name = getattr(self.env, 'unwrapped', self.env).__class__.__name__
        obs_shape = self.env.observation_space.shape
        action_space = getattr(self.env.action_space, 'n', 'continuous')

        baseline_str = None
        if isinstance(self.baseline, ValueFunctionBaseline):
            # If baseline is a value function, we can provide more details
            baseline_str = str(self.baseline.value_network)
            baseline_str = textwrap.indent(baseline_str, '    ')
        # Create a pretty formatted description
        description = f"""
╔{'═' * 70}╗
║{' ' * 28}REINFORCE AGENT{' ' * 28}║
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
║ {f"Batch Size: {batch_size}":<68} ║
║ {f"Entropy Coefficient: {entropy_beta if isinstance(entropy_beta, float) else 'dynamic'}":<68} ║
║ {f"Baseline: {self.baseline_type.name if hasattr(self, 'baseline_type') else 'NONE'}":<68} ║
╠{'═' * 70}╣
║ {' ' * 26}NETWORK ARCHITECTURE{' ' * 23}║
╠{'═' * 70}╣
{textwrap.indent(network_str, '║ ').rstrip()}
║ {' ' * 26}BASELINE NETWORK{' ' * 27}
╠{'═' * 70}╣
{textwrap.indent(baseline_str, '║ ').rstrip() if baseline_str else 'None ║'} 
╚{'═' * 70}╝
"""
        return description

    def train_online(self, episodes=1000, max_steps=200, render_every=-1, eval_every=100, 
                    eval_episodes=10, wandb_logging=False, wandb_project="reinforce-training", 
                    wandb_run=None, wandb_config=None, debug_timing=None, save_best_model_path: Optional[str] = None,
                    save_best_value_model_path: Optional[str] = None, entropy_beta=0.01, reload_best_after_episodes=-1, log_gradients=True, 
                    log_gradients_every=10, batch_size=1, flush_experiment_every=-1, max_grad_norm=1, scheduler_type: SchedulerType = SchedulerType.CYCLIC,
                    scheduler_kwargs: Optional[dict] = None): # Added scheduler params
        """Train the agent using the REINFORCE algorithm.
        
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
            entropy_beta (float): Entropy regularization coefficient.
            reload_best_after_episodes (int): Number of evaluation periods without improvement 
                                             before reloading best model. Set to 0 to disable.
            log_gradients (bool): Enable gradient logging.
            log_gradients_every (int): Log gradients every N episodes.
            batch_size (int): Number of episodes to accumulate before performing an update.
            flush_experiment_every (int): Flush experiment memory every N episodes to clear old data.
            max_grad_norm (float): Maximum gradient norm for clipping.
            scheduler_type (SchedulerType): Type of LR scheduler.
            scheduler_kwargs (dict, optional): Arguments for the scheduler.
        """
        if self.experiment is None:
            raise RuntimeError("Experiment is not set. Please set an Experiment using set_experiment() before training.")
        
        # Override debug timing if specified
        if debug_timing is not None:
            self.debug_timing = debug_timing
            self.timer = PolicyGradientUtils.DebugTimer() if debug_timing else None

        self.policy_network.train()
        
        # Print model and training description
        print(self._generate_model_description(episodes, max_steps, batch_size, entropy_beta))

        # Initialize wandb
        self._initialize_wandb(wandb_logging, wandb_project, wandb_run, wandb_config,
                               episodes, max_steps, eval_every, eval_episodes)
        
        # Initialize scheduler
        scheduler_kwargs = scheduler_kwargs or {}
        if scheduler_type == SchedulerType.CYCLIC:
            scheduler_kwargs.setdefault('base_lr', 1e-5)
            scheduler_kwargs.setdefault('max_lr', self.learning_rate)
            scheduler_kwargs.setdefault('step_size_up', 100)
            scheduler_kwargs.setdefault('mode', 'triangular2')
        elif scheduler_type == SchedulerType.STEP:
            scheduler_kwargs.setdefault('step_size', 100)
            scheduler_kwargs.setdefault('gamma', 0.5)
        self._initialize_scheduler(scheduler_type, scheduler_kwargs)

        best_eval_reward = self.evaluation_best if hasattr(self, 'evaluation_best') else float('-inf')
        episodes_without_improvement = 0
        evaluation_results = []
        pbar = tqdm(range(episodes), desc="Training", unit="episode")
        
        # For batching
        batch_log_probs = []
        batch_rewards = []
        batch_returns = []
        batch_logits = []
        batch_states = []
        batch_actions = []
        batch_rewards_sum = 0
        batch_lengths_sum = 0
        
        for episode in pbar:
            episode_start = time.perf_counter()

            if flush_experiment_every > 0 and (episode + 1) % flush_experiment_every == 0:
                # Flush the experiment memory to clear old data
                self.experiment.flush_memory()
                tqdm.write(f"Flushed experiment memory at episode {episode + 1}.")
            
            # Run episode with timing
            with self.timer.time_block("episode_execution") if self.timer else nullcontext():
                episode_data = self.experiment.run_episode(agent=self, max_steps=max_steps, render=False)
            
            # Optional rendering
            if render_every > 0 and (episode + 1) % render_every == 0:
                with self.timer.time_block("rendering") if self.timer else nullcontext():
                    self.policy_network.eval()
                    self.experiment.run_episode(agent=self, max_steps=max_steps, render=True)
                    self.policy_network.train()

            # Compute returns with timing
            with self.timer.time_block("data_preparation") if self.timer else nullcontext():
                logits_ep = episode_data.get_logits(return_pt=True).to(self.device)
                log_probs_ep = episode_data.get_log_probs(return_pt=True).to(self.device)
                
                
                if self.baseline is None:
                    rewards_ep = episode_data.get_rewards(return_pt=True).to(self.device)
                    returns_ep = PolicyGradientUtils.compute_discounted_returns(rewards_ep, self.gamma)
                        
                else:
                    rewards_ep = episode_data.get_rewards(return_pt=True).to(self.device)
                    returns_ep = self.baseline(episode_data, train=False)
                    if save_best_model_path and self.baseline_type == BaselineType.VALUE_FUNCTION:
                        save_best_value_model_path = save_best_model_path.replace(".pt", "_baseline.pt")

          
                returns_ep = returns_ep.to(self.device)

                batch_log_probs.append(log_probs_ep)
                batch_returns.append(returns_ep)
                batch_logits.append(logits_ep)
                batch_rewards.append(rewards_ep)
                batch_actions.append(episode_data.get_actions(return_pt=True).to(self.device))
                batch_states.append(episode_data.get_states(return_pt=True).to(self.device))
                batch_rewards_sum += episode_data.total_reward
                batch_lengths_sum += len(episode_data)

            # Perform update if batch is full or it's the last episode
            if (episode + 1) % batch_size == 0 or (episode + 1) == episodes:
                if not batch_log_probs: # Skip if batch is empty (e.g. last episode was already processed)
                    continue

                # Concatenate data from the batch
                all_log_probs = torch.cat(batch_log_probs)
                all_returns = torch.cat(batch_returns)
                all_rewards = torch.cat(batch_rewards)
                all_logits = torch.cat(batch_logits)
                all_states = torch.cat(batch_states)
                all_actions = torch.cat(batch_actions)

                # Update policy with timing
                with self.timer.time_block("gradient_computation") if self.timer else nullcontext():           
                    if torch.isnan(all_log_probs).any() or torch.isnan(all_returns).any():
                        print(f"NaN detected in batch ending at episode {episode + 1} inputs. Skipping update.")
                        if torch.isnan(all_log_probs).any():
                            print(f"Log probabilities contain NaN values.")
                        if torch.isnan(all_returns).any():
                            print(f"Returns contain NaN values.")
                        # Clear batch for next iteration
                        batch_log_probs, batch_returns, batch_logits = [], [], []
                        batch_rewards_sum, batch_lengths_sum = 0, 0
                        continue
                        
                    loss = (-all_log_probs * all_returns).mean()
                    
                    # Add entropy regularization with stability checks
                    if all_logits is not None:
                        entropy = PolicyGradientUtils.compute_entropy(all_logits)
                        current_beta = entropy_beta if isinstance(entropy_beta, float) else entropy_beta(episode // batch_size) # Use batch index for beta schedule
                        
                        if not torch.isnan(entropy).any() and not torch.isinf(entropy).any():
                            loss -= current_beta * entropy
                        else:
                            print(f"Invalid entropy for batch ending at episode {episode + 1}. Skipping entropy regularization.")

                with self.timer.time_block("optimizer_step") if self.timer else nullcontext():
                    from rlforge.functional import EpisodeData
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.baseline.train_step(EpisodeData.from_values(
                        log_probs=all_log_probs,
                        rewards=all_rewards,
                        states=all_states,
                    )) if self.baseline else None
                    torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_grad_norm)
                    self.optimizer.step()
                   
               
                
                # Log metrics for the batch
                avg_batch_reward = batch_rewards_sum / len(batch_log_probs)
                avg_batch_length = batch_lengths_sum / len(batch_log_probs)
                if wandb_logging:
                    log_data = {
                        "episode": episode + 1,  # Log the actual episode number
                        "batch_update_episode": episode + 1, # Log at the end of the batch
                        "avg_episode_reward_in_batch": avg_batch_reward,
                        "avg_episode_length_in_batch": avg_batch_length,
                        "loss": loss.item(),
                        "avg_return_in_batch": all_returns.mean().item(),
                        "std_return_in_batch": all_returns.std().item(),
                        "batch_size_actual": len(batch_log_probs),  # Actual number of episodes in this batch
                        "avg_entropy": entropy.item() if 'entropy' in locals() else None,
                        "batch_entropy_beta": current_beta if 'current_beta' in locals() else None,
                        "batch_entropy": entropy.item() if 'entropy' in locals() else None,
                        "learning_rate": self.optimizer.param_groups[0]['lr'],
                        "batch_action": all_actions                  
                    }
                    
                 
                    wandb.log(log_data)

                # Log gradients if enabled and at the right interval
                if log_gradients and wandb_logging and (episode + 1) % (log_gradients_every * batch_size) == 0:
                    self._log_gradient_metrics(wandb_logging, episode + 1)

                # Clear batch for next iteration
                batch_log_probs, batch_returns, batch_logits, batch_rewards, batch_states = [], [], [], [], []
                batch_rewards_sum, batch_lengths_sum = 0, 0
                
                # Update progress bar with timing info (using last episode's data for simplicity or batch average)
                episode_time = time.perf_counter() - episode_start
                desc = (f"Episode {episode+1} (Batch End) | Avg Reward (Batch): {avg_batch_reward:.2f} | "
                       f"Loss: {loss.item():.4f} | Avg Return (Batch): {all_returns.mean().item():.2f}")
            else: # Not end of batch yet
                episode_time = time.perf_counter() - episode_start
                desc = (f"Episode {episode+1} | Reward: {episode_data.total_reward:.2f} | "
                       f"Return: {returns_ep.mean().item():.2f} (Collecting for batch)")


            if self.debug_timing:
                desc += f" | Time: {episode_time*1000:.1f}ms"
            
            if reload_best_after_episodes > 0 and save_best_model_path:
                desc += f" | No improve: {episodes_without_improvement}/{reload_best_after_episodes}"
            
            pbar.set_description(desc)

            if eval_every > 0 and (episode + 1) % eval_every == 0:
                with self.timer.time_block("evaluation") if self.timer else nullcontext():
                    eval_result = self._perform_evaluation(wandb_logging, episode, eval_episodes, max_steps)
                    evaluation_results.append(eval_result)
                    # Check if we have a new best model
                    if save_best_model_path and eval_result['avg_reward'] > best_eval_reward:
                        best_eval_reward = eval_result['avg_reward']
                        episodes_without_improvement = 0  # Reset counter on improvement
                        current_beta = eval_result['entropy_beta'] if 'entropy_beta' in eval_result else None
                        # Ensure the directory exists
                        save_dir = os.path.dirname(save_best_model_path)
                        if save_dir and not os.path.exists(save_dir):
                            os.makedirs(save_dir, exist_ok=True)
                        
                        
                        torch.save(self.policy_network.state_dict(), save_best_model_path)
                        if self.baseline and self.baseline_type == BaselineType.VALUE_FUNCTION:
                            torch.save(self.baseline.value_network.state_dict(), save_best_value_model_path)
                        tqdm.write(f"New best model saved to {save_best_model_path} with avg reward: {best_eval_reward:.2f}")
                    else:
                        # No improvement - increment counter
                        episodes_without_improvement += 1
                        
                        # Check if we should reload the best model
                        if (reload_best_after_episodes > 0 and 
                            episodes_without_improvement >= reload_best_after_episodes and 
                            save_best_model_path and os.path.exists(save_best_model_path)):
                            
                            tqdm.write(f"No improvement for {episodes_without_improvement} evaluations. Reloading best model...")
                            self.policy_network.load_state_dict(torch.load(save_best_model_path, map_location=self.device))
                            episodes_without_improvement = 0  # Reset counter after reloading
                            
                            if wandb_logging:
                                wandb.log({
                                    "model_reloaded": True,
                                    "model_reload_episode": episode + 1,
                                    "best_reward_at_reload": best_eval_reward
                                })
                            
                            tqdm.write(f"Best model reloaded from {save_best_model_path} (reward: {best_eval_reward:.2f})")
                
                # Print timing summary during evaluation
                if self.debug_timing and self.timer:
                    self.timer.print_summary()
                    self.experiment.print_timing_summary()
        
        pbar.close()
        
        # Final timing summary
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

