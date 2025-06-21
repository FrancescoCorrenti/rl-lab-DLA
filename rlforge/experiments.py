import torch
import numpy as np
from typing import TYPE_CHECKING
import numpy as np
import time
from contextlib import contextmanager, nullcontext

if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

if TYPE_CHECKING:
    from .agents.agent import Agent
    from .agents.reinforce import REINFORCEAgent

from gymnasium.vector import SyncVectorEnv
from .data import StepData, EpisodeData
from .memory import Memory, MemoryManager


class Experiment:
    def __init__(self, env, env_renderer, name=None, description=None, debug_timing=False, solve_threshold=200):
        self.env = env
        self.env_renderer = env_renderer
        self.name = name
        self.description = description
        self.memory_manager = MemoryManager()
        self.debug_timing = debug_timing
        self.timer = None
        if debug_timing:
            from .functional import DebugTimer
            self.timer = DebugTimer()
        self._reward_threshold = solve_threshold
    
    @property
    def memory(self):
        """Get default memory."""
        return self.memory_manager.get_memory()
    
    def set_default_memory(self, name):
        """Set the default memory to a specific named memory."""
        if name not in self.memory_manager.memories:
            self.memory_manager.add_memory(name)
        self.memory_manager.default_memory_name = name
        
    def get_memory(self, name=None):
        """Get memory by name."""
        return self.memory_manager.get_memory(name)
    
    def add_memory(self, name):
        """Add a new named memory."""
        return self.memory_manager.add_memory(name)

    def run_episode(self, agent: 'Agent', max_steps=200, render=False, memory_name=None, debug_timing=None, store_in_memory=False):
        """
        Run a single episode of the experiment.
        """
        # Override debug timing if specified
        if debug_timing is not None:
            self.debug_timing = debug_timing
            if debug_timing and self.timer is None:
                from .functional import DebugTimer
                self.timer = DebugTimer()

        if agent.experiment is not self:
            raise ValueError(f"Agent's experiment does not match this Experiment instance. Please set the agent's experiment using agent.set_experiment(self).")
        
        # Use env_renderer if rendering, otherwise use env
        env_to_use = self.env_renderer if render else self.env
        
        with self.timer.time_block("env_reset") if self.timer else nullcontext():
            state, _ = env_to_use.reset()

        done = False
        memory = self.memory_manager.get_memory(memory_name)
        episode_data = EpisodeData()
        
        for step in range(max_steps):
            with self.timer.time_block("action_selection") if self.timer else nullcontext():
                action_result = agent.select_action(state)
                action, log_prob, logits = action_result
             
            with self.timer.time_block("env_step") if self.timer else nullcontext():
                next_state, reward, done, truncated, _ = env_to_use.step(action.item() if isinstance(action, torch.Tensor) else action)
                done = done or truncated
            with self.timer.time_block("data_collection") if self.timer else nullcontext():
                step_data = StepData(state, action, reward, next_state, done, log_prob, logits)
                episode_data.add_step(step_data)

            if done:
                break

            state = next_state
        if store_in_memory:
            with self.timer.time_block("memory_storage") if self.timer else nullcontext():
                memory.add(episode_data)
        
        return episode_data

    def save_video(self, agent: 'Agent', max_steps=200, path=None):
        """
        Renders a video of the agent's performance for one episode using `self.env_renderer`.
        To save the video, `self.env_renderer` should be a `gym.wrappers.RecordVideo` instance.
        The video is saved when `self.env_renderer.close()` is called. This function handles that.

        Args:
            agent (Agent): The agent to evaluate.
            max_steps (int): Maximum steps per episode.
        """
        from gymnasium.wrappers import RecordVideo
        if self.env_renderer is None:
            print("Warning: env_renderer is not set. Cannot render video.")
            return
        if not isinstance(self.env_renderer, RecordVideo):
            self.env_renderer = RecordVideo(self.env_renderer, path or f"videos/{self.name}_episode", episode_trigger=lambda ep: True)

        print("Rendering episode...")
        episode_data = self.run_episode(agent=agent, max_steps=max_steps, render=True, store_in_memory=False)
        
        self.env_renderer.close()
        
        print(f"Episode finished. Total reward: {episode_data.total_reward}. ")

    def get_rewards(self, memory_name=None, return_pt=False, flatten=False):
        """
        Get rewards from the specified memory.
        """
        memory = self.memory_manager.get_memory(memory_name)
        return memory.get_rewards(return_pt=return_pt, flatten=flatten)

    def get_states(self, memory_name=None, return_pt=False, flatten=False):
        """
        Get states from the specified memory.
        """
        memory = self.memory_manager.get_memory(memory_name)
        return memory.get_states(return_pt=return_pt, flatten=flatten)

    def get_actions(self, memory_name=None, return_pt=False, flatten=False):
        """
        Get actions from the specified memory.
        """
        memory = self.memory_manager.get_memory(memory_name)
        return memory.get_actions(return_pt=return_pt, flatten=flatten)

    def get_next_states(self, memory_name=None, return_pt=False, flatten=False):
        """
        Get next states from the specified memory.
        """
        memory = self.memory_manager.get_memory(memory_name)
        return memory.get_next_states(return_pt=return_pt, flatten=flatten)

    def get_dones(self, memory_name=None, return_pt=False, flatten=False):
        """
        Get done flags from the specified memory.
        """
        memory = self.memory_manager.get_memory(memory_name)
        return memory.get_dones(return_pt=return_pt, flatten=flatten)

    def get_log_probs(self, memory_name=None, return_pt=False, flatten=False):
        """
        Get log probabilities from the specified memory.
        """
        memory = self.memory_manager.get_memory(memory_name)
        return memory.get_log_probs(return_pt=return_pt, flatten=flatten)

    def get_distributions(self, memory_name=None, return_pt=False, flatten=False):
        """
        Get distributions from the specified memory.
        """
        memory = self.memory_manager.get_memory(memory_name)
        return memory.get_distributions(return_pt=return_pt, flatten=flatten)

    def get_episode_rewards(self, memory_name=None, return_pt=False):
        """
        Get total rewards for each episode.
        """
        memory = self.memory_manager.get_memory(memory_name)
        return memory.get_episode_rewards(return_pt=return_pt)

    def get_episode_lengths(self, memory_name=None, return_pt=False):
        """
        Get length of each episode.
        """
        memory = self.memory_manager.get_memory(memory_name)
        return memory.get_episode_lengths(return_pt=return_pt)
    
    def get_memory_names(self):
        """ Get list of all memory names.
        """
        return self.memory_manager.get_memory_names()
    
    def clear_memory(self, name=None):
        """
        Clear a specific memory or all memories if no name provided.
        """
        self.memory_manager.clear_memory(name)

    def print_timing_summary(self):
        """Print timing summary if debug timing is enabled."""
        if self.debug_timing and self.timer:
            print("\nEXPERIMENT TIMING SUMMARY:")
            self.timer.print_summary()

    def get_timing_stats(self):
        """Get timing statistics if debug timing is enabled."""
        if self.debug_timing and self.timer:
            return self.timer.get_all_stats()
        return {}

    def __repr__(self):
        return (f"Experiment(name={self.name}, description={self.description}, "
                f"env={self.env.spec.id if self.env else None}, "
                f"memory_manager={self.memory_manager}, "
                f"debug_timing={self.debug_timing})")

    def flush_memory(self, memory_name=None):
        """
        Flush the specified memory or the default memory.
        """
        memory = self.memory_manager.get_memory(memory_name or self.memory_manager.default_memory_name)
        if memory:
            memory.clear()
        else:
            raise ValueError(f"Memory '{memory_name}' does not exist.")
        
    @property
    def reward_threshold(self):
        """
        Get the reward threshold for solving the environment.
        This can be used to determine if the environment is solved.
        """
        return self._reward_threshold