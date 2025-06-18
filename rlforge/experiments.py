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

class StepData:
    def __init__(self, state, action, reward, next_state, done, log_prob=None, logits=None):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done
        self.log_prob = log_prob
        self.logits = logits  # Added logits

    def __repr__(self):
        return (f"StepData(state={self.state}, action={self.action}, reward={self.reward}, "
                f"next_state={self.next_state}, done={self.done}, log_prob={self.log_prob}, "
                f"logits={self.logits})")

class EpisodeData:
    def __init__(self):
        self.steps: list[StepData] = []
        self.total_reward = 0
        self.episode_length = 0

    @staticmethod
    def from_values(log_probs = None, states=None, actions=None, rewards=None, next_states=None, dones=None):
        """
        Create an EpisodeData instance from lists of values.
        All lists must be of the same length.
        """
        lists = [states, actions, rewards, next_states, dones, log_probs]
        lists_names = ['states', 'actions', 'rewards', 'next_states', 'dones', 'log_probs']
        lengths = {name: len(lst) for name, lst in zip(lists_names, lists) if lst is not None}

        effective_episode_length = 0
        if lengths: # If any list was provided
            if len(set(lengths.values())) > 1: # If provided lists have different lengths
                print(f"Warning: Input lists have different lengths: ")
                for name, length in lengths.items():
                    print(f"  {name}: {length}")

                min_len = min(lengths.values())
                # Truncate all provided lists to the minimum length
                states = states[:min_len] if states is not None else None
                actions = actions[:min_len] if actions is not None else None
                rewards = rewards[:min_len] if rewards is not None else None
                next_states = next_states[:min_len] if next_states is not None else None
                dones = dones[:min_len] if dones is not None else None
                log_probs = log_probs[:min_len] if log_probs is not None else None
                effective_episode_length = min_len
            else:
                # All provided lists have the same length
                effective_episode_length = list(lengths.values())[0]
        # If lengths is empty (no lists provided), effective_episode_length remains 0.

        # The problematic line `lengths = [min_length]` has been removed.
        # `effective_episode_length` is now correctly determined.

        states, actions, rewards, next_states, log_probs, dones = (
                states      if states      is not None else [None]  * effective_episode_length,
                actions     if actions     is not None else [None]  * effective_episode_length,
                rewards     if rewards     is not None else [None]  * effective_episode_length,
                next_states if next_states is not None else [None]  * effective_episode_length,
                log_probs   if log_probs   is not None else [None]  * effective_episode_length,
                dones       if dones       is not None else [False] * effective_episode_length,
            )

        episode_data = EpisodeData()
        for i in range(effective_episode_length): # Iterate up to the determined effective length
            step_data = StepData(
                state=states[i],
                action=actions[i],
                reward=rewards[i],
                next_state=next_states[i],
                done=dones[i],
                log_prob=log_probs[i],
            )
            episode_data.add_step(step_data)
        
        return episode_data

    def add_step(self, step_data: StepData):
        self.steps.append(step_data)
        self.total_reward += float(step_data.reward)  # Ensure Python float
        self.episode_length += 1

    def get_rewards(self, return_pt=False):
        if return_pt:
            rewards = [step.reward for step in self.steps]
            return torch.stack([torch.as_tensor(r, dtype=torch.float32) for r in rewards]) if rewards else torch.empty(0)
        return [step.reward for step in self.steps]

    def get_states(self, return_pt=False):
        if return_pt:
            states = [step.state for step in self.steps]
            return torch.stack([torch.as_tensor(s, dtype=torch.float32) for s in states]) if states else torch.empty(0)
        return [step.state for step in self.steps]

    def get_actions(self, return_pt=False):
        if return_pt:
            actions = [step.action for step in self.steps]
            return torch.stack([torch.as_tensor(a, dtype=torch.float32) for a in actions]) if actions else torch.empty(0)
        return [step.action for step in self.steps]

    def get_next_states(self, return_pt=False):
        if return_pt:
            next_states = [step.next_state for step in self.steps]
            return torch.stack([torch.as_tensor(ns, dtype=torch.float32) for ns in next_states]) if next_states else torch.empty(0)
        return [step.next_state for step in self.steps]

    def get_dones(self, return_pt=False):
        if return_pt:
            dones = [step.done for step in self.steps]
            return torch.stack([torch.as_tensor(float(d), dtype=torch.float32) for d in dones]) if dones else torch.empty(0)
        return [step.done for step in self.steps]

    def get_log_probs(self, return_pt=False):
        log_probs = [step.log_prob for step in self.steps]
        if return_pt:
            return torch.stack(log_probs)
        return log_probs

    def get_logits(self, return_pt=False):
        logits = [step.logits for step in self.steps if step.logits is not None]
        if return_pt:
            # Try to stack if tensor, else return as is
            try:
                return torch.stack(logits)
            except Exception:
                return logits
        return logits

    def __len__(self):
        return len(self.steps)

    def __getitem__(self, idx):
        return self.steps[idx]

    def __repr__(self):
        return f"EpisodeData(steps={len(self.steps)}, total_reward={self.total_reward})"
    
class Memory:
    def __init__(self):
        self.data : list[EpisodeData] = []

    def add(self, episode_data: EpisodeData):
        self.data.append(episode_data)

    def clear(self):
        self.data = []

    def get_rewards(self, return_pt=False, flatten=False):
        if flatten:
            rewards = []
            for episode in self.data:
                rewards.extend(episode.get_rewards())
            if return_pt:
                return torch.stack([torch.as_tensor(r, dtype=torch.float32) for r in rewards]) if rewards else torch.empty(0)
            return rewards
        else:
            if return_pt:
                return [episode.get_rewards(return_pt=True) for episode in self.data]
            return [episode.get_rewards() for episode in self.data]

    def get_states(self, return_pt=False, flatten=False):
        if flatten:
            states = []
            for episode in self.data:
                states.extend(episode.get_states())
            if return_pt:
                return torch.stack([torch.as_tensor(s, dtype=torch.float32) for s in states]) if states else torch.empty(0)
            return states
        else:
            if return_pt:
                return [episode.get_states(return_pt=True) for episode in self.data]
            return [episode.get_states() for episode in self.data]

    def get_actions(self, return_pt=False, flatten=False):
        if flatten:
            actions = []
            for episode in self.data:
                actions.extend(episode.get_actions())
            if return_pt:
                return torch.stack([torch.as_tensor(a, dtype=torch.float32) for a in actions]) if actions else torch.empty(0)
            return actions
        else:
            if return_pt:
                return [episode.get_actions(return_pt=True) for episode in self.data]
            return [episode.get_actions() for episode in self.data]

    def get_next_states(self, return_pt=False, flatten=False):
        if flatten:
            next_states = []
            for episode in self.data:
                next_states.extend(episode.get_next_states())
            if return_pt:
                return torch.stack([torch.as_tensor(ns, dtype=torch.float32) for ns in next_states]) if next_states else torch.empty(0)
            return next_states
        else:
            if return_pt:
                return [episode.get_next_states(return_pt=True) for episode in self.data]
            return [episode.get_next_states() for episode in self.data]

    def get_dones(self, return_pt=False, flatten=False):
        if flatten:
            dones = []
            for episode in self.data:
                dones.extend(episode.get_dones())
            if return_pt:
                return torch.stack([torch.as_tensor(float(d), dtype=torch.float32) for d in dones]) if dones else torch.empty(0)
            return dones
        else:
            if return_pt:
                return [episode.get_dones(return_pt=True) for episode in self.data]
            return [episode.get_dones() for episode in self.data]

    def get_log_probs(self, return_pt=False, flatten=False):
        if flatten:
            log_probs = []
            for episode in self.data:
                log_probs.extend(episode.get_log_probs())
            if return_pt:
                return torch.stack(log_probs)
            return log_probs
        else:
            if return_pt:
                return [episode.get_log_probs(return_pt=True) for episode in self.data]
            return [episode.get_log_probs() for episode in self.data]

    def get_distributions(self, return_pt=False, flatten=False):
        if flatten:
            distributions = []
            for episode in self.data:
                distributions.extend(episode.get_distributions())
            if return_pt:
                try:
                    return torch.stack(distributions)
                except Exception:
                    return distributions
            return distributions
        else:
            if return_pt:
                return [episode.get_distributions(return_pt=True) for episode in self.data]
            return [episode.get_distributions() for episode in self.data]

    def get_episode_rewards(self, return_pt=False):
        rewards = [episode.total_reward for episode in self.data]
        if return_pt:
            return torch.as_tensor(rewards, dtype=torch.float32)
        return rewards

    def get_episode_lengths(self, return_pt=False):
        lengths = [episode.episode_length for episode in self.data]
        if return_pt:
            return torch.as_tensor(lengths, dtype=torch.int32)
        return lengths
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def __repr__(self):
        return f"Memory(episodes={len(self.data)})"


class MemoryManager:
    def __init__(self):
        self.memories = {}
        self.default_memory_name = "default"
        self.memories[self.default_memory_name] = Memory()
    
    def add_memory(self, name):
        """Add a new memory with the given name."""
        if name not in self.memories:
            self.memories[name] = Memory()
        return self.memories[name]
    
    def get_memory(self, name=None):
        """Get memory by name. If no name provided, returns default memory."""
        if name is None:
            name = self.default_memory_name
        if name not in self.memories:
            self.add_memory(name)
        return self.memories[name]
    
    def clear_memory(self, name=None):
        """Clear a specific memory or all memories if no name provided."""
        if name is None:
            for memory in self.memories.values():
                memory.clear()
        elif name in self.memories:
            self.memories[name].clear()       
    
    def get_memory_names(self):
        """Get list of all memory names."""
        return list(self.memories.keys())
    
    def __repr__(self):
        return f"MemoryManager(memories={list(self.memories.keys())})"


class Experiment:
    def __init__(self, env, env_renderer, name=None, description=None, debug_timing=False):
        self.env = env
        self.env_renderer = env_renderer
        self.name = name
        self.description = description
        self.memory_manager = MemoryManager()
        self.debug_timing = debug_timing
        self.timer = None
        if debug_timing:
            from .functional import PolicyGradientUtils
            self.timer = PolicyGradientUtils.DebugTimer()
    
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

    def run_episode(self, agent: 'Agent', max_steps=200, render=False, memory_name=None, debug_timing=None, store_in_memory=True):
        """
        Run a single episode of the experiment.
        """
        # Override debug timing if specified
        if debug_timing is not None:
            self.debug_timing = debug_timing
            if debug_timing and self.timer is None:
                from .functional import PolicyGradientUtils
                self.timer = PolicyGradientUtils.DebugTimer()
        
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
                next_state, reward, done, truncated, _ = env_to_use.step(action)
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