from __future__ import annotations

from typing import Dict, List, Optional
from .data import EpisodeData
import torch

class Memory:
    def __init__(self):
        self.data: List[EpisodeData] = []

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
        self.memories: Dict[str, Memory] = {}
        self.default_memory_name: str = "default"
        self.memories[self.default_memory_name] = Memory()

    def add_memory(self, name: str) -> Memory:
        if name not in self.memories:
            self.memories[name] = Memory()
        return self.memories[name]

    def get_memory(self, name: Optional[str] = None) -> Memory:
        if name is None:
            name = self.default_memory_name
        if name not in self.memories:
            self.add_memory(name)
        return self.memories[name]

    def clear_memory(self, name: Optional[str] = None):
        if name is None:
            for memory in self.memories.values():
                memory.clear()
        elif name in self.memories:
            self.memories[name].clear()

    def get_memory_names(self) -> List[str]:
        return list(self.memories.keys())

    def __repr__(self) -> str:
        return f"MemoryManager(memories={list(self.memories.keys())})"
