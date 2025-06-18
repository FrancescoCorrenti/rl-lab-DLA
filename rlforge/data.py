from dataclasses import dataclass, field
from typing import Any, List, Optional
import torch

@dataclass
class StepData:
    state: Any
    action: Any
    reward: Any
    next_state: Any
    done: bool
    log_prob: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None

    def __repr__(self) -> str:
        return (
            f"StepData(state={self.state}, action={self.action}, reward={self.reward}, "
            f"next_state={self.next_state}, done={self.done}, log_prob={self.log_prob}, "
            f"logits={self.logits})"
        )

@dataclass
class EpisodeData:
    steps: List[StepData] = field(default_factory=list)
    total_reward: float = 0.0
    episode_length: int = 0

    @staticmethod
    def from_values(log_probs=None, states=None, actions=None, rewards=None, next_states=None, dones=None):
        """Create an EpisodeData instance from lists of values."""
        lists = [states, actions, rewards, next_states, dones, log_probs]
        lists_names = ['states', 'actions', 'rewards', 'next_states', 'dones', 'log_probs']
        lengths = {name: len(lst) for name, lst in zip(lists_names, lists) if lst is not None}

        effective_episode_length = 0
        if lengths:
            if len(set(lengths.values())) > 1:
                print("Warning: Input lists have different lengths: ")
                for name, length in lengths.items():
                    print(f"  {name}: {length}")
                min_len = min(lengths.values())
                states = states[:min_len] if states is not None else None
                actions = actions[:min_len] if actions is not None else None
                rewards = rewards[:min_len] if rewards is not None else None
                next_states = next_states[:min_len] if next_states is not None else None
                dones = dones[:min_len] if dones is not None else None
                log_probs = log_probs[:min_len] if log_probs is not None else None
                effective_episode_length = min_len
            else:
                effective_episode_length = list(lengths.values())[0]

        states, actions, rewards, next_states, log_probs, dones = (
            states if states is not None else [None] * effective_episode_length,
            actions if actions is not None else [None] * effective_episode_length,
            rewards if rewards is not None else [None] * effective_episode_length,
            next_states if next_states is not None else [None] * effective_episode_length,
            log_probs if log_probs is not None else [None] * effective_episode_length,
            dones if dones is not None else [False] * effective_episode_length,
        )

        episode_data = EpisodeData()
        for i in range(effective_episode_length):
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
        self.total_reward += float(step_data.reward)
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
            try:
                return torch.stack(logits)
            except Exception:
                return logits
        return logits

    def __len__(self):
        return len(self.steps)

    def __getitem__(self, idx):
        return self.steps[idx]

    def __repr__(self) -> str:
        return f"EpisodeData(steps={len(self.steps)}, total_reward={self.total_reward})"
