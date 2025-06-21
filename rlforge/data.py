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
    
    def summary(self):
        """Return a summary of the episode data."""
        return {
            "total_reward": self.total_reward,
            "episode_length": self.episode_length,
            "num_steps": len(self.steps),
            "average_reward_per_step": self.total_reward / self.episode_length if self.episode_length > 0 else 0.0,
            "average_log_prob": torch.mean(torch.stack(self.get_log_probs())) if self.get_log_probs() else 0.0,
        }
