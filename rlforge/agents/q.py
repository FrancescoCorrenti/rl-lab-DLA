import random
import torch

from .agent import Agent
from ..functional import ReplayBuffer, compute_td_loss, BaselineType
from ..policies import QNetwork, EpsilonScheduler


class QLearningAgent(Agent):
    """Deep Q-Learning agent."""

    def __init__(self, name, learning_rate=1e-3, gamma=0.99, experiment=None,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.995,
                 buffer_capacity=10000, batch_size=64, target_update=10):
        super().__init__(name=name, learning_rate=learning_rate, gamma=gamma,
                         baseline_type=BaselineType.NONE)

        self.batch_size = batch_size
        self.target_update = target_update
        self.buffer = ReplayBuffer(buffer_capacity)
        self.epsilon_scheduler = EpsilonScheduler(epsilon_start, epsilon_end, epsilon_decay)
        self.target_network = None

        if experiment is not None:
            self.set_experiment(experiment)

    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    def _optimize_model(self):
        if len(self.buffer) < self.batch_size:
            return None
        batch = self.buffer.sample(self.batch_size, device=self.device)
        loss = compute_td_loss(self.policy_network, self.target_network, batch,
                               self.gamma, self.optimizer)
        return loss.item()

    def train_online(self, episodes=500, max_steps=200, render=False):
        if self.experiment is None:
            raise RuntimeError(
                "Experiment is not set. Please set an Experiment using set_experiment() before training."
            )

        rewards = []
        for episode in range(episodes):
            episode_data = self.experiment.run_episode(
                agent=self, max_steps=max_steps, render=render
            )

            for step in episode_data.steps:
                self.buffer.push(
                    step.state,
                    step.action,
                    step.reward,
                    step.next_state,
                    step.done,
                )
                self._optimize_model()

            # Epsilon decay and target network update
            self.epsilon_scheduler.step()
            if (
                self.target_network
                and self.target_update > 0
                and (episode + 1) % self.target_update == 0
            ):
                self.target_network.load_state_dict(self.policy_network.state_dict())

            rewards.append(episode_data.total_reward)

        return rewards

