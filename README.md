# Deep Learning Applications Laboratory #2 - README
 
 ## Overview
 
 This lab explores advanced Deep Reinforcement Learning (DRL) techniques on **CartPole-v1** and **LunarLander-v3**.  
 The goals were to refactor a basic `REINFORCE` agent, experiment with several baselines, and implement a `Deep Q-Network (DQN)` for comparison.  

 ---
 #### Project Overview
The code is divided into several modules, the main are: 
- **Agents**: Contains implementations of various DRL agents and their training logic.
- **Experiments**: Manages the environment related elements, such as rendering and playing episodes.
- **Functional**: Contains utility functions and types.
- **Data**: Handles data storage and retrieval for experiments using the classes `StepData` and `EpisodeData`.

## Exercise 1: Improving the REINFORCE Implementation

 #### Models Key Features
 I implemented a `REINFORCEAgent` class and a `QLearningAgent` class, which are the main components of the lab. 
 The `REINFORCEAgent` is a policy gradient agent that uses the REINFORCE algorithm to learn a policy for a given environment. The `QLearningAgent` is a DQN agent that uses a neural network to approximate the Q-value function. Its main features are:
- **Episodic Training**: Trains the agents over multiple episodes, collecting data and updating the policy after each episode.
- **Entropy Regularization**: Adds an entropy bonus to the loss to encourage exploration.
- **Periodic Evaluation**: Includes an evaluation loop (`eval_every`) to track performance on a separate set of episodes.
- **Early Stopping & Model Checkpointing**: Saves the best-performing model and can stop training once the environment is solved.
- **WandB Integration**: Logs metrics, losses, gradients, and episode statistics to Weights & Biases.
- **Gradient Clipping**: Prevents exploding gradients via `max_grad_norm`.
- **Learning Rate Scheduling**: Supports schedulers to adjust the learning rate during training.
- **Flexible Architecture**: Easily configurable hidden layer dimensions.
 
 ### Instantiating Models
 
There is a separation between the agent's configuration and the training configuration.
Train specific parameters like `episodes`, `max_steps`, `learning_rate` are passed to the `train_online` method, while the agent's hyperparameters (like `hidden_dims`, `baseline_type`...) are passed to the agent's constructor.
 
 ```python 
import gym
from rlforge.agents import REINFORCEAgent
from rlforge.experiments import Experiment
from rlforge.functional import SchedulerType

experiment = Experiment(
    name="CartPole Experiment",
    env = gym.make("CartPole-v1"),
    env_render = gym.make("CartPole-v1", render_mode="human")
)

agent1 = REINFORCEAgent(name="CartPole-no-baseline",
                         gamma=0.99,
                         experiment=experiment, # <-- Setting the experiment automatically creates the policiy network and optimizer based on the dimensions of the environment.
                         entropy_weight=0.01,
                         hidden_dims=[128, 64],
                         learning_rate_scheduler=SchedulerType.COSINE_ANNEALING, 
                         )

agent1.train_online(episodes=2000,
                     max_steps=500,
                     eval_every=10,     # <-- *N* This parameter controls how often the agent is evaluated.
                     eval_episodes=3,   # <-- *M* Number of evaluation episodes per evaluation step.
                     render_every=-1,   # <-- -1 means no rendering during training
                     early_stop=True,
                     learning_rate=1e-3
                     )
 ``` 

 ## Model Notes
 
 * Added **evaluation handler**: every *N* episodes the agent plays *M* evaluation episodes with gradients disabled; metrics are logged and returned.  
 * Introduction of a `compare_agents` utility for side-by-side performance plots. 
 * The backward step consists of:
   ```python
   loss = -aggregation(log_probs * returns)
   ```
   where `aggregation` is either `torch.mean` or `torch.sum`, depending on the agent's configuration.
   ```python
   loss += self.entropy_weight * entropy
   ```
   The "mean" operation has been chosen for successive tests, to ensure the loss is independent of the number of timesteps in the episode, and to limit the loss scale overall in environments with similar reward scales accross the steps of an episode.
 ---
 
 ## Exercise 2: Baselines!
 
Introducing a technique to reduce the variance of the policy gradient estimates: the **baseline**.
In the REINFORCE algorithm, the update rule is :
```math
\theta \leftarrow \theta + \alpha \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t | s_t) (R_t - b(s_t))
```
Where \(b(s_t)\) is the baseline function. The baseline can be a constant (in the precedent exercise, it was 0), a learned value function, or a running average of returns.
I implemented four baseline types using the `BaselineType` enum, defined in `rlforge.functional`:
* **No Baseline**: 0 baseline, the original REINFORCE algorithm.
* **Episodic Standardization**: Subtracts the mean and divides by the standard deviation of returns within the episode.  
* **Running Standardization**: Maintains a running mean and standard deviation of returns across episodes.
* **Value Function Baseline**: Uses a separate neural network to estimate the value function \(v(s)\) and uses it as the baseline.
 
 ```python
from rlforge.functional import BaselineType
agent2 = REINFORCEAgent(name="CartPole-std-baseline",
                         learning_rate=1e-3,
                         gamma=0.99,
                         experiment=experiment,
                         baseline_type=BaselineType.EPISODIC_STANDARDIZATION # <-- Episodic Standardization baseline
                         )
 agent2.train_online(episodes=2000, max_steps=500, eval_every=10, early_stop=True)
 ```
 or
 ```python
from rlforge.functional import BaselineType
baseline_type = BaselineType.VALUE_FUNCTION
baseline_args = {"learning_rate": 1e-3,
                 "hidden_dims": [128, 64]}
agent2 = REINFORCEAgent(name="CartPole-running-std-baseline",
                         learning_rate=1e-3,
                         gamma=0.99,
                         experiment=experiment,
                         baseline_type=(baseline_type, baseline_args) # <-- Here we pass the baseline type and its specific arguments
                        )
```
### Results on CartPole-v1
I tested `BaselineType.NONE`,`BaselineType.EPISODIC_STANDARDIZATION` and `BaselineType.VALUE_FUNCTION` on CartPole-v1.
All agents were run with identical hyper-parameters for up to 2,000 episodes. Performance was checked every 10 episodes using 10 test episodes. An early-stopping hook halted training once the latest evaluation showed a mean reward of 200, provided the score could be confirmed on a larger test set (100 episodes by default).
The training parameters were:
```python
learning_rate=0.005, 
gamma=0.99, 
experiment=cartpole_experiment, 
entropy_weight=0
# plus, for the value function baseline:
baseline_type=(BaselineType.VALUE_FUNCTION,{"learning_rate":0.001}),
```
This graph shows the learning curves of the three agents by plotting the evaluation mean reward over time (every 10 episodes, averaged over 10 test episodes):
<div align="center">
  <img src="/images/cartpole_comp.png" width="600" alt="CartPole-v1 Learning Curves">
</div>

These results show that the **value-function baseline** can converge faster and with lower variance than both the plain REINFORCE and the episodic standardization baseline, plus the episodic standardization baseline is more stable than the plain REINFORCE agent. 
Its important to note that, empirically, the episodic standardization baseline can sometimes converge faster than the value function baseline (though, they both always get faster than the plain REINFORCE agent): a more thorough analysis should compare the two baselines on a wider range of hyper-parameters and seeds, to perform a more robust statistical analysis of the results.


 
## Exercise 3.1: Going to the Moon!

The LunarLander-v3 environment presents a significantly greater challenge than the humble CartPole-v1: here, our intrepid agent must orchestrate the delicate landing of a spacecraft on a specific pad, all while managing its fuel reserves.
Despite the exercise being cheerfully labelled “easy” by the instructor, I discovered that training REINFORCE into stable, efficient learning was anything but trivial. Instead, my agents appeared to have mastered the art of suboptimality:

  - **The High-Flyer:** Prefers to hover as high as possible, neatly avoiding both crash and landing alike.
  - **The Touch-and-Go:** Lands perfectly on the pad, only to immediately take off again, thus never actually landing.
  - **The Maverick:** Lands gracefully, but always just outside the designated pad. (The environment, in its benevolence, doesn’t penalize this, but it’s not exactly the stuff of NASA dreams.)

To address these creative misadventures, I methodically rolled out various improvements to my REINFORCE agent:
- **Higher Gamma**: Broadening the agent's temporal horizon to plan further ahead (shutting down the engines early, for instance).
- **Entropy Regularization:** Encouraging broader exploration, in the hopes of avoiding premature commitment to bad ideas.
- **Learning Rate Scheduling:** Allowing the learning rate to adapt gracefully over time.
- **Gradient Clipping:** To prevent exploding gradients.
- **Deeper Policy Network:** Hoping that additional layers might inspire some deeper thinking.

While these upgrades led to a somewhat more civilized training process (after testing A LOT of configurations), my agent still stubbornly refused to master the optimal landing strategy.

Wishing to avoid over-engineering, I refrained from adding further tweaks, such as:
- **Reward Shaping:** Tinkering with the reward function to provide richer feedback. (For instance, rewarding low altitude and engine shutdown could fix the “hovering” obsession.)
- **Curriculum Learning:** Starting with a gentler version of the environment (e.g., placing the agent closer to the ground) to teach basic landing skills before tackling full navigation.

These approaches, while tempting, felt more like engineering band-aids than principled solutions, and I was determined to stay true to the original spirit of REINFORCE.

In any case, the next exercise (Q-learning) devoured LunarLander-v3 with the enthusiasm of a wolf at dinner, so I decided my time was better spent elsewhere.

### Results on LunarLander-v3
I present the results of my REINFORCE agents on LunarLander-v3. The training parameters for this results, were:
```python
max_steps = 300
lr = 0.001
baseline_lr = 0.0001
gamma = 0.99
aggregation = "mean"
hidden_dims = [128, 64]
activation = "relu"  
episodes = 2000
```
I tested `BaselineType.NONE`,`BaselineType.RUNNING_STANDARDIZATION` and `BaselineType.VALUE_FUNCTION` on LunarLander-v3.
This graph shows the learning curves of the three agents by plotting the evaluation mean reward over time (every 100 episodes, averaged over 5 test episodes):
<div align="center">
  <img src="/images/lunar_comp.png" width="600" alt="LunarLander-v3 Learning Curves">
</div>

<div align="center">
    <img src="/images/correct_landings_reinforce.png" width="600" alt="LunarLander-v3 REINFORCE Some Landing Episodes">
    <p><em>Some rare successful landing episodes achieved by the REINFORCE agent. While these episodes demonstrate that landing is possible, they were not consistently reproducible.</em></p>
</div>

However, no agent was able to solve the environment, as the mean evaluation reward never exceeded 200, which was the threshold needed.
Upon reflection, it seems that the REINFORCE algorithm is simply not well-suited to environments plagued by sparse rewards and high variance, such as LunarLander-v3. Success seems to come down to a fortuitous run of good episodes early in training: statistically possible, but in my case, not happening. (I have seen repositories that manage to solve LunarLander-v3 with REINFORCE, so it is technically possible. Unfortunately, not within my time frame for this lab.)
The only way any of the REINFORCE agents managed to land the spacecraft was adopting an entropy schedule that peaked at around mid-training, then decayed to zero: this approach allowed the agent to converge to its suboptimal policy, then mix it up a bit to find a better one. However, this was not enough to solve the environment, and the agent still struggled to land the spacecraft with the consistency needed to reach the 200 mean reward threshold in evaluation.

## Exercise 3.2: Going to the Q!
### DQN Agent
The DQN agent is a more advanced DRL algorithm that uses a neural network to approximate the Q-value function.
It is based on the use of a replay buffer to store past experiences and a target network to stabilize training.
I implemented the DQN by leveraging the existing `StepData` class to store experiences into a `ReplayBuffer`.
The results were much better than the REINFORCE agent, with the DQN agent learning to land the spacecraft on the designated landing pad consistently and in less than 2000 episodes.
The initialization of the DQN agent is similar to the REINFORCE agent, but with additional parameters for the Q-network and replay buffer:
```python
from rlforge.agents.q import QLearningAgent
q_cartpole_agent = QLearningAgent(
    name="CartPole DQN Agent",
    learning_rate=0.01,
    gamma=0.9,
    epsilon_start=.9,       # <-- epsilon at the beginning of training
    epsilon_end=0.2,        # <-- epsilon at the end of training
    epsilon_decay=0.9999,   # <-- epsilon decay rate
    batch_size=256,         # <-- every update, the agent samples batch_size steps from the replay buffer
    buffer_capacity=500,    # <-- size of the replay buffer (in steps), it will remember the last 500 steps of the agent's experience
    target_update=10,       # <-- update the target network every 10 steps
)
```

The epsiilon is scheduled to decay from `epsilon_start` to `epsilon_end` by the class `EpsilonScheduler`, which is used to control the exploration-exploitation trade-off during training. The scheduler is a simple lienar decay that stops at the `epsilon_end` value:
```python 
class EpsilonScheduler:
    def __init__(self, start=1.0, end=0.05, decay=0.995):
        self.epsilon = start
        self.start = start
        self.last_epsilon = start
        self.end = end
        self.decay = decay

    def step(self):
        self.epsilon = max(self.end, self.epsilon * self.decay)
        self.last_epsilon = self.epsilon

    def get_epsilon(self):
        return self.epsilon
    
    def eval(self):
        self.last_epsilon = self.epsilon
        self.epsilon = 0
    
    def train(self):
        self.epsilon = self.last_epsilon
```
It's important to note that the epsilon value is set to 0 during evaluation, meaning the agent will always choose the action with the highest Q-value when testing (if `agent.eval()` is called). 


### Results on LunarLander-v3: CartPole-v1
Here are the results of the DQN agent on CartPole-v1, which was able to solve the environment really quickly.
The training parameters were:
```python
epsilon_start=.99,
epsilon_end=0.2,
epsilon_decay=0.9999,
batch_size=256,
buffer_capacity=500,
target_update=10,
episodes=2000,
max_steps=500,
learning_rate=0.01,
scheduler_type=SchedulerType.COSINE_ANNEALING,

```
This graph shows the learning curves of the DQN agent vs the best REINFORCE agent by plotting the evaluation mean reward over time (every 10 episodes, averaged over 10 test episodes):
<div align="center">
  <img src="/images/cartpole_comp_dqn.png" width="600" alt="CartPole-v1 DQN Learning Curves">
</div>
<div align="center">
  <video width="640" height="360" controls>
    <source src="videos/cartpole_dqn_video.mp4/rl-video-episode-0.mp4" type="video/mp4">
  </video>
</div>

### Results on LunarLander-v3: Q-Learning
Here are the results of the DQN agent on LunarLander-v3, again, the DQN agent was able to solve the environment without struggling too much.
The training parameters were:
```python
epsilon_start=1.0,
epsilon_end=0.0001,
epsilon_decay=0.995,
batch_size=512,
buffer_capacity=10000,
target_update=20,
episodes=2000,
max_steps=500,
learning_rate=0.01,
scheduler_type=SchedulerType.COSINE_ANNEALING,
```
This graph shows the learning curves of the DQN agent vs the best REINFORCE agent by plotting the evaluation mean reward over time (every 50 episodes, averaged over 10 test episodes):
<div align="center">
  <img src="/images/lunar_comp_dqn.png" width="600" alt="LunarLander-v3 DQN Learning Curves">
</div>
<div align="center">
  <video width="640" height="360" controls>
    <source src="videos/lunar_lander_q/rl-video-episode-0.mp4" type="video/mp4">
  </video>
</div>


---
 
 ## Lessons Learned
 
* **RL is FRUSTRATING** – watching your agent learn is an ultimate exercise in patience. It’s like trying to charm a girl: there are moments when everything seems to click, when you catch a glimmer of hope, and you think the stars are finally aligning. But then, crack, the fragile connection shatters. You are thrown back to square one.
* **RL is FUN!** – it's you against the environment, and every small victory feels like a triumph. This lab made me an RL enthusiast: I can't wait to explore more complex environments (i want to try it on real video games!) and algorithms (genetic algorithms have been on my mind for a while now).
* **REINFORCE is not enough** – the REINFORCE algorithm is a great starting point, but it has its limitations. It struggles with high variance and sparse rewards, as seen in the LunarLander-v3 environment. DQN instead, needs just a few episodes with good rewards to learn a good policy, being able to learn from past experiences.
