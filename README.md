# 🚗 RL Autonomous Vehicle Path Planning

Autonomous Vehicle Path Planning using Tabular Q-Learning and Gymnasium.

This project demonstrates how Reinforcement Learning can be used to learn an optimal navigation policy in a stochastic environment. The agent is trained using the Q-Learning algorithm to safely navigate toward a destination while maximizing long-term rewards.

---

## 📌 Project Overview

Path planning is one of the core challenges in autonomous driving systems.

An autonomous vehicle must continuously decide:

* Which direction to move
* How to avoid hazards
* How to reach the destination efficiently
* How to act under uncertainty

This project models that problem using a Grid World environment where an intelligent agent learns through trial and error.

The implementation uses:

* Tabular Q-Learning
* Bellman Equation
* Epsilon-Greedy Exploration
* Optimistic Q-Value Initialization
* Policy Evaluation

---

## 🎯 Objective

Train an agent to discover an optimal navigation policy that maximizes the probability of reaching a destination while avoiding failure states.

---

## 🧠 Reinforcement Learning Framework

### Agent

The autonomous vehicle.

### Environment

FrozenLake-v1 (8x8 map).

### States

64 possible locations.

### Actions

4 discrete actions:

* Left
* Down
* Right
* Up

### Reward Function

| Event           | Reward |
| --------------- | ------ |
| Reach Goal      | +1     |
| Any Other State | 0      |

---

## 🌍 Environment Configuration

```python
gym.make(
    "FrozenLake-v1",
    map_name="8x8",
    is_slippery=True
)
```

The environment is stochastic.

When the agent selects an action, the intended movement may not always occur due to slippery dynamics.

This simulates uncertainty commonly encountered in real-world navigation systems.

---

## ⚙️ Hyperparameters

| Parameter           | Value |
| ------------------- | ----- |
| Episodes            | 50000 |
| Max Steps           | 200   |
| Learning Rate (α)   | 0.1   |
| Discount Factor (γ) | 0.995 |
| Initial Epsilon     | 1.0   |
| Minimum Epsilon     | 0.01  |
| Q Initialization    | 1.0   |

---

## 📈 Bellman Update Equation

The Q-table is updated using the Bellman Equation:

Q(s,a) ← Q(s,a) + α[R + γ max(Q(s',a')) − Q(s,a)]

Where:

* s = current state
* a = selected action
* R = immediate reward
* γ = discount factor
* α = learning rate

This update enables the agent to estimate long-term action values and gradually improve its policy.

---

## 🔍 Exploration vs Exploitation

The project uses an Epsilon-Greedy strategy.

### Exploration

Random actions are selected to discover new states.

### Exploitation

The agent selects the action with the highest learned Q-value.

The exploration rate decays exponentially during training.

---

## 🚀 Optimistic Initialization

Instead of initializing the Q-table with zeros:

```python
q_table = np.zeros((64,4))
```

The project uses:

```python
q_table = np.ones((64,4))
```

This optimistic initialization encourages exploration and improves learning in sparse-reward environments.

---

## 🏗 Project Structure

```text
RL-Autonomous-Vehicle-Path-Planning
│
├── main.py
├── requirements.txt
├── README.md
├── LICENSE
│
├── outputs
│   ├── q_table.npy
│   └── training_results.csv
│
├── screenshots
│   ├── training.png
│   └── simulation.png
│
└── docs
```

---

## ▶️ Installation

Clone the repository:

```bash
git clone https://github.com/mahmoudabdelaziz120/RL-Autonomous-Vehicle-Path-Planning.git
```

Move into the project directory:

```bash
cd RL-Autonomous-Vehicle-Path-Planning
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run Training

```bash
python main.py
```

---

## 📊 Expected Results

Typical performance after training:

| Metric         | Value |
| -------------- | ----- |
| Episodes       | 50000 |
| States         | 64    |
| Actions        | 4     |
| Average Reward | 0.70+ |
| Success Rate   | High  |

Results may vary because the environment contains stochastic transitions.

---

## 🎮 Evaluation Phase

After training, the agent is evaluated using a fully greedy policy:

```python
np.argmax(Q[state])
```

No random exploration is allowed during evaluation.

The learned policy is visualized using Gymnasium rendering.

---

## 🚘 Autonomous Vehicle Interpretation

FrozenLake can be interpreted as a simplified autonomous navigation problem.

| FrozenLake  | Autonomous Vehicle  |
| ----------- | ------------------- |
| Start State | Vehicle Position    |
| Frozen Tile | Safe Road           |
| Hole        | Hazard / Crash Zone |
| Goal        | Destination         |
| Policy      | Driving Strategy    |

---

## 🔬 Future Improvements

Planned extensions:

* Custom road-network environment
* Dynamic obstacles
* Traffic simulation
* Vehicle kinematic constraints
* Deep Q-Network (DQN)
* Double DQN
* Prioritized Experience Replay
* CARLA Simulator Integration
* Autonomous Driving Benchmarking

---

## 👨‍💻 Author

Mahmoud Abdelaziz

Mechatronics Engineer

Artificial Intelligence, Robotics, and Autonomous Systems Enthusiast

---

## 📄 License

This project is licensed under the MIT License.
