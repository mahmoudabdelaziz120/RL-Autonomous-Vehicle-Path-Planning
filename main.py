import numpy as np
import gymnasium as gym
import random
import time
import matplotlib.pyplot as plt
from IPython import display

# ==============================================================================
# 1. REPRODUCIBILITY & ENVIRONMENT SETUP
# ==============================================================================
# Enforcing deterministic PRNG seeds to ensure hyperparameter tuning is measurable.
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# Initialize the Stochastic Environment (Slippery = True forces transition probabilities < 1.0)
env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=True)

state_space_size = env.observation_space.n
action_space_size = env.action_space.n

print(f"State Space: {state_space_size} | Action Space: {action_space_size}")

# ==============================================================================
# 2. OPTIMISTIC INITIALIZATION & HYPERPARAMETERS
# ==============================================================================
# Overcoming sparse rewards (R=1 only at goal, R=0 elsewhere) by initializing 
# Q-values to the maximum possible theoretical reward. This forces systematic 
# exploration of the entire state space as unvisited states remain mathematically 
# more attractive than visited states with R=0.
q_table = np.ones((state_space_size, action_space_size)) * 1.0
best_q_table = np.ones((state_space_size, action_space_size)) * 1.0

# Calibrated for an 8x8 grid complexity
num_episodes = 50000
max_steps_per_episode = 200
learning_rate = 0.1  # Alpha: Update step size
discount_rate = 0.995 # Gamma: High value to propagate the delayed reward from step 64 back to 0

exploration_rate = 1.0
max_exploration_rate = 1.0
min_exploration_rate = 0.01

# Dynamic Exponential Decay: Designed to hit the minimum epsilon at 90% of total episodes
exploration_decay_rate = -np.log(min_exploration_rate) / (num_episodes * 0.9)

rewards_all_episodes = []
best_average_reward = 0.0

print("Architecture: Tabular Q-Learning | Initialization: Optimistic (1.0)")

# ==============================================================================
# 3. CORE TRAINING LOOP (BELLMAN EQUATION UPDATE)
# ==============================================================================
print("Initiating Training Phase...")

for episode in range(num_episodes):
    # Pass seed only on the first episode to anchor the PRNG sequence
    state, info = env.reset(seed=SEED if episode == 0 else None)
    done = False
    rewards_current_episode = 0

    for step in range(max_steps_per_episode):
        # Epsilon-Greedy Action Selection
        if random.uniform(0, 1) > exploration_rate:
            action = np.argmax(q_table[state, :]) # Exploitation
        else:
            action = env.action_space.sample()    # Exploration

        # Execute transition
        new_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Bellman Equation Update: Q(s,a) <- Q(s,a) + alpha * [R + gamma * max Q(s',a') - Q(s,a)]
        best_next_action_value = np.max(q_table[new_state, :])
        q_table[state, action] = q_table[state, action] * (1 - learning_rate) + \
            learning_rate * (reward + discount_rate * best_next_action_value)

        state = new_state
        rewards_current_episode += reward

        if done:
            break

    # Decay Epsilon
    exploration_rate = min_exploration_rate + \
        (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate * episode)

    rewards_all_episodes.append(rewards_current_episode)

    # Checkpointing: Save the optimal policy derived from the moving average
    if (episode + 1) % 5000 == 0:
        current_avg = np.mean(rewards_all_episodes[-5000:])
        print(f"Episode {episode + 1}: Moving Avg Reward = {current_avg:.4f}, Epsilon = {exploration_rate:.3f}")

        if current_avg > best_average_reward:
            best_average_reward = current_avg
            best_q_table = np.copy(q_table)

print("Training Complete. Optimal policy checkpoint saved.")
env.close()

# ==============================================================================
# 4. EVALUATION & VISUAL RENDERING (CLOUD/COLAB COMPATIBLE)
# ==============================================================================
def render_colab_simulation(q_policy, env_size="8x8", slippery=True, episodes=1):
    """
    Executes a deterministic evaluation (Epsilon=0) using the learned policy.
    Renders via Matplotlib to bypass X11 display constraints on headless cloud servers.
    """
    print("\nInitiating headless visual simulation...")
    eval_env = gym.make(
        "FrozenLake-v1", 
        map_name=env_size, 
        is_slippery=slippery, 
        render_mode="rgb_array"
    )

    for ep in range(episodes):
        state, info = eval_env.reset()
        done = False
        step_count = 0
        
        plt.figure(figsize=(6, 6))
        img = plt.imshow(eval_env.render())
        plt.axis('off')
        
        while not done:
            # Pure exploitation of the optimal policy
            action = np.argmax(q_policy[state, :])
            
            new_state, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
            state = new_state
            step_count += 1
            
            # Frame update
            img.set_data(eval_env.render())
            display.display(plt.gcf())
            display.clear_output(wait=True)
            time.sleep(0.2)
            
        if reward == 1.0:
            print(f"Episode {ep + 1} Result: SUCCESS! Reached Goal in {step_count} steps.")
        else:
            print(f"Episode {ep + 1} Result: FAILED. Terminated at step {step_count}.")
            
    eval_env.close()

# Execute 3 evaluation episodes
render_colab_simulation(best_q_table, env_size="8x8", slippery=True, episodes=3)
