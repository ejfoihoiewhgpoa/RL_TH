import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os

env = gym.make("Blackjack-v1")

def random_policy(state):
    return env.action_space.sample()

def generate_episode(env, policy, seed=None):
    episode = []
    state, info = env.reset(seed=seed)
    done = False
    while not done:
        action = policy(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        episode.append((state, action, reward))
        state = next_state
        done = terminated or truncated
    return episode

def compute_returns(rewards, gamma=1.0):
    returns = [0.0] * len(rewards)
    G = 0.0
    for t in reversed(range(len(rewards))):
        G = rewards[t] + gamma * G
        returns[t] = G
    return returns

# Bài 10
gammas = [0.5, 0.8, 0.9, 0.99, 1.0]
n_episodes = 10000

# Tạo episode một lần, dùng lại cho mọi gamma để so sánh công bằng
all_rewards = []
for i in range(n_episodes):
    episode = generate_episode(env, random_policy, seed=i if i == 0 else None)
    all_rewards.append([r for (s, a, r) in episode])

mean_G0 = []
for gamma in gammas:
    G0_list = [compute_returns(rewards, gamma)[0] for rewards in all_rewards]
    mean_G0.append(np.mean(G0_list))
    print(f"gamma = {gamma}: mean G_0 = {mean_G0[-1]:.4f}")

# Vẽ biểu đồ
plt.figure(figsize=(7, 4))
plt.plot(gammas, mean_G0, marker="o")
plt.xlabel("gamma")
plt.ylabel("mean G_0")
plt.title("Mean G_0 theo gamma (Blackjack, random policy)")
plt.grid(True)

save_dir = "D:/reinforcement_learning/RL_TH/TH4/figures"
os.makedirs(save_dir, exist_ok=True) 
plt.savefig(os.path.join(save_dir, "gamma_comparison.png"), dpi=150, bbox_inches="tight")
plt.show()

env.close()