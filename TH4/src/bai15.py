import os
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

env = gym.make("Blackjack-v1")

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

def stick_on_20_policy(state):
    player_sum, dealer_card, usable_ace = state
    if player_sum >= 20:
        return 0        # STICK
    return 1            # HIT

# Bài 15
target_state = (20, 10, False)
checkpoints = [100, 500, 1000, 5000, 10000]

returns = defaultdict(list)
estimates = {}                                    # {số episode: V(target_state)}

for i in range(1, max(checkpoints) + 1):
    episode = generate_episode(env, stick_on_20_policy, seed=0 if i == 1 else None)
    rewards = [r for (s, a, r) in episode]
    G_list = compute_returns(rewards, gamma=1.0)

    for (state, action, reward), G in zip(episode, G_list):
        returns[state].append(G)

    if i in checkpoints:
        if len(returns[target_state]) > 0:
            estimates[i] = np.mean(returns[target_state])
        else:
            estimates[i] = np.nan                 # state chưa xuất hiện lần nào
        print(f"Sau {i:>5} episode: V{target_state} = {estimates[i]:.3f} "
              f"(số lần xuất hiện: {len(returns[target_state])})")

# Vẽ convergence curve
plt.figure(figsize=(7, 4))
plt.plot(list(estimates.keys()), list(estimates.values()), marker="o")
plt.xscale("log")
plt.xlabel("Số episode (log scale)")
plt.ylabel(f"Ước lượng V{target_state}")
plt.title(f"Convergence của V{target_state}")
plt.grid(True)

save_dir = "D:/reinforcement_learning/RL_TH/TH4/figures"
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, "convergence_curve.png"), dpi=150, bbox_inches="tight")
plt.show()

env.close()