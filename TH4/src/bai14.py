import gymnasium as gym
import numpy as np
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

# Bài 13: thu thập return theo state
returns = defaultdict(list)
n_episodes = 5000

for i in range(n_episodes):
    episode = generate_episode(env, stick_on_20_policy)
    rewards = [r for (s, a, r) in episode]
    G_list = compute_returns(rewards, gamma=1.0)
    for (state, action, reward), G in zip(episode, G_list):
        returns[state].append(G)

# Bài 14: ước lượng V(s) = trung bình các return của state đó
V = {}
for state in returns:
    V[state] = np.mean(returns[state])

print("Số state đã xuất hiện:", len(V))

# In value của 10 state xuất hiện nhiều nhất
top_states = sorted(returns, key=lambda s: len(returns[s]), reverse=True)[:10]
for state in top_states:
    print(f"V{state} = {V[state]:.3f}   (số lần xuất hiện: {len(returns[state])})")

env.close()