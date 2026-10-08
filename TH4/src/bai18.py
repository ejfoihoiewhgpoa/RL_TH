import gymnasium as gym
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

def find_first_visits(episode):
    first_visit = {}
    for t, (state, action, reward) in enumerate(episode):
        if state not in first_visit:
            first_visit[state] = t
    return first_visit

def first_visit_mc_prediction(env, policy, n_episodes, gamma=1.0):
    returns_sum = defaultdict(float)
    returns_count = defaultdict(int)
    V = defaultdict(float)

    for _ in range(n_episodes):
        episode = generate_episode(env, policy)
        rewards = [r for (s, a, r) in episode]
        G_list = compute_returns(rewards, gamma)
        first_visit = find_first_visits(episode)

        for state, t in first_visit.items():
            returns_sum[state] += G_list[t]
            returns_count[state] += 1
            V[state] = returns_sum[state] / returns_count[state]

    return dict(V), dict(returns_count)

def stick_on_20_policy(state):
    player_sum, dealer_card, usable_ace = state
    if player_sum >= 20:
        return 0        # STICK
    return 1            # HIT

# Bài 18
for n_episodes in [100, 1000, 10000, 50000]:
    V, returns_count = first_visit_mc_prediction(env, stick_on_20_policy, n_episodes)
    print(f"{n_episodes:>6} episode -> số state đã ước lượng: {len(V)}")

env.close()