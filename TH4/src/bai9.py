import gymnasium as gym

env = gym.make("Blackjack-v1")

def random_policy(state):
    return env.action_space.sample()

# Bài 5
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

# Bài 7
def compute_returns(rewards, gamma=1.0):
    returns = [0.0] * len(rewards)
    G = 0.0
    for t in reversed(range(len(rewards))):
        G = rewards[t] + gamma * G
        returns[t] = G
    return returns

# Bài 9
episode = generate_episode(env, random_policy, seed=0)
rewards = [r for (s, a, r) in episode]
returns = compute_returns(rewards, gamma=1.0)

episode_with_returns = [(s, a, r, G) for (s, a, r), G in zip(episode, returns)]

for t, step in enumerate(episode_with_returns):
    print(t, step)

env.close()