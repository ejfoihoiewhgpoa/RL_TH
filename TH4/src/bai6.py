import gymnasium as gym

def generate_episode(env, policy, seed=None):
    episode = []                                   # danh sách các (state, action, reward)
    state, info = env.reset(seed=seed)
    done = False

    while not done:
        action = policy(state)                     # policy nhận state, trả về action
        next_state, reward, terminated, truncated, info = env.step(action)
        episode.append((state, action, reward))    # lưu state TRƯỚC khi hành động
        state = next_state
        done = terminated or truncated

    return episode

env = gym.make("Blackjack-v1")

def random_policy(state):
    return env.action_space.sample()

episode = generate_episode(env, random_policy, seed=0)
for step in episode:
    print(step)

# episode lấy từ Bài 5: danh sách các (state, action, reward)
episode = generate_episode(env, random_policy, seed=0)

rewards = [r for (s, a, r) in episode]

print("Episode length:", len(episode))
print("Rewards       :", rewards)
print("Total reward  :", sum(rewards))