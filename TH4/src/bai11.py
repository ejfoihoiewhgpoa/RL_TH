import gymnasium as gym

env = gym.make("Blackjack-v1")

ACTION_NAMES = {0: "STICK", 1: "HIT"}

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

# Bài 11
def stick_on_20_policy(state):
    player_sum, dealer_card, usable_ace = state

    if player_sum >= 20:
        return 0        # STICK
    return 1            # HIT

# Kiểm tra ý nghĩa action
print(env.action_space)                      
for a in range(env.action_space.n):
    print(a, ACTION_NAMES[a])

# Kiểm tra policy trên vài state mẫu
test_states = [(12, 10, False), (19, 5, False), (20, 10, False), (21, 1, True)]
for s in test_states:
    a = stick_on_20_policy(s)
    print(f"state={s} -> action={a} ({ACTION_NAMES[a]})")

# Chạy thử một episode với policy này
episode = generate_episode(env, stick_on_20_policy, seed=0)
for s, a, r in episode:
    print(f"state={s}, action={a} ({ACTION_NAMES[a]}), reward={r}")

env.close()