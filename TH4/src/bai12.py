import gymnasium as gym

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

def stick_on_20_policy(state):
    player_sum, dealer_card, usable_ace = state
    if player_sum >= 20:
        return 0        # STICK
    return 1            # HIT

# Bài 12
n_episodes = 100
win = loss = draw = 0

for i in range(n_episodes):
    episode = generate_episode(env, stick_on_20_policy, seed=0 if i == 0 else None)
    final_reward = episode[-1][2]          # reward ở bước cuối quyết định kết quả ván

    if final_reward > 0:
        win += 1
    elif final_reward < 0:
        loss += 1
    else:
        draw += 1

print(f"Win : {win}  ({win / n_episodes:.2%})")
print(f"Loss: {loss}  ({loss / n_episodes:.2%})")
print(f"Draw: {draw}  ({draw / n_episodes:.2%})")
print(f"Tổng: {win + loss + draw}")

env.close()