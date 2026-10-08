import gymnasium as gym

env = gym.make("Blackjack-v1")

for ep in range(10):
    obs, info = env.reset()
    done = False
    while not done:
        print(ep, obs)
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        done = terminated or truncated
    print(ep, obs, "(kết thúc)")