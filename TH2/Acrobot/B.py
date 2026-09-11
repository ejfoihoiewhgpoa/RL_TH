import gymnasium as gym
import numpy as np

env = gym.make("Acrobot-v1")
u = env.unwrapped

u.LINK_LENGTH_1 = 1.0
u.LINK_LENGTH_2 = 1.5   # đổi độ dài link
t = 2.0
u.AVAIL_TORQUE = [-t, 0.0, t]  # đổi torque

rewards = []
for i in range(50):
    obs, info = env.reset(seed=i)
    total = 0
    for step in range(500):
        action = env.action_space.sample()
        obs, r, terminated, truncated, info = env.step(action)
        total += r
        if terminated or truncated:
            break
    rewards.append(total)

print("Avg reward:", np.mean(rewards))
env.close()