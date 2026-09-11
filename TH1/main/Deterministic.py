# Deterministic 
import gymnasium as gym

env = gym.make("FrozenLake-v1", is_slippery=False)
actions = [env.action_space.sample() for _ in range(20)]

print("DETERMINISTIC: FrozenLake-v1 (is_slippery=False)")
for run in range(3):
    obs, info = env.reset(seed=123)
    for a in actions:
        obs, reward, terminated, truncated, info = env.step(a)
    print(f"Lần {run+1}: Observation cuối = {obs}")

env.close()