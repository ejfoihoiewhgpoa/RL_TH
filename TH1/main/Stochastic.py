# Stochastic
import gymnasium as gym

env = gym.make("FrozenLake-v1", is_slippery=True)
actions = [env.action_space.sample() for _ in range(20)]

print("STOCHASTIC: FrozenLake-v1 (is_slippery=True)")

obs, info = env.reset(seed=123)  # chỉ seed 1 lần duy nhất, ở NGOÀI vòng lặp
for run in range(3):
    obs, info = env.reset()  # reset KHÔNG kèm seed để giữ nguyên trạng thái RNG đang chạy
    for a in actions:
        obs, reward, terminated, truncated, info = env.step(a)
    print(f"Lần {run+1}: Observation cuối = {obs}")

env.close()