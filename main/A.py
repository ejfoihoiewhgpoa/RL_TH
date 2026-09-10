import gymnasium as gym

env = gym.make("MountainCar-v0", render_mode="human")

print(env.action_space)
print(env.observation_space)

obs, info = env.reset()

print("\nCHẠY VỚI HÀNH ĐỘNG CỐ ĐỊNH (luôn là 0: đẩy trái)")
for step in range(200):
    action = 0  # luôn luôn chọn hành động 0
    obs, reward, terminated, truncated, info = env.step(action)
    
    print(f"Bước {step}: Observation = {obs}, Reward = {reward}")
    
    if terminated or truncated:
        obs, info = env.reset()
        print("Môi trường đã reset")

env.close()