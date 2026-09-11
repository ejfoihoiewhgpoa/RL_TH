import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def run_episode_random(env, seed):
    obs, info = env.reset(seed=seed)
    total_reward = 0
    for step in range(500):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            return total_reward, step + 1
    return total_reward, 500

env = gym.make("CartPole-v1")
u = env.unwrapped

# Thay đổi tham số
u.length = 1.5              # pole dài hơn (mặc định 0.5)
u.polemass_length = u.masspole * u.length
u.gravity = 9.8              # trọng lực (mặc định 9.8)
u.force_mag = 10.0           # lực đẩy (mặc định 10.0)

rewards = [run_episode_random(env, seed=i)[0] for i in range(50)]
print(f"Avg reward: {np.mean(rewards):.2f}")
env.close()

# Policy chọn action ngẫu nhiên
def random_policy_cartpole(obs, action_space):
    return action_space.sample()


# Chạy nhiều episode
def run_many_episodes(env, policy, n_episodes=50):
    rewards = []
    steps_list = []

    for episode in range(n_episodes):
        obs, info = env.reset(seed=episode)
        total_reward = 0

        for step in range(500):
            action = policy(obs, env.action_space)

            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += reward

            if terminated or truncated:
                break

        rewards.append(total_reward)
        steps_list.append(step + 1)

    # Tính các kết quả
    avg_reward = np.mean(rewards)
    avg_steps = np.mean(steps_list)

    # CartPole đạt tối đa 500 bước
    success_rate = np.mean(np.array(rewards) >= 500)

    return {
        'avg_reward': avg_reward,
        'avg_steps': avg_steps,
        'success_rate': success_rate
    }


# Các cấu hình CartPole
cartpole_configs = {
    'default (length=0.5, gravity=9.8, force=10)':
        dict(length=0.5, gravity=9.8, force_mag=10.0),

    'longer pole (length=1.5)':
        dict(length=1.5, gravity=9.8, force_mag=10.0),

    'shorter pole (length=0.2)':
        dict(length=0.2, gravity=9.8, force_mag=10.0),

    'high gravity (gravity=20)':
        dict(length=0.5, gravity=20.0, force_mag=10.0),

    'low gravity (gravity=3)':
        dict(length=0.5, gravity=3.0, force_mag=10.0),

    'weak push (force=3)':
        dict(length=0.5, gravity=9.8, force_mag=3.0),

    'strong push (force=30)':
        dict(length=0.5, gravity=9.8, force_mag=30.0),
}


# Chạy thử từng cấu hình
cartpole_results = {}

for name, params in cartpole_configs.items():

    env = gym.make("CartPole-v1")

    u = env.unwrapped

    # Thay đổi tham số
    u.length = params['length']
    u.polemass_length = u.masspole * u.length
    u.gravity = params['gravity']
    u.force_mag = params['force_mag']

    # Chạy 50 episode
    res = run_many_episodes(
        env,
        random_policy_cartpole,
        n_episodes=50
    )

    cartpole_results[name] = res

    env.close()


# Đưa kết quả vào DataFrame
df_cartpole_params = pd.DataFrame({
    k: {
        'avg_reward': v['avg_reward'],
        'avg_steps': v['avg_steps'],
        'success_rate': v['success_rate']
    }
    for k, v in cartpole_results.items()
}).T

print(df_cartpole_params)

plt.figure(figsize=(12, 6))

plt.bar(
    df_cartpole_params.index,
    df_cartpole_params["avg_reward"]
)

plt.xlabel("CartPole configuration")
plt.ylabel("Average Reward")
plt.title("Comparison of CartPole configurations")

plt.xticks(rotation=45, ha="right")
plt.tight_layout()

plt.show()

labels = [
    "Default",
    "Long pole",
    "Short pole",
    "High gravity",
    "Low gravity",
    "Weak force",
    "Strong force"
]

plt.figure(figsize=(10, 6))

plt.bar(
    labels,
    df_cartpole_params["avg_reward"]
)

plt.xlabel("Configuration")
plt.ylabel("Average Reward")
plt.title("CartPole Performance under Different Parameters")

plt.xticks(rotation=30)
plt.tight_layout()

plt.savefig("D:/reinforcement_learning/RL_TH/TH2/figures/cartpole_comparison.png", dpi=300)

plt.show()