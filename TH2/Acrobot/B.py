import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

env = gym.make("Acrobot-v1")
u = env.unwrapped

u.LINK_LENGTH_1 = 1.0
u.LINK_LENGTH_2 = 1.5   # đổi độ dài link
t = 2.0
u.AVAIL_TORQUE = [-t, 0.0, t]  # đổi torque

# =========================
# Random policy
# =========================
def random_policy_acrobot(obs, action_space):
    return action_space.sample()


# =========================
# Chạy nhiều episode
# =========================
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

    return {
        "avg_reward": np.mean(rewards),
        "avg_steps": np.mean(steps_list),
        "min_reward": np.min(rewards),
        "max_reward": np.max(rewards)
    }


# =========================
# Các cấu hình Acrobot
# =========================
acrobot_configs = {

    "Default":
        dict(
            link1=1.0,
            link2=1.0,
            torque=1.0
        ),

    "Long link 2":
        dict(
            link1=1.0,
            link2=1.5,
            torque=1.0
        ),

    "Short link 2":
        dict(
            link1=1.0,
            link2=0.5,
            torque=1.0
        ),

    "Weak torque":
        dict(
            link1=1.0,
            link2=1.0,
            torque=0.5
        ),

    "Strong torque":
        dict(
            link1=1.0,
            link2=1.0,
            torque=2.0
        )
}


# =========================
# Chạy thí nghiệm
# =========================
acrobot_results = {}

for name, params in acrobot_configs.items():

    env = gym.make("Acrobot-v1")

    u = env.unwrapped

    # Thay đổi độ dài hai link
    u.LINK_LENGTH_1 = params["link1"]
    u.LINK_LENGTH_2 = params["link2"]

    # Thay đổi torque
    t = params["torque"]
    u.AVAIL_TORQUE = [-t, 0.0, t]

    # Chạy 50 episode
    res = run_many_episodes(
        env,
        random_policy_acrobot,
        n_episodes=50
    )

    acrobot_results[name] = res

    env.close()


# =========================
# Tạo bảng kết quả
# =========================
df_acrobot_params = pd.DataFrame(acrobot_results).T

print(df_acrobot_params)


# =========================
# Vẽ biểu đồ Average Reward
# =========================
plt.figure(figsize=(10, 6))

plt.bar(
    df_acrobot_params.index,
    df_acrobot_params["avg_reward"]
)

plt.xlabel("Acrobot configuration")
plt.ylabel("Average Reward")
plt.title("Acrobot Performance under Different Parameters")

plt.xticks(rotation=30)
plt.tight_layout()

os.makedirs(
    "D:/reinforcement_learning/RL_TH/TH2/figures",
    exist_ok=True
)
# Lưu ảnh
plt.savefig("D:/reinforcement_learning/RL_TH/TH2/figures/acrobot_comparison.png", dpi=300)

plt.show()