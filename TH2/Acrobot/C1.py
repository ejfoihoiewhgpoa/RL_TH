import gymnasium as gym
from gymnasium.utils.play import play

env = gym.make("Acrobot-v1", render_mode="rgb_array")

# Acrobot action: 0 = torque -1, 1 = torque 0, 2 = torque +1
keys_to_action = {
    "a": 0,   # phím A = torque -1
    "d": 2,   # phím D = torque +1
    # không nhấn gì = noop (torque 0)
}

play(env, keys_to_action=keys_to_action, noop=1, fps=15)  # giảm từ 30 xuống 15