import gymnasium as gym
from gymnasium.utils.play import play

env = gym.make("CartPole-v1", render_mode="rgb_array")

# CartPole action: 0 = trái, 1 = phải
keys_to_action = {
    "a": 0,   # phím A = đẩy trái
    "d": 1,   # phím D = đẩy phải
}

play(env, keys_to_action=keys_to_action, noop=0, fps=15, zoom=3)
