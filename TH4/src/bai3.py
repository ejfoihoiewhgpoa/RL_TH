import gymnasium as gym 

env = gym.make("Blackjack-v1")
print(env.action_space)

ACTION_NAME = {
    0: "STAND",
    1: "HIT",
}

for i in range(env.action_space.n):
    print(i, ACTION_NAME[i])