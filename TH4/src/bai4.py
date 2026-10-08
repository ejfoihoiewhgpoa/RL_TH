import gymnasium as gym

env = gym.make("Blackjack-v1")

ACTION_NAME = {
    0: "STAND",
    1: "HIT",
}

state, info = env.reset()
done = False
step = 0

while not done:
    action = env.action_space.sample()
    next_state, reward, terminated, trucated, info = env.step(action)

    print(f"step {step}")
    print("state: ", state)
    print("action: ", action, ACTION_NAME[action])
    print("reward: ", reward)
    print("terminated: ", terminated)
    print("trucated: ", trucated)

    state = next_state
    done = terminated or trucated
    step += 1

env.close()