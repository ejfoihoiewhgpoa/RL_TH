import gymnasium as gym 
def main():
    env = gym.make("Blackjack-v1")
    obs_space = env.observation_space
    
    print("Action space:", env.action_space)
    print("Number of actions:", env.action_space.n)
    print("Shape:", obs_space.shape)
    
    env.close()
if __name__ == "__main__":
    main()

