def compute_returns(rewards, gamma=1.0):
    returns = [0.0] * len(rewards)
    G = 0.0
    for t in reversed(range(len(rewards))):    
        G = rewards[t] + gamma * G            
        returns[t] = G
    return returns

rewards = [0, 0, 1]

for gamma in [1.0, 0.9, 0.5]:
    print(f"gamma = {gamma}: {compute_returns(rewards, gamma)}")

rewards = [0.0, 0.0, 1.0]
