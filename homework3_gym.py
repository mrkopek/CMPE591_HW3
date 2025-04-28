import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
# Hyperparameters
learning_rate = 1e-4
gamma = 0.99
episodes = 1000000

# Simple Policy Network
class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(512, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))  # Trainable std
        self.std_const = 5e-2  # Constant std for exploration

    def forward(self, x):
        x = self.fc(x)
        mean = self.mean_head(x)
        std = self.log_std.exp() + self.std_const  # Ensure std is positive
        return mean, std

def compute_returns(rewards, gamma):
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return returns

# Main training loop
def train():
    env = gym.make("Pusher-v5", max_episode_steps=1000)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    policy = PolicyNetwork(obs_dim, act_dim)
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=20000, eta_min=1e-5)

    episode_rewards = []   # To store total reward each episode
    mean_rewards = []      # To store sliding window mean reward
    window_size = 300       # Window size for mean reward computation

    # Set up interactive plotting
    import matplotlib.pyplot as plt
    plt.ion()
    fig, ax = plt.subplots(figsize=(10,6))
    ep_line, = ax.plot([], [], label="Episode Reward")
    mean_line, = ax.plot([], [], label=f"Mean Reward (window size = {window_size})", linewidth=2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("Training Progress")
    ax.legend()

    for episode in range(episodes):
        state = env.reset() 
        state = state[0]  # Unwrap the tuple
        log_probs = []
        rewards = []

        done = False
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32)
            mean, std = policy(state_tensor)

            dist = Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum()

            action_clipped = torch.clamp(action,
                                         float(env.action_space.low[0]),
                                         float(env.action_space.high[0]))
            next_state, reward, terminated, truncated, _ = env.step(action_clipped.detach().numpy())
            done = terminated or truncated

            log_probs.append(log_prob)
            rewards.append(reward)
            state = next_state

        total_reward = sum(rewards)
        episode_rewards.append(total_reward)

        # Compute sliding mean reward
        if len(episode_rewards) >= window_size:
            mean_reward = np.mean(episode_rewards[-window_size:])
        else:
            mean_reward = np.mean(episode_rewards)
        mean_rewards.append(mean_reward)
        
        # Compute and normalize returns
        returns = compute_returns(rewards, gamma)
        returns = torch.tensor(returns, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        loss = []
        for log_prob, R in zip(log_probs, returns):
            loss.append(-log_prob * R)
        loss = torch.stack(loss).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        if policy.std_const > 1e-5:
            policy.std_const *= 0.99995  # Decay std constant for exploration
        if episode % 10 == 0:
            print(f"Episode {episode}: Total Reward = {total_reward:.2f}, Mean Reward = {mean_reward:.2f}, Policy Std = {policy.std_const:.4f}")

            # Update the dynamic plot after every episode
            ep_line.set_data(range(len(episode_rewards)), episode_rewards)
            mean_line.set_data(range(len(mean_rewards)), mean_rewards)
            ax.relim()
            ax.autoscale_view()
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.001)
        if episode % 100 == 0:
            # Save the model every 100 episodes
            torch.save(policy.state_dict(), f"policy_vpg_{episode}.pth")
            print(f"Model saved as policy_{episode}.pth")
        if episode % 1000 == 0:
                fig.savefig(f"training_progress_gym_vpg_{episode}.png")


    # Turn off interactive mode and display the final plot
    plt.ioff()
    plt.show()

    # Save the model
    torch.save(policy.state_dict(), "policy.pth")
    print("Model saved as policy.pth")

    #Save figure
    fig.savefig("training_progress.png")

    env.close()

def test():
    env = gym.make("Pusher-v5", render_mode="human", max_episode_steps=1000)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    policy = PolicyNetwork(obs_dim, act_dim)
    policy.load_state_dict(torch.load("policy_vpg_290000.pth"))
    policy.std_const = 0
    policy.eval()
    
    for episode in range(10):
        state = env.reset()
        state = state[0]  # Unwrap the tuple
        done = False
        total_reward = 0

        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32)
            mean, std = policy(state_tensor)
            
            dist = Normal(mean, 1e-9)
            action = dist.sample()
            action_clipped = torch.clamp(action, float(env.action_space.low[0]), float(env.action_space.high[0]))
            
            next_state, reward, terminated, truncated, _ = env.step(action_clipped.detach().numpy())
            done = terminated or truncated

            total_reward += reward
            state = next_state

        print(f"Total Reward in Test: {total_reward:.2f}")
    env.close()

if __name__ == "__main__":
    #train()
    test()