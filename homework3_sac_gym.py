import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from collections import deque
import random
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import datetime
import argparse
# Hyperparameters
learning_rate = 1e-6
gamma = 0.99
tau = 0.005
alpha = 0.15  # Entropy coefficient
buffer_size = 1000000
batch_size = 256
episodes = 1000000



class Actor(nn.Module):
    def __init__(self, state_dim, action_space):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(512, action_space.shape[0])
        self.log_std = nn.Linear(512, action_space.shape[0])  # Trainable std
        self.std_const = 1  # Constant std for exploration

        self.action_scale = torch.FloatTensor((action_space.high - action_space.low) / 2.)
        self.action_bias = torch.FloatTensor((action_space.high + action_space.low) / 2.)

    def forward(self, x):
        x = self.fc(x)
        mean = self.mean_head(x)
        log_std = self.log_std(x) 
        log_std = torch.clamp(log_std, -20, 2)
        return mean, log_std
            
    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.q = nn.Sequential(
            nn.Linear(state_dim + action_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, state, action):
        return self.q(torch.cat([state, action], dim=-1))

class SACAgent:
    def __init__(self, observation_space, action_space, lr=3e-4, gamma=0.99, tau=0.005, alpha=0.1, eta_min=1e-4):
        self.actor = Actor(observation_space.shape[0], action_space)
        self.critic_1 = Critic(observation_space.shape[0], action_space.shape[0])
        self.critic_2 = Critic(observation_space.shape[0], action_space.shape[0])
        self.target_critic_1 = Critic(observation_space.shape[0], action_space.shape[0])
        self.target_critic_2 = Critic(observation_space.shape[0], action_space.shape[0])

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=3e-4)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=3e-4)

        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.scheduler = CosineAnnealingLR(self.actor_optimizer, T_max=1000000, eta_min=eta_min)
        self.scheduler_critic_1 = CosineAnnealingLR(self.critic_1_optimizer, T_max=1000000, eta_min=eta_min)
        self.scheduler_critic_2 = CosineAnnealingLR(self.critic_2_optimizer, T_max=1000000, eta_min=eta_min)

        # Copy weights
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())

        # Use a deque as the replay buffer with a fixed maximum length.
        self.replay_buffer = deque(maxlen=buffer_size)

    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action, _, _ = self.actor.sample(state)
        return action.detach().numpy()[0]

    def store(self, transition):
        # Append the transition to the replay buffer.
        self.replay_buffer.append(transition)

    def soft_update(self, target, source):
        for t_param, s_param in zip(target.parameters(), source.parameters()):
            t_param.data.copy_(self.tau * s_param.data + (1.0 - self.tau) * t_param.data)

    def learn(self, batch_size=128):


        # Sample a batch using random.sample:
        batch = list(zip(*random.sample(list(self.replay_buffer), batch_size)))

        # Properly stack each component of the transition.
        states = torch.tensor(np.vstack(batch[0]), dtype=torch.float32)
        actions = torch.tensor(np.vstack(batch[1]), dtype=torch.float32)
        rewards = torch.tensor(np.vstack(batch[2]), dtype=torch.float32)
        next_states = torch.tensor(np.vstack(batch[3]), dtype=torch.float32)
        dones = torch.tensor(np.vstack(batch[4]), dtype=torch.float32)

        with torch.no_grad():
            next_actions, next_log_probs, _ = self.actor.sample(next_states)
            q1_next = self.target_critic_1(next_states, next_actions)
            q2_next = self.target_critic_2(next_states, next_actions)
            min_q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_probs
            target_q = rewards + self.gamma * (1 - dones) * min_q_next

        # Critic updates
        q1 = self.critic_1(states, actions)
        q2 = self.critic_2(states, actions)
        loss_critic_1 = F.mse_loss(q1, target_q)
        loss_critic_2 = F.mse_loss(q2, target_q)

        self.critic_1_optimizer.zero_grad()
        loss_critic_1.backward()
        #torch.nn.utils.clip_grad_norm_(self.critic_1.parameters(), max_norm=1.0)
        self.critic_1_optimizer.step()

        self.critic_2_optimizer.zero_grad()
        loss_critic_2.backward()
        #torch.nn.utils.clip_grad_norm_(self.critic_2.parameters(), max_norm=1.0)
        self.critic_2_optimizer.step()

        # Actor update
        #print(f"Buffer size {len(self.replay_buffer)}")
        #if len(self.replay_buffer) == buffer_size:
            #print("Buffer size is full, updating actor")
        actions_new, log_probs_new, _ = self.actor.sample(states)
        q1_new = self.critic_1(states, actions_new)
        q2_new = self.critic_2(states, actions_new)
        min_q_new = torch.min(q1_new, q2_new)
        actor_loss = ((self.alpha * log_probs_new) - min_q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()
        #self.scheduler.step()

        #self.actor.std_const = self.actor.std_const * 0.99  # Decay epsilon for exploration
            #print(f"Actor loss: {actor_loss.item()}, Critic 1 loss: {loss_critic_1.item()}, Critic 2 loss: {loss_critic_2.item()}")
        #else:
            #print(f"Critic 1 loss: {loss_critic_1.item()}, Critic 2 loss: {loss_critic_2.item()}")

        # Step learning rate schedulers for actor and critics

        #self.scheduler_critic_1.step()
        #self.scheduler_critic_2.step()

        return loss_critic_1.item(), loss_critic_2.item(), actor_loss.item()

    def update_model(self):
        # This function is not needed in SAC, as learning is done in the learn method.
        
        # Soft update target networks
        self.soft_update(self.target_critic_1, self.critic_1)
        self.soft_update(self.target_critic_2, self.critic_2)

# Main Training Loop
def train():
    writer = SummaryWriter('runs/{}_SAC'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))


    env = gym.make("Pusher-v5", max_episode_steps=1000)


    agent = SACAgent(observation_space=env.observation_space, action_space=env.action_space, lr=learning_rate, eta_min= 1e-5, alpha=alpha)

    episode_rewards = []   # To store total reward each episode
    mean_rewards = []      # To store sliding window mean reward
    window_size = 300       # Window size for mean reward computation
    episode_steps = 0

    # Set up interactive plotting
    import matplotlib.pyplot as plt
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))
    ep_line, = ax.plot([], [], label="Episode Reward")
    mean_line, = ax.plot([], [], label=f"Mean Reward (window size = {window_size})", linewidth=2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("Training Progress - SAC")
    ax.legend()
    updates = 0

    for episode in range(episodes):
        state = env.reset()[0]
        done = False
        cumulative_reward = 0.0
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action, _, _ = agent.actor.sample(state_tensor)
            action = action.detach().numpy()[0]

            next_state, reward, is_terminal, is_truncated, _ = env.step(action)
        
            cumulative_reward += reward
            done = is_terminal or is_truncated

            agent.store((state, action, reward, next_state, done))

            state = next_state
            episode_steps += 1
                
            
            #if episode_steps % 500 == 0:
            if len(agent.replay_buffer) > batch_size:
                critic_1_loss, critic_2_loss, policy_loss = agent.learn(batch_size)             
                agent.update_model()
                


                writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                writer.add_scalar('loss/actor', policy_loss, updates)

                updates += 1


        episode_rewards.append(cumulative_reward)

        # Compute sliding mean reward
        if len(episode_rewards) >= window_size:
            mean_reward = np.mean(episode_rewards[-window_size:])
        else:
            mean_reward = np.mean(episode_rewards)
        mean_rewards.append(mean_reward)

        writer.add_scalar('reward/train', cumulative_reward, episode)
        writer.add_scalar('reward/mean', mean_reward, episode)

        if episode % 10 == 0:
            # Update the dynamic plot after every episode
            ep_line.set_data(range(len(episode_rewards)), episode_rewards)
            mean_line.set_data(range(len(mean_rewards)), mean_rewards)
            ax.relim()
            ax.autoscale_view()
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.001)
            print(f"Episode {episode}: Total Reward = {cumulative_reward:.2f}, Mean Reward = {mean_reward:.2f}")

        if episode % 100 == 0:
            # Save the model every 100 episodes
            torch.save(agent.actor.state_dict(), f"actor_sac_{episode}.pth")
            print(f"Model saved as actor_sac_{episode}.pth")

        if episode % 1000 == 0:
            fig.savefig(f"training_progress_sac_{episode}.png")

    # Turn off interactive mode and display the final plot
    plt.ioff()
    plt.show()

    # Save the final model
    torch.save(agent.actor.state_dict(), "actor_sac.pth")
    print("Model saved as actor_sac.pth")

    # Save the final plot
    fig.savefig("training_progress_sac.png")

    env.close()

def test():
    env = gym.make("Pusher-v5", max_episode_steps=500, render_mode="human")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    agent = SACAgent(observation_space=env.observation_space, action_space=env.action_space, lr=learning_rate, eta_min= 5e-4)
    agent.actor.load_state_dict(torch.load("actor_sac_5800.pth"))
    agent.actor.eval()

    state = env.reset()[0]
    done = False
    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        _, _, action = agent.actor.sample(state_tensor)
        action = action.detach().numpy()[0]


        next_state, reward, is_terminal, is_truncated, _ = env.step(action)
        state = next_state
        done = is_terminal or is_truncated

    env.close()
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAC Agent for Pusher-v5")
    parser.add_argument("--train", action="store_true", help="Train the agent")
    parser.add_argument("--test", action="store_true", help="Test the agent")
    args = parser.parse_args()
    if args.train:
        train()
    elif args.test:
        test()
    else:
        print("Please specify --train or --test.")