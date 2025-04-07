import torch
import numpy as np
import torch.nn.functional as F
from model_sac import Actor, Critic
import random  # <-- add this import

class SACAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2):
        self.actor = Actor(state_dim, action_dim)
        self.critic_1 = Critic(state_dim, action_dim)
        self.critic_2 = Critic(state_dim, action_dim)
        self.target_critic_1 = Critic(state_dim, action_dim)
        self.target_critic_2 = Critic(state_dim, action_dim)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=lr)

        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        # Copy weights
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())

        self.replay_buffer = []

    def act(self, state):
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action, _ = self.actor.sample(state)
        return action.squeeze(0).numpy()

    def store(self, transition):
        self.replay_buffer.append(transition)

    def soft_update(self, target, source):
        for t_param, s_param in zip(target.parameters(), source.parameters()):
            t_param.data.copy_(self.tau * s_param.data + (1.0 - self.tau) * t_param.data)

    def update(self, batch_size=64):
        if len(self.replay_buffer) < batch_size:
            return

        # Sample using random.sample instead of np.random.choice
        batch = list(zip(*random.sample(self.replay_buffer, batch_size)))
        # Use np.vstack to properly stack each component of the transition
        states = torch.tensor(np.vstack(batch[0]), dtype=torch.float32)
        actions = torch.tensor(np.vstack(batch[1]), dtype=torch.float32)
        rewards = torch.tensor(np.vstack(batch[2]), dtype=torch.float32)
        next_states = torch.tensor(np.vstack(batch[3]), dtype=torch.float32)
        dones = torch.tensor(np.vstack(batch[4]), dtype=torch.float32)

        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
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
        self.critic_1_optimizer.step()

        self.critic_2_optimizer.zero_grad()
        loss_critic_2.backward()
        self.critic_2_optimizer.step()

        # Actor update
        actions_new, log_probs_new = self.actor.sample(states)
        q1_new = self.critic_1(states, actions_new)
        q2_new = self.critic_2(states, actions_new)
        min_q_new = torch.min(q1_new, q2_new)

        actor_loss = (self.alpha * log_probs_new - min_q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update target networks
        self.soft_update(self.target_critic_1, self.critic_1)
        self.soft_update(self.target_critic_2, self.critic_2)
