import torch
from torch import optim
import numpy as np
from model import VPG
import torch.nn.functional as F

gamma = 0.99


class Agent():
    def __init__(self, model_path=None, learning_rate=1e-5):
        self.model_path = model_path
        self.model = VPG()
        if model_path is not None:
            self.model.load_state_dict(torch.load(model_path))
            self.model.eval()
        else:
            self.model.train()
        self.rewards = []
        self.state = torch.empty(0)
        self.action = torch.empty(0)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def decide_action(self, state):
        self.state = torch.cat((self.state, state), dim=0)
        action_mean, action_std = self.model(state)

        # Sample action
        dist = torch.distributions.Normal(action_mean, action_std)
        action = dist.sample() 
        self.action = torch.cat((self.action, action), dim=0)

        return action

    def compute_discounted_rewards(self):
        rewards = np.array(self.rewards)
        R = 0
        returns = []
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        return torch.tensor(returns, dtype=torch.float32)
    
    def update_model(self):
        # Compute discounted rewards efficiently
        rewards = np.array(self.rewards)
        #R = torch.tensor([np.sum(rewards[i:]*(gamma**np.array(range(i, len(rewards))))) for i in range(len(rewards))])
        R = self.compute_discounted_rewards()
        r_std = R.std()
        #print (r_std)
        if torch.isnan(r_std).any() or torch.isinf(r_std).any():
            
            # Reset memory
            self.rewards.clear()
            self.state = torch.empty(0)
            self.action = torch.empty(0)
            return True
        else:
            R = (R - R.mean()) / R.std()
        # Stack states and actions into tensors
        states = self.state
        actions = self.action

        # Compute loss
        action_mean, action_std = self.model(self.state)

        if torch.isnan(action_std).any() or torch.isinf(action_std).any():
            print("Warning: action_std contains NaN or Inf!", action_std)
            return
        dist = torch.distributions.Normal(action_mean, action_std)
        log_probs = dist.log_prob(actions).sum(dim=-1)  # Sum over action dimensions
        
        loss = []
        for prob, r in zip(log_probs, R):
            loss.append(-prob * r) 
        loss = torch.stack(loss).sum()

        # Update model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Reset memory
        self.rewards.clear()
        self.state = torch.empty(0)
        self.action = torch.empty(0)
        return False

    def add_reward(self, reward):
        self.rewards.append(reward)
        
