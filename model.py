import torch.nn as nn
import torch.nn.functional as F
import torch

class VPG(nn.Module):
    def __init__(self, obs_dim=6, act_dim=2, hl=[512, 512, 512]) -> None:
        super(VPG, self).__init__()
        layers = []
        layers.append(nn.Linear(obs_dim, hl[0]))
        layers.append(nn.ReLU())
        for i in range(1, len(hl)):
            layers.append(nn.Linear(hl[i-1], hl[i]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hl[-1], act_dim*2))  # act_dim * (1 for mean + 1 for std)
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.model(x)
        mean, std = x.chunk(2, dim=-1)
        std = F.softplus(torch.clamp(std, -20, 20)) + 1e-6
        return mean, std
        