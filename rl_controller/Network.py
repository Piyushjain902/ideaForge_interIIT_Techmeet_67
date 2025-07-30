import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --- Neural Network Architectures ---
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim=21, action_dim=4, hidden_dim=256):
        """
        Initializes the Policy Network.

        Args:
            state_dim (int): Dimension of the input state.
            action_dim (int): Dimension of the output action.
            hidden_dim (int): Number of neurons in hidden layers.
        """
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.action_mean = nn.Linear(hidden_dim, action_dim)
        self.action_log_std = nn.Parameter(torch.zeros(action_dim))  # Learnable log standard deviation

        self.apply(self.weights_init)

    def forward(self, state):
        """
        Forward pass through the network.

        Args:
            state (torch.Tensor): Input state tensor.

        Returns:
            tuple: Mean and standard deviation of the action distribution.
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.action_mean(x)
        std = torch.exp(self.action_log_std)
        return mean, std

    @staticmethod
    def weights_init(m):
        """
        Initializes network weights using Xavier Uniform Initialization.

        Args:
            m (nn.Module): Network module.
        """
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)


class ValueNetwork(nn.Module):
    def __init__(self, state_dim=21, hidden_dim=256):
        """
        Initializes the Value Network.

        Args:
            state_dim (int): Dimension of the input state.
            hidden_dim (int): Number of neurons in hidden layers.
        """
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, 1)

        self.apply(self.weights_init)

    def forward(self, state):
        """
        Forward pass through the network.

        Args:
            state (torch.Tensor): Input state tensor.

        Returns:
            torch.Tensor: Estimated value of the input state.
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        v = self.value(x)
        return v

    @staticmethod
    def weights_init(m):
        """
        Initializes network weights using Xavier Uniform Initialization.

        Args:
            m (nn.Module): Network module.
        """
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
