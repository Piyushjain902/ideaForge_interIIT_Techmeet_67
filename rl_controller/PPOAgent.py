import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

from rl_controller.Network import PolicyNetwork, ValueNetwork


# --- PPO Agent Implementation ---
class PPOAgent:
    def __init__(self, state_dim=21, action_dim=4, hidden_dim=256, 
                 lr_policy=3e-4, lr_value=1e-3, gamma=0.99, 
                 clip_epsilon=0.2, entropy_coef=0.01, vf_coef=0.5, max_grad_norm=0.5,
                 device='cpu',model_path='ppo_quadrotor.pth'):
        """
        Initializes the PPO Agent.

        Args:
            state_dim (int): Dimension of the input state.
            action_dim (int): Dimension of the action space.
            hidden_dim (int): Number of neurons in hidden layers.
            lr_policy (float): Learning rate for the policy network.
            lr_value (float): Learning rate for the value network.
            gamma (float): Discount factor.
            clip_epsilon (float): Clipping parameter for PPO.
            entropy_coef (float): Coefficient for entropy bonus.
            vf_coef (float): Coefficient for value function loss.
            max_grad_norm (float): Maximum gradient norm for clipping.
            device (str): Device to run computations on ('cpu' or 'cuda').
        """
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.device = torch.device(device)

        # Initialize networks
        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.value_net = ValueNetwork(state_dim, hidden_dim).to(self.device)

        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr_policy)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr_value)

        # Memory for storing trajectories
        self.memory = {
            'states': [],
            'actions': [],
            'logprobs': [],
            'rewards': [],
            'is_terminals': []
        }
        
        if os.path.exists(model_path):
            self.load_model(path=model_path)
            

    def select_action(self, state):
        """
        Selects an action based on the current state.

        Args:
            state (np.ndarray): Current state.

        Returns:
            tuple: Action (numpy array) and log probability of the action.
        """
        state = torch.FloatTensor(state).to(self.device)
        mean, std = self.policy_net(state)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        action_logprob = dist.log_prob(action).sum(dim=-1)
        action = torch.clamp(action, 0.4, 1.0)  # Ensure actions are within [0,1] bounds

        # Convert logprob from tensor to scalar float
        logprob_scalar = action_logprob.cpu().detach().numpy().item()

        return action.cpu().detach().numpy(), logprob_scalar

    def store_transition(self, state, action, logprob, reward, done):
        """
        Stores a transition in memory.

        Args:
            state (np.ndarray): Current state.
            action (np.ndarray): Action taken.
            logprob (float): Log probability of the action.
            reward (float): Reward received.
            done (bool): Whether the episode has ended.
        """
        # Ensure logprob is a scalar float
        if isinstance(logprob, np.ndarray):
            logprob = logprob.item()
        elif isinstance(logprob, torch.Tensor):
            logprob = logprob.cpu().detach().numpy().item()
        elif not isinstance(logprob, float):
            logprob = float(logprob)

        self.memory['states'].append(state)
        self.memory['actions'].append(action)
        self.memory['logprobs'].append(logprob)
        self.memory['rewards'].append(reward)
        self.memory['is_terminals'].append(done)

    def clear_memory(self):
        """Clears the memory after an update."""
        self.memory = {
            'states': [],
            'actions': [],
            'logprobs': [],
            'rewards': [],
            'is_terminals': []
        }

    def compute_gae(self, rewards, dones, values, next_values, gamma=0.99, lam=0.95):
        """
        Computes Generalized Advantage Estimation (GAE).

        Args:
            rewards (list): Rewards collected.
            dones (list): Done flags.
            values (torch.Tensor): Predicted state values.
            next_values (torch.Tensor): Predicted next state values.
            gamma (float): Discount factor.
            lam (float): GAE lambda.

        Returns:
            torch.Tensor: Advantage estimates.
            torch.Tensor: Target values.
        """
        advantages = []
        gae = 0
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + gamma * next_values[step] * (1 - dones[step]) - values[step]
            gae = delta + gamma * lam * (1 - dones[step]) * gae
            advantages.insert(0, gae)
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        returns = advantages + torch.tensor(values, dtype=torch.float32).to(self.device)
        return advantages, returns

    def update(self, K_epochs=4, batch_size=64):
        """
        Updates the policy and value networks using PPO.

        Args:
            K_epochs (int): Number of epochs to update.
            batch_size (int): Mini-batch size.
        """
        # Debugging: Check the type and contents of logprobs
        logprobs = self.memory['logprobs']
        print(f"Type of self.memory['logprobs']: {type(logprobs)}")
        if isinstance(logprobs, list):
            print(f"Length of self.memory['logprobs']]: {len(logprobs)}")
            print(f"First few logprobs: {logprobs[:5]}")
        else:
            print("self.memory['logprobs'] is not a list.")

        # Check if logprobs is a list
        if not isinstance(logprobs, list):
            raise TypeError(f"Expected 'logprobs' to be a list, but got {type(logprobs)}")

        # Check if logprobs is empty
        if len(logprobs) == 0:
            print("No log probabilities stored. Skipping update.")
            return

        # Convert lists to tensors efficiently by converting to numpy array first
        try:
            states = torch.FloatTensor(np.array(self.memory['states'], dtype=np.float32)).to(self.device)
            actions = torch.FloatTensor(np.array(self.memory['actions'], dtype=np.float32)).to(self.device)
            old_logprobs = torch.FloatTensor(np.array(self.memory['logprobs'], dtype=np.float32)).to(self.device)
        except Exception as e:
            print(f"Error converting memory to tensors: {e}")
            return

        rewards = self.memory['rewards']
        dones = self.memory['is_terminals']

        # Check if rewards and dones have the same length as logprobs
        if not (len(rewards) == len(dones) == len(logprobs)):
            print("Mismatch in lengths of rewards, dones, and logprobs.")
            return

        # Compute state values
        with torch.no_grad():
            values = self.value_net(states).squeeze().cpu().numpy()
            next_states = torch.FloatTensor(np.array([self.memory['states'][-1]], dtype=np.float32)).to(self.device)
            next_values = self.value_net(next_states).squeeze().cpu().numpy()
            next_values = np.append(values[1:], next_values)

        # Compute advantages and returns
        advantages, returns = self.compute_gae(rewards, dones, values, next_values, self.gamma, 0.95)

        # Normalize advantages if not empty
        if len(advantages) > 0:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        else:
            print("Computed advantages are empty. Skipping normalization.")

        # Optimize policy for K epochs
        for epoch in range(K_epochs):
            # Shuffle the indices
            indices = np.arange(len(states))
            np.random.shuffle(indices)

            for start in range(0, len(states), batch_size):
                end = start + batch_size
                minibatch_indices = indices[start:end]

                mb_states = states[minibatch_indices]
                mb_actions = actions[minibatch_indices]
                mb_old_logprobs = old_logprobs[minibatch_indices]
                mb_advantages = advantages[minibatch_indices]
                mb_returns = returns[minibatch_indices]

                # Evaluating old actions and values
                logprobs, state_values, dist_entropy = self.evaluate(mb_states, mb_actions)

                # Finding the ratio (pi_theta / pi_theta_old)
                ratios = torch.exp(logprobs - mb_old_logprobs)

                # Finding Surrogate Loss
                surr1 = ratios * mb_advantages
                surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * mb_advantages

                # Final loss of clipped objective PPO
                loss = (-torch.min(surr1, surr2)).mean() + self.vf_coef * F.mse_loss(state_values, mb_returns) - self.entropy_coef * dist_entropy.mean()

                # Take gradient step
                self.policy_optimizer.zero_grad()
                self.value_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm)
                self.policy_optimizer.step()
                self.value_optimizer.step()

        # Clear memory after update
        self.clear_memory()
        print("Update completed successfully.")

    def evaluate(self, state, action):
        """
        Evaluates the action log probabilities, state values, and entropy.

        Args:
            state (torch.Tensor): Batch of states.
            action (torch.Tensor): Batch of actions.

        Returns:
            tuple: Log probabilities, state values, and entropy.
        """
        mean, std = self.policy_net(state)
        dist = torch.distributions.Normal(mean, std)
        action_logprobs = dist.log_prob(action).sum(dim=-1)
        dist_entropy = dist.entropy().sum(dim=-1)
        state_values = self.value_net(state).squeeze()
        return action_logprobs, state_values, dist_entropy

    def save_model(self, path='ppo_quadrotor.pth'):
        """
        Saves the policy and value networks.

        Args:
            path (str): File path to save the model.
        """
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'value_net_state_dict': self.value_net.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
        }, path)
        print(f"Model saved to {path}")

    def load_model(self, path='ppo_quadrotor.pth'):
        """
        Loads the policy and value networks.

        Args:
            path (str): File path from which to load the model.
        """
        if os.path.exists(path):
            try:
                # Load the checkpoint from the specified path
                checkpoint = torch.load(path, map_location=self.device)
                
                # Load the state dictionaries into the respective networks and optimizers
                self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
                self.value_net.load_state_dict(checkpoint['value_net_state_dict'])
                self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
                self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
                
                print(f"Model loaded successfully from {path}")
    
            except FileNotFoundError:
                print(f"Error: The file at {path} was not found.")
            
            except KeyError as e:
                print(f"Error: Missing key in checkpoint: {e}")
            
            except RuntimeError as e:
                print(f"RuntimeError while loading the model: {e}")
            
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                
        else:
            print(f"No model found at {path}. Initializing new networks.")
