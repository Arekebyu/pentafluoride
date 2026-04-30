import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
import gymnasium as gym
from gymnasium_env.envs import TetrisEnv

class ParametricActionNet(nn.Module):
    def __init__(self, state_dim, action_feature_dim, embedding_dim=256):
        super().__init__()
        
        # Tower 1: Encodes the state
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim)
        )
        
        # Tower 2: Encodes the action features (shared weights)
        self.action_encoder = nn.Sequential(
            nn.Linear(action_feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim)
        )

    def forward(self, state, action_features, action_mask=None):
        # 1. Encode state: [Batch, State_Dim] -> [Batch, Embedding_Dim, 1]
        state_emb = self.state_encoder(state.flatten(start_dim=1)).unsqueeze(2) 

        # 2. Encode all 128 actions: [Batch, 128, Action_Dim] -> [Batch, 128, Embedding_Dim]
        action_emb = self.action_encoder(action_features) 

        # 3. Dot Product: [Batch, 128, Embedding_Dim] x [Batch, Embedding_Dim, 1] -> [Batch, 128]
        logits = torch.bmm(action_emb, state_emb).squeeze(2) 

        # 4. Masking invalid actions (if a mask is provided)
        if action_mask is not None:
            # action_mask should be 1 for valid, 0 for invalid
            MASK_PENALTY = 1e9
            logits = logits + (action_mask - 1.0) * MASK_PENALTY

        return logits

class TwoTowerPolicy(ActorCriticPolicy):
    def __init__(self, observation_space: gym.spaces.Dict, action_space: gym.spaces.Discrete, lr_schedule, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)
        
    def _build(self, lr_schedule):
        """Overrides the default network building process."""
        # Initialize standard SB3 distributions and optimizers
        super()._build(lr_schedule)
        
        # Extract dimensions from your dictionary observation space
        # state_dim = self.observation_space["state"].shape[0]
        state_dim = TetrisEnv.state_dim  # This is a property we defined in our TetrisEnv
        action_feature_dim = TetrisEnv.action_dim  # This is also a property we defined in our TetrisEnv
        
        # Define the Actor (Our custom Parametric Action Net)
        self.parametric_actor = ParametricActionNet(state_dim, action_feature_dim)
        
        # Define the Critic (Value Function). 
        # The Critic generally only needs to look at the state to estimate its value.
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        # Re-initialize the optimizer with our new custom network parameters
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def forward(self, obs, deterministic=False):
        """Used during rollout collection (interacting with the environment)."""
        state = obs["state"]
        action_features = obs["action_features"]
        
        # It is highly recommended to add an 'action_mask' to your Dict space.
        # If you don't have one, default to None (or infer it if invalid actions are all zeros).
        action_mask = obs.get("action_mask", None) 

        # 1. Get Logits from the Two-Tower network
        logits = self.parametric_actor(state, action_features, action_mask)
        
        # 2. Create the probability distribution
        distribution = self.action_dist.proba_distribution(action_logits=logits)
        
        # 3. Sample an action
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        
        # 4. Get Value estimate from the Critic
        values = self.value_net(state.flatten(start_dim=1))
        
        return actions, values, log_prob

    def _predict(self, obs, deterministic=False):
        """Used during model.predict() (evaluating the trained model)."""
        state = obs["state"]
        action_features = obs["action_features"]
        action_mask = obs.get("action_mask", None)
        
        logits = self.parametric_actor(state, action_features, action_mask)
        distribution = self.action_dist.proba_distribution(action_logits=logits)
        
        return distribution.get_actions(deterministic=deterministic)

    def predict_values(self, obs):
        """Get the estimated values according to the current policy given the observations."""
        state = obs["state"]
        return self.value_net(state.flatten(start_dim=1))

    def evaluate_actions(self, obs, actions):
        """Used during the PPO update step to calculate loss and gradients."""
        state = obs["state"]
        action_features = obs["action_features"]
        action_mask = obs.get("action_mask", None)
        
        logits = self.parametric_actor(state, action_features, action_mask)
        distribution = self.action_dist.proba_distribution(action_logits=logits)
        
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        values = self.value_net(state.flatten(start_dim=1))
        
        return values, log_prob, entropy