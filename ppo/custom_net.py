import torch
import torch.nn as nn

class TetrisValueNetwork(nn.Module):
    """
    An off-policy Q-Network for Tetris.
    Evaluates the expected future reward (Q-value) of each available action
    by computing a dot product between the state embedding and action embeddings.
    """
    def __init__(self, state_dim, action_feature_dim, embedding_dim=256):
        super().__init__()
        
        # Tower 1: Encodes the state
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim)
        )
        
        # Tower 2: Encodes the action features (shared weights across all actions)
        self.action_encoder = nn.Sequential(
            nn.Linear(action_feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim)
        )

    def forward(self, state, action_features, action_mask=None):
        """
        Args:
            state: [Batch, state_dim] or [Batch, channels, cols]
            action_features: [Batch, N_actions, action_feature_dim]
            action_mask: [Batch, N_actions] (1 for valid, 0 for invalid)
        Returns:
            q_values: [Batch, N_actions]
        """
        # Ensure state is flattened: [Batch, State_Dim]
        if state.dim() > 2:
            state = state.flatten(start_dim=1)

        # 1. Encode state: [Batch, State_Dim] -> [Batch, Embedding_Dim, 1]
        state_emb = self.state_encoder(state).unsqueeze(2) 

        # 2. Encode actions: [Batch, N_actions, Action_Dim] -> [Batch, N_actions, Embedding_Dim]
        action_emb = self.action_encoder(action_features) 

        # 3. Dot Product: [Batch, N_actions, Embedding_Dim] x [Batch, Embedding_Dim, 1] -> [Batch, N_actions]
        q_values = torch.bmm(action_emb, state_emb).squeeze(2) 

        # 4. Masking invalid actions
        # We subtract a very large number from invalid actions so their Q-value is essentially -infinity.
        if action_mask is not None:
            MASK_PENALTY = 1e9
            q_values = q_values + (action_mask - 1.0) * MASK_PENALTY

        return q_values
