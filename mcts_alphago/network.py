import torch
import torch.nn as nn
import torch.nn.functional as F

class DualHeadNet(nn.Module):
    """
    AlphaZero-style Dual-Headed Network for Tetris.
    
    Inputs:
        board: [B, 1, 31, 10] - Raw board state
        current_piece: [B, 7] - One-hot encoded
        queue: [B, 42] - 5 one-hot pieces (35) + 7-dim bag probability vector
        scalars: [B, 2] - b2b and combo
        heuristics: [B, num_heuristics] - E.g. holes, bumpiness, max_height
        action_features: [B, N, action_feature_dim] - Features of valid moves
        
    Outputs:
        policy_logits: [B, N] - Unnormalized log probabilities for each valid move
        value: [B, 1] - Expected future value
    """
    def __init__(self, action_feature_dim, num_heuristics=6, embedding_dim=256):
        super().__init__()
        
        # Board CNN
        self.board_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 16x5
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 8x3
            nn.ReLU(),
            nn.Flatten()
        )
        
        # After flatten, we have 128 * 8 * 3 = 3072 features.
        
        # State vector dimension: board(3072) + piece(7) + queue(42) + scalars(2) + heuristics(num_heuristics)
        self.state_fc_input_dim = 3072 + 7 + 42 + 2 + num_heuristics
        
        # Shared State MLP
        self.shared_mlp = nn.Sequential(
            nn.Linear(self.state_fc_input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, embedding_dim),
            nn.ReLU()
        )
        
        # Value Head
        self.value_head = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1) # Outputs a scalar value
        )
        
        # Policy Head (Action Encoder for dynamic action space)
        # Instead of fixed-size output, we use dot-product attention over action features
        self.action_encoder = nn.Sequential(
            nn.Linear(action_feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim)
        )
        
    def forward(self, board, current_piece, queue, scalars, heuristics, action_features, action_mask=None):
        """
        Args:
            board: [B, 1, 31, 10]
            current_piece: [B, 7]
            queue: [B, 42]
            scalars: [B, 2]
            heuristics: [B, num_heuristics]
            action_features: [B, N, action_feature_dim]
            action_mask: [B, N] (Optional boolean mask for valid actions)
        """
        B, N, _ = action_features.shape
        
        # Process board
        board_features = self.board_conv(board)
        
        # Concatenate all state info
        state_input = torch.cat([board_features, current_piece, queue, scalars, heuristics], dim=1)
        
        # Shared representation
        state_emb = self.shared_mlp(state_input) # [B, embedding_dim]
        
        # 1. Compute Value
        value = self.value_head(state_emb) # [B, 1]
        
        # 2. Compute Policy Logits
        # Encode actions: [B, N, Action_Dim] -> [B, N, embedding_dim]
        action_emb = self.action_encoder(action_features)
        
        # Dot product between state embedding and action embeddings
        # state_emb: [B, embedding_dim, 1]
        # action_emb: [B, N, embedding_dim]
        # Result: [B, N]
        policy_logits = torch.bmm(action_emb, state_emb.unsqueeze(2)).squeeze(2)
        
        # Mask invalid actions
        if action_mask is not None:
            # Mask out invalid actions with a large negative number
            policy_logits = policy_logits.masked_fill(~action_mask, -1e9)
            
        return policy_logits, value
