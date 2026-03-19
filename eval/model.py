import torch
import torch.nn as nn
import torch.nn.functional as F
import pentafluoride

class TetrisNet(nn.Module):
    def __init__(self, board_channels=4, scalar_input_dim=15, num_actions=40):
        super(TetrisNet, self).__init__()
        
        self.conv_branch = nn.Sequential(
            nn.Conv2d(board_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten() # Result: 64 * 20 * 10 = 12,800
        )
        
        # Scalar branch for hold piece, combo, b2b, and queue pieces
        self.scalar_branch = nn.Sequential(
            nn.Linear(scalar_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        # --- 3. MERGED LAYERS ---
        # Combine CNN (12800) + MLP (64)
        combined_dim = 12800 + 64
        
        # Policy Head: Probability of moves
        self.policy_head = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions),
            nn.LogSoftmax(dim=1)
        )
        
        # Value Head: Board quality (-1 to 1)
        self.value_head = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )

    def forward(self, board, scalars):
        # Process branches in parallel
        spatial_out = self.conv_branch(board)
        scalar_out = self.scalar_branch(scalars)
        
        # Late Fusion: Concatenate the results
        merged = torch.cat((spatial_out, scalar_out), dim=1)
        
        # Final outputs
        policy = self.policy_head(merged)
        value = self.value_head(merged)
        
        return policy, value
    def save_model(self, path):
        torch.save(self.state_dict(), path)
    
    def load_model(self, path):
        self.load_state_dict(torch.load(path))
    
    def evaluate_gamestate(self, game_state: pentafluoride.GameState, queue: list):
        # Convert game state to model input format
        board = [[1 if row & (1<<cell) == 1 else 0 for cell in range(20)] for row in game_state.board] # converts bitboard to vec
        board_tensor = torch.tensor(board, dtype=torch.float32).unsqueeze(0) # convert vec to tensor

        scalar_input = [
            encode_piece(game_state.hold), 
            game_state.combo,     
            game_state.b2b        
        ], + [encode_piece(piece) for piece in queue]  # might use bag instead
        scalar_tensor = torch.tensor(scalar_input, dtype=torch.float32)
        
        
        board_tensor = board_tensor.unsqueeze(0)
        scalar_tensor = scalar_tensor.unsqueeze(0)
        
        with torch.no_grad():
            policy, value = self.forward(board_tensor, scalar_tensor)
        
        return policy.squeeze(0), value.item()

def encode_piece(piece):
    # One-hot encode the piece type
    match piece:
        case pentafluoride.Piece.I:
            return [1, 0, 0, 0, 0, 0, 0]
        case pentafluoride.Piece.O:
            return [0, 1, 0, 0, 0, 0, 0]
        case pentafluoride.Piece.T:
            return [0, 0, 1, 0, 0, 0, 0]
        case pentafluoride.Piece.S:
            return [0, 0, 0, 1, 0, 0, 0]
        case pentafluoride.Piece.Z:
            return [0, 0, 0, 0, 1, 0, 0]
        case pentafluoride.Piece.J:
            return [0, 0, 0, 0, 0, 1, 0]
        case pentafluoride.Piece.L:
            return [0, 0, 0, 0, 0, 0, 1]
    return [0, 0, 0, 0, 0, 0, 0]