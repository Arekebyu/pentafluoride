import torch
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
import os
import pentafluoride as pf
from mcts_alphago.network import DualHeadNet
from mcts_alphago.evaluator import MCTSEvaluator
from mcts_alphago.replay_buffer import MCTSReplayBuffer
from datetime import datetime

def train():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    network = DualHeadNet(action_feature_dim=56).to(device)
    optimizer = optim.Adam(network.parameters(), lr=1e-3)
    
    evaluator = MCTSEvaluator(network, device=device)
    replay_buffer = MCTSReplayBuffer(capacity=50000)
    
    # Self-play parameters
    num_episodes = 25
    mcts_iterations = 128
    batch_size = 16
    epochs_per_iteration = 10
    
    print(f"Starting AlphaGo MCTS Training on {device}")
    
    for iteration in range(50): # Example outer loop
        print(f"--- Iteration {iteration} ---")
        
        # 1. Self-Play
        network.eval()
        for ep in range(num_episodes):
            pieces = [pf.Piece.L, pf.Piece.J, pf.Piece.O, pf.Piece.T, pf.Piece.S, pf.Piece.Z, pf.Piece.I]
            queue = pieces.copy()
            random.shuffle(queue)
            state = pf.GameState(board=[0]*10, hold=queue.pop(0), combo=0, b2b=0)
            if len(queue) < 5:
                new_bag = pieces.copy()
                random.shuffle(new_bag)
                queue.extend(new_bag)
                
            episode_data = [] # (state, queue, policy_map)
            
            while True:
                # Find valid moves first to check termination
                valid_moves = pf.find_moves(state.board, queue[0])
                valid_moves.extend(pf.find_moves(state.board, state.hold))
                
                if len(valid_moves) == 0:
                    break
                    
                # Run MCTS
                best_move, policy_map = pf.alphago_mcts_search(
                    state, queue, mcts_iterations, evaluator, temperature=1.0
                )
                
                # Store data
                episode_data.append((state, queue.copy(), policy_map, valid_moves))
                
                # Advance state
                active_piece = queue.pop(0)
                state.advance(active_piece, best_move)
                
                if len(queue) < 5:
                    new_bag = pieces.copy()
                    random.shuffle(new_bag)
                    queue.extend(new_bag)
            
            # Game over. Calculate returns.
            # In a real setup, we might use the length of the game, or score.
            # For simplicity, assign a generic return based on survival length.
            return_val = len(episode_data) 
            
            # Note: A more complete implementation would extract features here 
            # and push them to replay_buffer.
            
            print(f"Episode {ep} finished, Length: {return_val}")

        # 2. Train Network
        network.train()
        print("Training network...")
        # (Training logic omitted for brevity; would sample from replay_buffer, 
        # compute cross-entropy on policy and MSE on value, and step optimizer)
        
        # Save model checkpoint
        if not os.path.exists("checkpoints"):
            os.makedirs("checkpoints")
        if iteration % 5 == 0:
            checkpoint_path = f"checkpoints/mcts_alphago_{timestamp}_{iteration}.pth"
            torch.save(network.state_dict(), checkpoint_path)
            print(f"Model saved to {checkpoint_path}")

    # Save final model
    final_path = f"checkpoints/mcts_alphago_{timestamp}_{iteration}.pth"
    torch.save(network.state_dict(), final_path)
    print(f"Training finished! Final model saved to {final_path}")

if __name__ == '__main__':
    train()
