import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
import random
import numpy as np
import argparse
import pentafluoride as pf
from mcts_alphago.network import DualHeadNet
from mcts_alphago.evaluator import MCTSEvaluator

def visualize_game(network_path=None, output_path="tetris_mcts.mp4", num_steps=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    network = DualHeadNet(action_feature_dim=56).to(device)
    if network_path:
        network.load_state_dict(torch.load(network_path, map_location=device))
        
    evaluator = MCTSEvaluator(network, device=device)
    
    # Initialize game
    pieces = [pf.Piece.L, pf.Piece.J, pf.Piece.O, pf.Piece.T, pf.Piece.S, pf.Piece.Z, pf.Piece.I]
    queue = pieces.copy()
    random.shuffle(queue)
    state = pf.GameState(board=[0]*10, hold=queue.pop(0), combo=0, b2b=0)
    if len(queue) < 5:
        new_bag = pieces.copy()
        random.shuffle(new_bag)
        queue.extend(new_bag)

    frames = []
    print(f"Generating gameplay for {num_steps} steps...")

    for step in range(num_steps):
        valid_moves = pf.find_moves(state.board, queue[0])
        valid_moves.extend(pf.find_moves(state.board, state.hold))
        
        if not valid_moves:
            break
            
        # Extract visual state
        board_array = np.zeros((31, 10))
        for x in range(10):
            col = state.board.cols[x]
            for y in range(31):
                if col & (1 << y):
                    board_array[30 - y, x] = 1 # Invert y so row 0 is bottom visually
        frames.append(board_array)
        
        # MCTS
        best_move, policy_map = pf.alphago_mcts_search(
            state, queue, 25, evaluator, temperature=0.1
        )
        
        active_piece = queue.pop(0)
        state.advance(active_piece, best_move)
        
        if len(queue) < 5:
            new_bag = pieces.copy()
            random.shuffle(new_bag)
            queue.extend(new_bag)

    print("Rendering animation...")
    fig, ax = plt.subplots()
    ax.axis('off')
    im = ax.imshow(frames[0], cmap='Blues', vmin=0, vmax=1)
    
    def update(frame_idx):
        im.set_array(frames[frame_idx])
        ax.set_title(f"Step: {frame_idx + 1}/{len(frames)}")
        return [im]
        
    anim = animation.FuncAnimation(fig, update, frames=len(frames), blit=False)
    anim.save(output_path, fps=5, extra_args=['-vcodec', 'libx264'])
    print(f"Animation saved to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize AlphaGo MCTS gameplay.")
    parser.add_argument("--model", type=str, default=None, help="Path to the saved model (.pth file)")
    parser.add_argument("--output", type=str, default="tetris_mcts.mp4", help="Output path for the animation")
    parser.add_argument("--steps", type=int, default=100, help="Number of steps to play")
    args = parser.parse_args()
    
    visualize_game(network_path=args.model, output_path=args.output, num_steps=args.steps)
