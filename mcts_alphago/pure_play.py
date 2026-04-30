import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pentafluoride as pf

def play_pure_mcts(output_path="pure_tetris.mp4", num_steps=100):
    print("Initializing Pure MCTS (No ML)...")
    
    # Initialize game
    pieces = [pf.Piece.L, pf.Piece.J, pf.Piece.O, pf.Piece.T, pf.Piece.S, pf.Piece.Z, pf.Piece.I]
    queue = pieces.copy()
    random.shuffle(queue)
    state = pf.GameState(board=[0]*10, hold=queue.pop(0), combo=0, b2b=0)
    
    if len(queue) < 5:
        new_bag = pieces.copy()
        random.shuffle(new_bag)
        queue.extend(new_bag)

    total_score = 0
    frames = []
    print(f"Starting game for {num_steps} steps...")
    
    for step in range(num_steps):
        valid_moves = pf.find_moves(state.board, queue[0])
        valid_moves.extend(pf.find_moves(state.board, state.hold))
        
        if not valid_moves:
            print(f"Game over at step {step}. Total Score: {total_score}")
            break
            
        # Extract visual state
        board_array = np.zeros((31, 10))
        for x in range(10):
            col = state.board.cols[x]
            for y in range(31):
                if col & (1 << y):
                    board_array[30 - y, x] = 1 # Invert y so row 0 is bottom visually
        frames.append(board_array)
            
        # We can run thousands of iterations easily since there's no CNN
        iterations = 1000 
        
        best_move, _ = pf.pure_mcts_search(state, queue, iterations)
        
        active_piece = queue.pop(0)
        info = state.advance(active_piece, best_move)
        reward = pf.calculate_reward(info)
        total_score += reward
        
        if step % 10 == 0:
            print(f"Step {step}, Current Score: {total_score}")
        
        if len(queue) < 5:
            new_bag = pieces.copy()
            random.shuffle(new_bag)
            queue.extend(new_bag)

    print("Rendering animation...")
    fig, ax = plt.subplots()
    ax.axis('off')
    if frames:
        im = ax.imshow(frames[0], cmap='Greens', vmin=0, vmax=1)
        
        def update(frame_idx):
            im.set_array(frames[frame_idx])
            ax.set_title(f"Pure MCTS Step: {frame_idx + 1}/{len(frames)}")
            return [im]
            
        anim = animation.FuncAnimation(fig, update, frames=len(frames), blit=False)
        anim.save(output_path, fps=5, extra_args=['-vcodec', 'libx264'])
        print(f"Animation saved to {output_path}. Final Score: {total_score}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize Pure MCTS gameplay.")
    parser.add_argument("--output", type=str, default="pure_tetris.mp4", help="Output path for the animation")
    parser.add_argument("--steps", type=int, default=100, help="Number of steps to play")
    args = parser.parse_args()
    
    play_pure_mcts(output_path=args.output, num_steps=args.steps)
