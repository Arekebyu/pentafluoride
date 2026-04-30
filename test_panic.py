import pentafluoride as pf
import random

def run():
    random.seed(42)
    queue = [pf.Piece.L, pf.Piece.J, pf.Piece.O, pf.Piece.T, pf.Piece.S, pf.Piece.Z, pf.Piece.I]
    random.shuffle(queue)
    game_state = pf.GameState(
        board=[0 for _ in range(10)],
        hold=queue.pop(0),
        combo=0,
        b2b=0,
    )
    placements = pf.find_moves(game_state.board, queue[0]) + pf.find_moves(game_state.board, game_state.hold)
    
    for step in range(500):
        if len(placements) == 0: break
        action = random.randint(0, len(placements) - 1)
        placement = placements[action][0]
        active_piece = queue.pop(0)
        
        if len(queue) < 5:
            new_bag = [pf.Piece.L, pf.Piece.J, pf.Piece.O, pf.Piece.T, pf.Piece.S, pf.Piece.Z, pf.Piece.I]
            random.shuffle(new_bag)
            queue.extend(new_bag)
            
        game_state.advance(active_piece, placement)
        try:
            placements = pf.find_moves(game_state.board, queue[0])
            placements += pf.find_moves(game_state.board, game_state.hold)
        except BaseException as e:
            print(f"Panic at step {step}: {e}")
            print(f"Board: {game_state.board.cols}")
            print(f"Piece: {queue[0]}")
            print(f"Hold: {game_state.hold}")
            break

run()
