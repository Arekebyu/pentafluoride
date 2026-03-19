import pentafluoride

if __name__ == "__main__":
    root = pentafluoride.GameState(
        board=[0 for _ in range(10)],
        hold=pentafluoride.Piece.I,
        combo=0,
        b2b=0,
    )
    queue = [pentafluoride.Piece.L, pentafluoride.Piece.J, pentafluoride.Piece.O, pentafluoride.Piece.T, pentafluoride.Piece.S]
    evaluator = lambda s, q: (1.0, 0.0)
    # placement = pentafluoride.mcts_search(root, queue, 200, evaluator)
    # print(placement.spin, placement.location.x, placement.location.y, placement.location.rotation, placement.location.piece)
    print(pentafluoride.Piece.L.value)
