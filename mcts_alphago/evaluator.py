import torch
import numpy as np

class MCTSEvaluator:
    def __init__(self, network, device='cpu'):
        self.network = network
        self.device = device
        self.network.to(self.device)
        self.network.eval()
        
    def __call__(self, state, queue, placements):
        """
        Called from Rust MCTS.
        state: pentafluoride.GameState
        queue: list of pentafluoride.Piece
        placements: list of pentafluoride.Placement
        """
        if len(placements) == 0:
            return [], 0.0

        # 1. Build board tensor [1, 1, 31, 10]
        board = np.zeros((1, 1, 31, 10), dtype=np.float32)
        heights = [0] * 10
        for x in range(10):
            col = state.board.cols[x]
            for y in range(31):
                if col & (1 << y):
                    board[0, 0, y, x] = 1.0
                    heights[x] = y + 1
                    
        # 2. Build current piece [1, 7]
        current_piece = np.zeros((1, 7), dtype=np.float32)
        if len(queue) > 0:
            current_piece[0, queue[0].value] = 1.0
            
        # 3. Build queue tensor [1, 42]
        queue_tensor = np.zeros((1, 42), dtype=np.float32)
        for i in range(min(5, len(queue) - 1)):
            queue_tensor[0, i * 7 + queue[i + 1].value] = 1.0
            
        # Bag probability calculation (simplified baseline)
        bag_counts = [0] * 7
        for p in queue[:6]: # current piece + up to 5 in queue
            bag_counts[p.value] += 1
            
        # Remaining pieces in current 7-bag
        # We assume one of each piece per bag.
        # This is a heuristic baseline for the bag state.
        for i in range(7):
            remaining = 1 - (bag_counts[i] % 1)
            queue_tensor[0, 35 + i] = remaining / max(1, sum(1 - (c%1) for c in bag_counts))
            
        # 4. Scalars [1, 2]
        scalars = np.array([[state.b2b / 64.0, state.combo / 21.0]], dtype=np.float32)
        
        # 5. Heuristics [1, 6]
        holes = 0
        for x in range(10):
            col = state.board.cols[x]
            for y in range(heights[x]):
                if not (col & (1 << y)):
                    holes += 1
        bumpiness = sum(abs(heights[i] - heights[i+1]) for i in range(9))
        max_height = max(heights)
        sum_heights = sum(heights)
        
        heuristics = np.zeros((1, 6), dtype=np.float32)
        heuristics[0, 0] = holes / 40.0
        heuristics[0, 1] = bumpiness / 40.0
        heuristics[0, 2] = max_height / 32.0
        heuristics[0, 3] = sum_heights / 320.0
        
        # 6. Action Features [1, N, 56]
        action_features = []
        for p in placements:
            feature = [0] * 56
            feature[p.location.piece.value] = 1
            feature[7 + p.location.rotation.value] = 1
            feature[11 + p.location.x] = 1
            feature[21 + min(p.location.y, 31)] = 1
            feature[53 + p.spin.value] = 1
            action_features.append(feature)
            
        action_features_tensor = np.array([action_features], dtype=np.float32)
        
        with torch.no_grad():
            b = torch.from_numpy(board).to(self.device)
            cp = torch.from_numpy(current_piece).to(self.device)
            q = torch.from_numpy(queue_tensor).to(self.device)
            s = torch.from_numpy(scalars).to(self.device)
            h = torch.from_numpy(heuristics).to(self.device)
            af = torch.from_numpy(action_features_tensor).to(self.device)
            
            policy_logits, value = self.network(b, cp, q, s, h, af)
            
            probs = torch.softmax(policy_logits, dim=-1)[0].cpu().numpy().tolist()
            val = value[0, 0].item()
            
        return probs, val
