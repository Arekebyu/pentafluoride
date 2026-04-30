import random

class MCTSReplayBuffer:
    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.buffer = []
        self.idx = 0

    def add(self, state_features, action_features, policy_targets, value_target):
        """
        state_features: Tuple of (board, current_piece, queue_tensor, scalars, heuristics)
        action_features: [N, 56] array of valid move features
        policy_targets: [N] array of probabilities from MCTS
        value_target: Scalar value (e.g., actual game return)
        """
        data = (state_features, action_features, policy_targets, value_target)
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.idx] = data
        self.idx = (self.idx + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
