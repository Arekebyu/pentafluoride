import random

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pentafluoride as pf



class TetrisEnv(gym.Env):
    '''features are 
        - piece type (7 types)
        - piece rotation (4 rotations)
        - piece x position 0-9 (we use one-hot encoding for this because it's categorical)
        - piece y position 0-31 (likewise)
        - piece spin type (3 types: none, mini, full)'''
    action_dim =  7 + 4 + 10 + 32 + 3   # piece type + rotation + x pos + y pos + spin type

    '''state has:
        board: 32 rows x 10 columns, 1 if occupied, 0 if empty
        the 33rd row is used to encode the following features:
        - b2b status (0 to 1)
        - combo count (0 to 1)
        - max height (0 to 1)
        - total holes (0 to 1)
        - bumpiness (0 to 1)
        - aggregate height (0 to 1)
        the 34th row is used to encode the heights of the 10 columns (0 to 1)
        '''
    state_dim = (32 + 2) * 10          # 32 rows + 2 feature rows, 10 columns

    def __init__(self):
        super().__init__()
        self.pieces  = [pf.Piece.L, pf.Piece.J, pf.Piece.O, pf.Piece.T, pf.Piece.S, pf.Piece.Z, pf.Piece.I]   

        # Curriculum learning phase
        # Phase 1: Not used here directly (handled outside)
        # Phase 2: Survival and board neatness
        # Phase 3: Sparse reward
        self.phase = 2

        # 64 next piece actions + 64 hold piece actions = 128 total actions
        self.action_space = spaces.Discrete(128)
        self.observation_space = spaces.Dict({
            "state": spaces.Box(low=0, high=1, shape=(34, 10), dtype=np.float32),
            "action_features": spaces.Box(low=0, high=1, shape=(128, 7 + 4 + 10 + 32 + 3), dtype=np.float32),
            "action_mask": spaces.Box(low=0, high=1, shape=(128,), dtype=np.float32)
        })
        
    def set_phase(self, phase):
        self.phase = phase

    def step(self, action):
        if action >= len(self.placements):
            action = action % len(self.placements)
        placement = self.placements[action][0]
        active_piece = self.queue.pop(0)

        info = self.game_state.advance(active_piece, placement)
        
        if self.phase == 2:
            # Phase 2: Survival and messiness penalty
            # Features are calculated inside _get_obs, so we just use game_state and simple metrics
            # Or we can recalculate bumpiness/holes here. To be efficient, let's just 
            # approximate survival reward and penalize height.
            # We'll calculate holes and bumpiness inline
            heights = [0] * 10
            for x in range(10):
                col = self.game_state.board.cols[x]
                for y in range(32):
                    if col & (1 << y):
                        heights[x] = y + 1
            
            holes = 0
            for x in range(10):
                col = self.game_state.board.cols[x]
                for y in range(heights[x]):
                    if not (col & (1 << y)):
                        holes += 1
            bumpiness = sum((heights[i] - heights[i+1]) ** 2 for i in range(9))
            
            # Quadratic survival bonus based on pieces placed
            reward = 1.0 + (self.num_steps / 20.0) ** 2
            # reward -= 0.1 * holes
            # reward -= 0.05 * bumpiness
        else:
            # Phase 3: Sparser standard reward
            reward = pf.calculate_reward(info)
        
        if len(self.queue) < 5:
            new_bag = self.pieces.copy()
            random.shuffle(new_bag)
            self.queue.extend(new_bag)

        self.placements = pf.find_moves(self.game_state.board, self.queue[0])
        self.placements.extend(pf.find_moves(self.game_state.board, self.game_state.hold))

        terminated = False
        if self.placements is None or len(self.placements) == 0:
            terminated = True
            # penalty for loss (should be tweaked, not sure about the exact value)
            reward = -10 # theoretical maximum attack is ~ 60 with tst 20 combo with lvl 4 b2b.
        
        
        self.num_steps += 1
        truncated = False
        
        max_steps = 100 if self.phase == 2 else 500
        if self.num_steps >= max_steps:
            truncated = True

        return self._get_obs(), reward, terminated, truncated, self._get_info() 


    def reset(self, seed=None, options=None):
        # Reset the state of the environment to an initial state
        super().reset(seed=seed)
        random.seed(seed)
        self.queue = self.pieces.copy()
        random.shuffle(self.queue)
        self.num_steps = 0

        self.game_state = pf.GameState(
            board=[0 for _ in range(10)],
            hold=self.queue.pop(0),
            combo=0,
            b2b=0,
        )
        if len(self.queue) < 5:
            new_bag = self.pieces.copy()
            random.shuffle(new_bag)
            self.queue.extend(new_bag)

        self.placements = pf.find_moves(self.game_state.board, self.queue[0]) + pf.find_moves(self.game_state.board, self.game_state.hold)
        return self._get_obs(), self._get_info()

    def render(self, mode='human'):
        # todo!
        pass

    def _get_obs(self):
        board = np.zeros((34, 10), dtype=np.float32)
        heights = [0] * 10
        for x in range(10):
            col = self.game_state.board.cols[x]
            for y in range(32):
                if col & (1 << y):
                    board[y][x] = 1
                    heights[x] = y + 1

        # encode b2b and combo in the 33rd row
        board[32][0] = min(self.game_state.b2b / 64, 1)
        board[32][1] = min(self.game_state.combo / 21, 1)
        
        max_height = max(heights)
        board[32][2] = max_height / 32.0
        
        holes = 0
        for x in range(10):
            col = self.game_state.board.cols[x]
            for y in range(heights[x]):
                if not (col & (1 << y)):
                    holes += 1
        board[32][3] = min(holes / 40.0, 1.0)
        
        bumpiness = 0
        for x in range(9):
            bumpiness += abs(heights[x] - heights[x+1])
        board[32][4] = min(bumpiness / 40.0, 1.0)

        aggregate_height = sum(heights)
        board[32][5] = min(aggregate_height / 320.0, 1.0)
        
        # 34th row for explicit column heights
        for x in range(10):
            board[33][x] = heights[x] / 32.0

        action_features = []
        placements = self.placements
        for p_tuple in placements:
            p = p_tuple[0]
            piece_type = [0] * 7
            piece_type[p.location.piece.value] = 1
            rotation = [0] * 4
            rotation[p.location.rotation.value] = 1
            x_pos = [0] * 10
            x_pos[p.location.x] = 1
            y_pos = [0] * 32
            y_idx = max(0, min(p.location.y, 31))
            y_pos[y_idx] = 1
            spin_type = [0] * 3
            spin_type[p.spin.value] = 1
            
            action_features.append(piece_type + rotation + x_pos + y_pos + spin_type)
        
        # pad action features with zeros if there are less than 128 possible actions
        num_valid_actions = len(action_features)
        action_mask = [1.0] * num_valid_actions + [0.0] * (128 - num_valid_actions)
        
        while len(action_features) < 128:
            action_features.append([0] * (7 + 4 + 10 + 32 + 3))
        
        return {
            "state": board,
            "action_features": np.array(action_features, dtype=np.float32),
            "action_mask": np.array(action_mask, dtype=np.float32)
        }

    def _get_info(self):
        return {}

if __name__ == '__main__':
    env = TetrisEnv()
    obs, info = env.reset()
    env.render()
    truncated = False
    terminated = False
    while not (truncated or terminated):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
