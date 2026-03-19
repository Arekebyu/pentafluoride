from random import random

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pentafluoride as pf

class TetrisEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.pieces  = [pf.Piece.L, pf.Piece.J, pf.Piece.O, pf.Piece.T, pf.Piece.S, pf.Piece.Z, pf.Piece.I]   
        self.game_state = pf.GameState()

        # 64 next piece actions + 64 hold piece actions = 128 total actions
        self.action_space = spaces.Discrete(128)
        self.observation_space = spaces.Dict({
            '''state is currently only the board, can add more features later'''
            "state": spaces.Box(low=0, high=1, shape=(32, 10), dtype=np.uint8),
            '''features are 
            - piece type (7 types)
            - piece rotation (4 rotations)
            - piece x position 0-9 (we use one-hot encoding for this because it's categorical)
            - piece y position 0-31 (likewise)
            - piece spin type (3 types: none, mini, full)'''
            "action_features": spaces.Box(low=0, high=1, shape=(128, 7 + 4 + 10 + 32 + 3), dtype=np.float32)
        })
        
    def step(self, action):
        placement = self.observation_space["action_features"][action]
        # scale reward to be between -1 and 1 
        max_reward = 0
        for p in self.observation_space["action_features"]:
            gs = self.game_state.copy()
            info = gs.advance(p)
            reward = pf.calculate_reward(info)
            if reward > max_reward:
                max_reward = reward
        info = self.game_state.advance(placement)
        reward = pf.calculate_reward(info) - max_reward
        self.placements = pf.find_moves(self.game_state.board, self.queue[0]) + pf.find_moves(self.game_state.board, self.game_state.hold)
        terminated = False
        if len(self.placements) == 0:
            terminated = True
            # penalty for loss (should be tweaked, not sure about the exact value)
            reward = -60 # theoretical maximum attack is ~ 60 with tst 20 combo with lvl 4 b2b.
        
        truncated = False
        if self.num_episodes >= 100:
            truncated = True
        
        if self.queue.empty():
            self.queue = self.pieces.copy()
            random.shuffle(self.queue)

        return self._get_obs(), reward, terminated, truncated, self._get_info() 


    def reset(self, seed=None, options=None):
        # Reset the state of the environment to an initial state
        super().reset(seed=seed)
        random.seed(seed)
        self.queue = self.pieces.copy()
        random.shuffle(self.queue)
        self.num_episodes = 0

        self.game_state = pf.GameState(
            board=[0 for _ in range(10)],
            hold=self.queue.pop(0),
            combo=0,
            b2b=0,
        )
        self.placements = pf.find_moves(self.game_state.board, self.queue[0]) + pf.find_moves(self.game_state.board, self.game_state.hold)
        return self._get_obs(), self._get_info()

    def render(self, mode='human'):
        # todo!
        pass

    def _get_obs(self):
        board = np.zeros((32, 10), dtype=np.uint8)
        for x in range(10):
            for y in range(32):
                if self.game_state.board[x] & (1 << y):
                    board[y][x] = 1
        action_features = []
        placements = self.placements
        for p in placements:
            piece_type = [0] * 7
            piece_type[p.location.piece.value] = 1
            rotation = [0] * 4
            rotation[p.location.rotation] = 1
            x_pos = [0] * 10
            x_pos[p.location.x] = 1
            y_pos = [0] * 32
            y_pos[p.location.y] = 1
            spin_type = [0] * 3
            if p.spin == pf.SpinType.NONE:
                spin_type[0] = 1
            elif p.spin == pf.SpinType.MINI:
                spin_type[1] = 1
            elif p.spin == pf.SpinType.FULL:
                spin_type[2] = 1
            
            action_features.append(piece_type + rotation + x_pos + y_pos + spin_type)
        
        # pad action features with zeros if there are less than 128 possible actions
        while len(action_features) < 128:
            action_features.append([0] * (7 + 4 + 10 + 32 + 3))
        
        return {
            "state": board,
            "action_features": np.array(action_features, dtype=np.float32)
        }

    def _get_info(self):
        return {}

if __name__ == '__main__':
    env = TetrisEnv()
    obs, info = env.reset()
    env.render()
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
