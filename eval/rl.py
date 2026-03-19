import pentafluoride
import model
import os
import torch
import random
import math
from collections import namedtuple, deque

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.01
EPS_DECAY = 2500
TAU = 0.005
LR = 3e-4



if __name__ == "__main__":
    model = model.TetrisNet()
    
    # Load the model weights from a file
    model_path = "tetris_model.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
    
    pieces  = [pentafluoride.Piece.L, pentafluoride.Piece.J, pentafluoride.Piece.O, pentafluoride.Piece.T, pentafluoride.Piece.S, pentafluoride.Piece.Z, pentafluoride.Piece.I]   

    