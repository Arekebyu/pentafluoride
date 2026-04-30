import numpy as np
import torch
import random
from collections import deque

class HindsightReplayBuffer:
    def __init__(self, capacity, state_shape, action_feature_shape, n_actions=128):
        self.capacity = capacity
        
        # Main persistent buffer
        self.states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.action_features = np.zeros((capacity, *action_feature_shape), dtype=np.float32)
        self.action_masks = np.zeros((capacity, n_actions), dtype=np.float32)
        self.actions = np.zeros((capacity, 1), dtype=np.int64)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.next_action_features = np.zeros((capacity, *action_feature_shape), dtype=np.float32)
        self.next_action_masks = np.zeros((capacity, n_actions), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        
        self.ptr = 0
        self.size = 0
        
        # Temporary rolling window for Hindsight Replay
        self.window_size = 10
        self.trigger_window = 5
        self.app_threshold = 6.0  # Average Attack Per Piece > 6
        self.random_commit_prob = 0
        self.temp_buffer = deque(maxlen=self.window_size)
        self.recently_committed = 0 # Prevent committing the same spike multiple times

    def add_to_window(self, state, action_feats, action_mask, action, reward, next_state, next_action_feats, next_action_mask, done):
        """Adds a transition to the rolling window and checks for APP spike."""
        transition = (state, action_feats, action_mask, action, reward, next_state, next_action_feats, next_action_mask, done)
        self.temp_buffer.append(transition)
        
        if self.recently_committed > 0:
            self.recently_committed -= 1

        # Check for spike in the last `trigger_window` pieces
        if len(self.temp_buffer) >= self.trigger_window and self.recently_committed == 0:
            recent_rewards = [t[4] for t in list(self.temp_buffer)[-self.trigger_window:]]
            # Since reward = attack + 0.1, we subtract 0.1 to get true attack
            avg_attack = sum([r - 0.1 for r in recent_rewards]) / self.trigger_window
            
            if avg_attack > self.app_threshold or random.random() < self.random_commit_prob:
                self._commit_window()
                self.recently_committed = self.window_size # Wait before committing overlapping windows

        # If episode ends, clear the window so we don't bleed into the next episode
        if done:
            self.temp_buffer.clear()
            self.recently_committed = 0

    def _commit_window(self):
        """Commits the entire temporary rolling window into the persistent replay buffer."""
        for transition in self.temp_buffer:
            self._add(*transition)
        # We don't clear the temp buffer here because we might want the end of this spike
        # to serve as the 'prior' context for a subsequent consecutive spike, but we use
        # recently_committed to prevent exact duplicate commits.

    def _add(self, state, action_feats, action_mask, action, reward, next_state, next_action_feats, next_action_mask, done):
        """Internal method to add to the persistent buffer."""
        self.states[self.ptr] = state
        self.action_features[self.ptr] = action_feats
        self.action_masks[self.ptr] = action_mask
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.next_action_features[self.ptr] = next_action_feats
        self.next_action_masks[self.ptr] = next_action_mask
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, device="cpu"):
        """Samples a random batch from the persistent buffer."""
        ind = np.random.randint(0, self.size, size=batch_size)
        
        return (
            torch.FloatTensor(self.states[ind]).to(device),
            torch.FloatTensor(self.action_features[ind]).to(device),
            torch.FloatTensor(self.action_masks[ind]).to(device),
            torch.LongTensor(self.actions[ind]).to(device),
            torch.FloatTensor(self.rewards[ind]).to(device),
            torch.FloatTensor(self.next_states[ind]).to(device),
            torch.FloatTensor(self.next_action_features[ind]).to(device),
            torch.FloatTensor(self.next_action_masks[ind]).to(device),
            torch.FloatTensor(self.dones[ind]).to(device)
        )

    def __len__(self):
        return self.size
