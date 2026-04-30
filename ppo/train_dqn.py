import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
import time

from gymnasium_env.envs import TetrisEnv
from custom_net import TetrisValueNetwork
from replay_buffer import HindsightReplayBuffer

def make_tensor(x, device):
    return torch.FloatTensor(x).unsqueeze(0).to(device)

def pretrain_regression(q_net, optimizer, env, device, epochs=50, mcts_iterations=1000):
    import pentafluoride as pf
    print("--- Starting Phase 1: Pre-training (Rule Learning via Regression) ---")
    loss_fn = nn.MSELoss()
    q_net.train()
    
    for epoch in range(epochs):
        obs, _ = env.reset()
        epoch_loss = 0
        steps = 3 # number of bags to play
        
        for step in range(steps):
            state = obs["state"]
            action_feats = obs["action_features"]
            action_mask = obs["action_mask"]
            # Ensure we have a 7-piece queue for the MCTS lookahead
            queue_7 = env.queue.copy()
            while len(queue_7) < 7:
                import random
                new_bag = env.pieces.copy()
                random.shuffle(new_bag)
                queue_7.extend(new_bag)
            queue_7 = queue_7[:7]
            
            # Generate regression targets (lookahead 7 pieces)
            targets = pf.mcts_generate_targets(env.game_state, queue_7, mcts_iterations)
            
            target_q = torch.zeros(128).to(device)
            valid_indices = []
            
            best_action_idx = 0
            best_reward = -float('inf')
            
            for placement, reward in targets:
                for i, p_tuple in enumerate(env.placements):
                    if (p_tuple[0].location.x == placement.location.x and 
                        p_tuple[0].location.y == placement.location.y and 
                        p_tuple[0].location.rotation == placement.location.rotation and 
                        p_tuple[0].location.piece == placement.location.piece):
                        
                        target_q[i] = reward
                        valid_indices.append(i)
                        
                        if reward > best_reward:
                            best_reward = reward
                            best_action_idx = i
                        break
            
            if len(valid_indices) == 0:
                break
                
            q_values = q_net(
                make_tensor(state, device),
                make_tensor(action_feats, device),
                make_tensor(action_mask, device)
            ).squeeze(0)
            
            loss = loss_fn(q_values[valid_indices], target_q[valid_indices])
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Advance environment
            obs, _, terminated, truncated, _ = env.step(best_action_idx)
            if terminated or truncated:
                break
                
        if (epoch + 1) % 10 == 0:
            print(f"Pre-training Epoch [{epoch+1}/{epochs}], Avg Loss: {epoch_loss/max(1, step+1):.4f}, Survived: {step+1} steps")

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Hyperparameters
    total_timesteps = 1_000_000
    batch_size = 64
    gamma = 0.999
    lr = 1e-3
    target_update_freq = 1000
    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_decay = 100_000
    learning_starts = 1000

    env = TetrisEnv()
    
    state_dim = 340
    action_feature_dim = 56
    
    q_net = TetrisValueNetwork(state_dim, action_feature_dim).to(device)
    target_net = TetrisValueNetwork(state_dim, action_feature_dim).to(device)
    target_net.load_state_dict(q_net.state_dict())
    
    optimizer = optim.Adam(q_net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    replay_buffer = HindsightReplayBuffer(
        capacity=100_000, 
        state_shape=(34, 10), 
        action_feature_shape=(128, 56)
    )

    # --- Phase 1: Pre-training ---
    pretrain_regression(q_net, optimizer, env, device, epochs=0, mcts_iterations=2000)
    
    # Save after pretraining
    torch.save(q_net.state_dict(), "./checkpoints/tetris_qnet_pretrained.pth")
    print("Pre-training complete. Saved to ./checkpoints/tetris_qnet_pretrained.pth")
    target_net.load_state_dict(q_net.state_dict())

    obs, _ = env.reset()
    episode_reward = 0
    episodes = 0
    
    # Track survival to transition phases
    recent_survivals = []
    
    # Start in Phase 2
    env.set_phase(2)
    print("--- Starting Phase 2: Survival Training (Dense Rewards) ---")

    os.makedirs("./checkpoints", exist_ok=True)

    for step in range(total_timesteps):
        state = obs["state"]
        action_feats = obs["action_features"]
        action_mask = obs["action_mask"]

        # Epsilon-greedy action selection
        epsilon = max(epsilon_end, epsilon_start - step / epsilon_decay)
        
        valid_actions = np.where(action_mask == 1.0)[0]
        if len(valid_actions) == 0:
            action = 0 # Fallback
        elif random.random() < epsilon:
            action = random.choice(valid_actions)
        else:
            with torch.no_grad():
                q_values = q_net(
                    make_tensor(state, device),
                    make_tensor(action_feats, device),
                    make_tensor(action_mask, device)
                )
                action = q_values.argmax(dim=1).item()

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Add to rolling window for hindsight replay evaluation
        replay_buffer.add_to_window(
            state, action_feats, action_mask, action, reward,
            next_obs["state"], next_obs["action_features"], next_obs["action_mask"], float(done)
        )

        obs = next_obs
        episode_reward += reward

        if done:
            episodes += 1
            recent_survivals.append(env.num_steps)
            if len(recent_survivals) > 50:
                recent_survivals.pop(0)
            
            if episodes % 10 == 0:
                avg_survival = sum(recent_survivals) / len(recent_survivals)
                print(f"Phase {env.phase} | Step: {step}, Episode: {episodes}, Reward: {episode_reward:.2f}, Avg Survival: {avg_survival:.1f}, Buffer: {replay_buffer.size}, Eps: {epsilon:.2f}")
            
            # Curriculum transition logic
            if env.phase == 2 and episodes >= 100:
                avg_survival = sum(recent_survivals) / len(recent_survivals)
                if avg_survival >= 97: 
                    env.set_phase(3)
                    print("\n>>> Transitioning to Phase 3: Sparse Rewards! <<<\n")
                    # Optionally reset epsilon or learning rate
                    epsilon_start = 0.5
                    epsilon_decay = 200_000
            
            obs, _ = env.reset()
            episode_reward = 0

        # Training
        if replay_buffer.size > batch_size and step > learning_starts:
            b_states, b_action_feats, b_action_masks, b_actions, b_rewards, b_next_states, b_next_action_feats, b_next_action_masks, b_dones = replay_buffer.sample(batch_size, device)

            # Current Q Values
            q_values = q_net(b_states, b_action_feats, b_action_masks)
            q_values = q_values.gather(1, b_actions)

            # Target Q Values
            with torch.no_grad():
                next_q_values = target_net(b_next_states, b_next_action_feats, b_next_action_masks)
                max_next_q_values = next_q_values.max(dim=1, keepdim=True)[0]
                target_q_values = b_rewards + gamma * max_next_q_values * (1.0 - b_dones)

            loss = loss_fn(q_values, target_q_values)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update target network
            if step % target_update_freq == 0:
                target_net.load_state_dict(q_net.state_dict())

        # Save Checkpoint
        if step > 0 and step % 50_000 == 0:
            torch.save(q_net.state_dict(), f"./checkpoints/tetris_qnet_step_{step}.pth")
            print(f"Saved checkpoint to ./checkpoints/tetris_qnet_step_{step}.pth")
            
        # LR Decay
        if step > 0 and step % 100_000 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5
            print(f"Decayed learning rate to {optimizer.param_groups[0]['lr']}")

if __name__ == "__main__":
    train()
