import gymnasium as gym
import numpy as np
import os
import sys
from custom_ppo import TwoTowerPolicy

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback

from gymnasium_env.envs import TetrisEnv

if __name__ == "__main__":
    MODEL_NAME = "ppo_tetris_v1"
    LOG_DIR = "./ppo_tetris_logs/"
    CHECKPOINT_DIR = "./checkpoints/"
    TOTAL_TIMESTEPS = 1_000_000 
    SAVE_FREQ = 10000 
    LEARNING_RATE = 3e-4
    GAMMA = 0.99
    N_STEPS = 2048 # Number of steps to run for each environment per update
    BATCH_SIZE = 64
    N_EPOCHS = 10
    ENT_COEF = 0.01 

    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Create the environment
    # stable-baselines3 requires vectorized environments.
    # DummyVecEnv is suitable for single-environment, non-parallel execution.
    # VecMonitor adds episode statistics to the TensorBoard logs.
    vec_env = make_vec_env(TetrisEnv, n_envs=1, seed=0)
    vec_env = VecMonitor(vec_env, LOG_DIR)

    # Saves checkpoint every SAVE_FREQ steps
    checkpoint_callback = CheckpointCallback(
        save_freq=SAVE_FREQ,
        save_path=CHECKPOINT_DIR,
        name_prefix=MODEL_NAME,
        save_replay_buffer=True,
        save_vecnormalize=True,  
    )

    # Initialize the PPO model
    # For a Box observation space (33,10), PPO will use an MlpPolicy by default,
    # flattening the input. This is a reasonable starting point.
    model = PPO(
        policy=TwoTowerPolicy,  # Use our custom Two-Tower policy
        env=vec_env,
        verbose=1,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        gamma=GAMMA,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=ENT_COEF,
        learning_rate=LEARNING_RATE,
        tensorboard_log=LOG_DIR
    )

    # Train the model
    print(f"Starting training for {TOTAL_TIMESTEPS} timesteps...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=checkpoint_callback)
    print("Training finished.")

    # Save the final model
    model.save(f"{MODEL_NAME}.zip")
    print(f"Final model saved as {MODEL_NAME}.zip")

    # --- Test the trained agent (optional) ---
    print("\nTesting trained agent...")
    obs = vec_env.reset()
    for episode in range(5): # Run for 5 test episodes
        done = False
        total_reward = 0
        steps = 0
        while not done:
            action, _states = model.predict(obs, deterministic=True) # Use deterministic actions for testing
            obs, rewards, dones, info = vec_env.step(action)
            total_reward += rewards[0] # rewards is an array for vectorized envs
            steps += 1
            # env.render() # Uncomment if you implement render in TetrisEnv
            if dones:
                print(f"Test Episode {episode+1}: finished after {steps} steps with reward {total_reward:.2f}")
                break
    vec_env.close()