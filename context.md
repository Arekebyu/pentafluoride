# Pentafluoride Project Context

## Overview
`pentafluoride` is a hybrid Rust/Python reinforcement learning project aimed at training a high-performance Tetris agent. The project leverages Rust for speed in core game logic and tree search algorithms, while using Python and PyTorch for flexible, deep reinforcement learning architectures. 

The ultimate goal of the project is to develop an agent capable of achieving high Average Attack Per Piece (APP) and discovering advanced gameplay strategies through self-play and curriculum learning.

## Architecture

The system is split into two primary domains, bridging high-performance systems programming with machine learning research:

### 1. High-Performance Core (Rust)
Located in `/src/`, the Rust backend is compiled into a shared library via `PyO3` and exposed to Python as the `pentafluoride` module.
- **`data.rs` :** Implements the core Tetris mechanics, board representation.
- **`movegen.rs` :** Implements the move generation logic for a given board and piece.
- **`mcts.rs`:** A Monte Carlo Tree Search algorithm used to evaluate board states and generate synthetic high-quality training data.
- **`lib.rs`:** Defines the PyO3 bindings, exposing necessary structs (like `Board` and `Piece`) and functions directly to Python.

### 2. Reinforcement Learning Pipeline (Python)
Located in `/ppo/`, the training loop integrates with the Rust core to learn optimal policies.
- **Gymnasium Environment (`gymnasium_env/envs/tetris.py`):** A custom OpenAI Gymnasium environment that wraps the Rust engine, managing episode state, observations, and reward calculation.
- **Custom Value Networks (`custom_net.py`):** PyTorch-based neural networks (e.g., `TetrisValueNetwork`) that ingest board state features and predict raw Q-values for all possible move placements (up to 128 possible moves).
- **Hindsight Experience Replay (`replay_buffer.py`):** A rolling-window replay buffer designed to implement Hindsight Experience Replay (HER). It commits high-performance gameplay segments to memory (e.g., sequences with APP > 6) for self-imitation learning.
- **DQN Training Loop (`train_dqn.py`):** An off-policy custom training loop replacing older on-policy architectures (PPO via Stable Baselines 3). This pipeline implements a **Curriculum Learning** strategy:
  - **Phase 1:** MCTS-based pre-training to learn game rules via supervised regression.
  - **Phase 2:** Reinforcement learning with dense heuristic rewards to encourage survival duration.
  - **Phase 3:** Transition to sparse, raw environment rewards to discover advanced gameplay strategies.
- **Legacy Implementations:** `rl_sb3.py` and `custom_ppo.py` are older iterations that used on-policy PPO algorithms via Stable Baselines 3.

## Getting Started
- Build the Rust extension: Use `maturin develop` to compile the `pentafluoride` crate and install it in the current Python environment.
- Dependencies: Requires `PyTorch`, `Gymnasium`, `PyO3`, and the compiled `pentafluoride` extension.
- Training is primarily orchestrated by executing `python ppo/train_dqn.py`.

## Current Focus & Recent Work
1. **Curriculum Learning:** Developing a dynamic `phase` mechanism within the environment to shift from dense heuristic pre-training to sparse advanced reward learning.
2. **Off-policy Q-Learning:** Using deep Q-networks (DQN) paired with a specialized Hindsight Replay Buffer to focus on episodes yielding high Attack Per Piece values.
3. **Rust Interoperability:** Addressing type conversions, lifetime management, and environment stability when passing large state tensors between Python (PyTorch) and the Rust backend.
4. **Performance:** Moving performance-critical paths (MCTS simulation, move legality checks) out of Python and heavily optimizing the Rust counterparts.

# Current Issues
1. Current training fails to converge to a good policy and instead performance decays over time, with the learned policy being worse than random play.