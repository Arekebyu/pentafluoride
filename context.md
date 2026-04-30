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
- **AlphaGo-style MCTS Pipeline (`mcts_alphago/`):** The new primary training architecture utilizing Monte Carlo DAG Search guided by a `DualHeadNet` (Value and Policy). The network receives complex state features including heuristic values, the current board, and a 42-dimensional queue vector to reason about uncertainty via the "7-bag" system.
- **Legacy DQN Training Loop (`ppo/train_dqn.py`):** [DEPRECATED] An off-policy custom training loop using Hindsight Experience Replay.
- **Legacy PPO Implementations:** [DEPRECATED] `ppo/rl_sb3.py` and `ppo/custom_ppo.py` are older iterations that used on-policy PPO algorithms via Stable Baselines 3.

## Getting Started
- Build the Rust extension: Use `maturin develop` to compile the `pentafluoride` crate and install it in the current Python environment.
- Dependencies: Requires `PyTorch`, `Gymnasium`, `PyO3`, and the compiled `pentafluoride` extension.
- Training is primarily orchestrated by executing `python mcts_alphago/train.py`.
- Visualization can be run via `python mcts_alphago/visualize.py`.
## Current Focus & Recent Work
1. **AlphaGo MCTS Transition:** Replacing epsilon-greedy DQN with a Monte Carlo DAG search guided by a dual-headed Value and Policy network.
2. **Curriculum Learning:** Developing a dynamic `phase` mechanism within the environment to shift from dense heuristic pre-training to sparse advanced reward learning.
3. **Off-policy Q-Learning:** [DEPRECATED] Using deep Q-networks (DQN) paired with a specialized Hindsight Replay Buffer.
3. **Rust Interoperability:** Addressing type conversions, lifetime management, and environment stability when passing large state tensors between Python (PyTorch) and the Rust backend.
4. **Performance:** Moving performance-critical paths (MCTS simulation, move legality checks) out of Python and heavily optimizing the Rust counterparts.

# Current Issues
1. Current training fails to converge to a good policy and instead performance decays over time, with the learned policy being worse than random play.