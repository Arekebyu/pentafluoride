use crate::data::*;
use crate::movegen::find_moves;
use pyo3::prelude::*;
use rand::seq::IndexedRandom;
use std::collections::HashMap;
use std::vec;

pub struct MCTS {
    nodes: Vec<Node>,
    evaluator: Option<Py<PyAny>>,
}
#[derive(Clone)]
pub struct Node {
    parent: Option<usize>,
    children: HashMap<Placement, (f32, usize)>, // placement -> (policy, index)
    unexpanded_actions: Vec<(Placement, u32)>,
    queue: Vec<Piece>,

    visits: usize,
    total_score: f32,
    max_reward: f32,

    state: GameState,

    // for regularizing rewards
    min_score: f32,
    max_score: f32,
    is_terminal: bool,
}

#[pyfunction]
pub fn ppo_mcts_search(
    py: Python<'_>,
    root: GameState,
    queue: Vec<Piece>,
    iteration: usize,
    evaluator: Py<PyAny>,
) -> Placement {
    let mut tree = MCTS::new(root, queue, Some(evaluator));

    for _ in 0..iteration {
        let mut node_idx = 0;
        let mut path = vec![node_idx];

        // 1. Selection: Traverse down the tree using the UCB1 policy
        loop {
            let node = &tree.nodes[node_idx];
            if node.is_terminal {
                break;
            }
            if node.unexpanded_actions.is_empty() && !node.children.is_empty() {
                if let Some(next_idx) = tree.select(node_idx) {
                    node_idx = next_idx;
                    path.push(node_idx);
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        // 2. Expansion & 3. Simulation: Create a new node and get its initial reward
        let reward = if let Some((new_node_idx, r)) = tree.expand(py, node_idx) {
            path.push(new_node_idx);
            r
        } else {
            // If terminal, use the current node's value
            tree.nodes[node_idx].value()
        };

        // 4. Backpropagation: Update statistics for all nodes in the path
        for &idx in path.iter().rev() {
            tree.update(reward, idx);
        }
    }

    // Choose the best move based on the most visited child of the root
    *tree.nodes[0]
        .children
        .iter()
        .max_by_key(|&(_, &idx)| tree.nodes[idx.1].visits)
        .map(|(k, _)| k)
        .expect("MCTS failed to find any valid moves")
}

fn rollout(mut state: GameState, mut queue: Vec<Piece>) -> f32 {
    let mut total_reward = 0.0;
    while !queue.is_empty() {
        let piece = queue.remove(0);
        let mut moves = find_moves(&state.board, piece);
        moves.extend(find_moves(&state.board, state.hold));
        if moves.is_empty() {
            break;
        }
        let (m, _) = *moves.choose(&mut rand::rng()).unwrap();
        let info = state.advance(piece, m);
        total_reward += calculate_reward(&info);
    }
    total_reward
}

#[pyfunction]
pub fn ppo_mcts_generate_targets(
    py: Python<'_>,
    root: GameState,
    queue: Vec<Piece>,
    iteration: usize,
) -> Vec<(Placement, f32)> {
    let mut tree = MCTS::new(root, queue, None);

    for _ in 0..iteration {
        let mut node_idx = 0;
        let mut path = vec![node_idx];

        loop {
            let node = &tree.nodes[node_idx];
            if node.is_terminal {
                break;
            }
            if node.unexpanded_actions.is_empty() && !node.children.is_empty() {
                if let Some(next_idx) = tree.select(node_idx) {
                    node_idx = next_idx;
                    path.push(node_idx);
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        let reward = if let Some((new_node_idx, r)) = tree.expand(py, node_idx) {
            path.push(new_node_idx);
            r
        } else {
            tree.nodes[node_idx].value()
        };

        for &idx in path.iter().rev() {
            tree.update(reward, idx);
        }
    }

    tree.nodes[0]
        .children
        .iter()
        .map(|(&p, &(_, idx))| (p, tree.nodes[idx].value()))
        .collect()
}


impl MCTS {
    pub fn new(root: GameState, queue: Vec<Piece>, evaluator: Option<Py<PyAny>>) -> Self {
        // we make some assumptions here, namely, root gamestate already has hold piece
        let mut root = Node::new(root, queue);
        use rand::seq::SliceRandom;
        root.unexpanded_actions.shuffle(&mut rand::rng());
        Self {
            nodes: vec![root],
            evaluator,
        }
    }

    fn select(&self, node_id: usize) -> Option<usize> {
        const C_PUCT: f32 = 1.0;
        let cur = self
            .nodes
            .get(node_id)
            .expect("reference to nonexistent node in arena in select");
        let denom = cur.max_score - cur.min_score;
        let denom = if denom.abs() < f32::EPSILON {
            1.0
        } else {
            denom
        };

        let mut best = f32::MIN;
        let mut node = None;

        for (&_placement, &(prior_prob, child_id)) in &cur.children {
            let child = self
                .nodes
                .get(child_id)
                .expect("reference to nonexistent node in arena in select");

            let q_value = (child.value() - cur.min_score) / denom;
            let u_value =
                C_PUCT * prior_prob * (cur.visits as f32).sqrt() / (1.0 + child.visits as f32);
            let score = q_value + u_value;
            if score > best {
                node = Some(child_id);
                best = score;
            }
        }

        return node;
    }

    fn expand(&mut self, py: Python<'_>, node_id: usize) -> Option<(usize, f32)> {
        if self.nodes[node_id].is_terminal || self.nodes[node_id].unexpanded_actions.is_empty() {
            return None;
        }

        let (placement, _) = self.nodes[node_id].unexpanded_actions.pop()?;

        let mut new_state = self.nodes[node_id].state;
        let piece = *self.nodes[node_id].queue.first().unwrap_or(&Piece::O);
        let _info = new_state.advance(piece, placement);
        self.nodes[node_id].max_reward =
            self.nodes[node_id].max_reward.max(calculate_reward(&_info));

        let next_queue = if self.nodes[node_id].queue.is_empty() {
            vec![]
        } else {
            self.nodes[node_id].queue[1..].to_vec()
        };

        let mut is_terminal = next_queue.is_empty();

        let (policy_score, quality_score) = if let Some(evaluator) = &self.evaluator {
            let eval_result = evaluator
                .call1(py, (new_state, next_queue.clone()))
                .ok()?;
            eval_result.extract(py).ok()?
        } else {
            // Pure rust rollout for generating targets
            let rollout_reward = rollout(new_state, next_queue.clone());
            (1.0, rollout_reward) // prior uniform
        };

        let child_index = self.nodes.len();
        let mut child = Node::new(new_state, next_queue);
        use rand::seq::SliceRandom;
        child.unexpanded_actions.shuffle(&mut rand::rng());
        child.parent = Some(node_id);
        if child.unexpanded_actions.is_empty() {
            is_terminal = true;
        }
        child.is_terminal = is_terminal;

        self.nodes[node_id]
            .children // The key for children is `Placement`, not a tuple.
            .insert(placement, (policy_score, child_index));
        self.nodes.push(child);
        return Some((child_index, quality_score));
    }
    fn update(&mut self, reward: f32, idx: usize) {
        let child = self.nodes.get_mut(idx).expect("bad child index in update");
        child.visits += 1;
        child.total_score += reward;
        let child_value = child.value();
        if let Some(parent) = self.nodes.get(idx).expect("bad index in update").parent {
            let parent = self
                .nodes
                .get_mut(parent)
                .expect("bad parent index in update");
            parent.min_score = parent.min_score.min(child_value);
            parent.max_score = parent.max_score.max(child_value);
        }
    }
}

impl Node {
    pub fn new(state: GameState, queue: Vec<Piece>) -> Self {
        let unexpanded_actions = {
            if let Some(piece) = queue.first() {
                find_moves(&state.board, *piece)
            } else {
                vec![]
            }
        }.into_iter().chain(find_moves(&state.board, state.hold)).collect::<Vec<_>>();

        let is_terminal = queue.is_empty() || unexpanded_actions.is_empty();

        Node {
            parent: None,
            children: HashMap::new(),
            unexpanded_actions,
            queue,

            visits: 0,
            total_score: 0.0,
            max_reward: 0.0,

            state,

            min_score: f32::INFINITY,
            max_score: f32::NEG_INFINITY,
            is_terminal,
        }
    }
    fn value(&self) -> f32 {
        match self.visits {
            0 => 0.0,
            visits => self.total_score / visits as f32,
        }
    }
}
