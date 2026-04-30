use crate::data::*;
use crate::movegen::find_moves;
use pyo3::prelude::*;
use std::collections::HashMap;

pub struct MCTS {
    nodes: Vec<Node>,
    evaluator: Py<PyAny>,
}

#[derive(Clone)]
pub struct Node {
    parent: Option<usize>,
    children: HashMap<Placement, usize>, // placement -> index
    
    // Valid moves and their prior probabilities from the network
    valid_moves: Vec<(Placement, u32)>,
    priors: HashMap<Placement, f32>,
    
    queue: Vec<Piece>,
    visits: usize,
    total_value: f32,
    state: GameState,

    is_terminal: bool,
}

#[pyfunction]
pub fn alphago_mcts_search(
    py: Python<'_>,
    root: GameState,
    queue: Vec<Piece>,
    iteration: usize,
    evaluator: Py<PyAny>,
    temperature: f32,
) -> PyResult<(Placement, HashMap<Placement, f32>)> {
    let mut tree = MCTS::new(py, root, queue, evaluator.clone_ref(py))?;

    for _ in 0..iteration {
        let mut node_idx = 0;
        let mut path = vec![node_idx];

        // 1. Selection
        loop {
            let node = &tree.nodes[node_idx];
            if node.is_terminal {
                break;
            }
            
            // If we have valid moves but not all of them are expanded, we should expand one.
            // Wait, in AlphaZero, we can add all children at once or lazily.
            // Usually, selection goes down until it finds a node that hasn't been expanded.
            if node.children.len() < node.valid_moves.len() {
                break; 
            } else if !node.children.is_empty() {
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

        // 2. Expansion & 3. Evaluation
        let value = if !tree.nodes[node_idx].is_terminal {
            if let Some((new_node_idx, v)) = tree.expand(py, node_idx)? {
                path.push(new_node_idx);
                v
            } else {
                tree.nodes[node_idx].value()
            }
        } else {
            // Terminal node
            tree.nodes[node_idx].value()
        };

        // 4. Backpropagation
        for &idx in path.iter().rev() {
            tree.update(value, idx);
        }
    }

    // Get the policy from visits
    let root_node = &tree.nodes[0];
    let mut policy = HashMap::new();
    let mut total_visits = 0;
    
    for (&placement, &child_idx) in &root_node.children {
        let child_visits = tree.nodes[child_idx].visits;
        total_visits += child_visits;
        policy.insert(placement, child_visits as f32);
    }

    if total_visits > 0 {
        for val in policy.values_mut() {
            if temperature == 0.0 {
                // Not supported here perfectly, but handled outside usually.
                *val /= total_visits as f32;
            } else {
                // Apply temperature
                *val = (*val).powf(1.0 / temperature);
            }
        }
        // re-normalize
        let sum: f32 = policy.values().sum();
        if sum > 0.0 {
            for val in policy.values_mut() {
                *val /= sum;
            }
        }
    }

    // Best move
    let best_move = *root_node
        .children
        .iter()
        .max_by_key(|&(_, &idx)| tree.nodes[idx].visits)
        .map(|(k, _)| k)
        .expect("MCTS failed to find any valid moves");

    Ok((best_move, policy))
}


impl MCTS {
    pub fn new(py: Python<'_>, state: GameState, queue: Vec<Piece>, evaluator: Py<PyAny>) -> PyResult<Self> {
        let mut root = Node::new(state, queue);
        
        // Evaluate root
        if !root.is_terminal {
            let placements: Vec<Placement> = root.valid_moves.iter().map(|(p, _)| *p).collect();
            let eval_result = evaluator.call1(py, (state, root.queue.clone(), placements.clone()))?;
            let (priors, value): (Vec<f32>, f32) = eval_result.extract(py)?;
            
            for (p, prob) in placements.into_iter().zip(priors.into_iter()) {
                root.priors.insert(p, prob);
            }
            root.total_value = value;
            root.visits = 1;
        }

        Ok(Self {
            nodes: vec![root],
            evaluator,
        })
    }

    fn select(&self, node_id: usize) -> Option<usize> {
        const C_PUCT: f32 = 1.0;
        let cur = &self.nodes[node_id];
        
        let mut best = f32::MIN;
        let mut best_node = None;

        for (&placement, &child_id) in &cur.children {
            let child = &self.nodes[child_id];
            
            let prior = cur.priors.get(&placement).copied().unwrap_or(0.0);
            
            let q_value = if child.visits > 0 { child.value() } else { 0.0 };
            let u_value = C_PUCT * prior * (cur.visits as f32).sqrt() / (1.0 + child.visits as f32);
            
            let score = q_value + u_value;
            if score > best {
                best_node = Some(child_id);
                best = score;
            }
        }

        best_node
    }

    fn expand(&mut self, py: Python<'_>, node_id: usize) -> PyResult<Option<(usize, f32)>> {
        if self.nodes[node_id].is_terminal {
            return Ok(None);
        }

        // Find an unexpanded action
        let mut expanded_action = None;
        let mut soft_drops = 0;
        
        for (p, sd) in &self.nodes[node_id].valid_moves {
            if !self.nodes[node_id].children.contains_key(p) {
                expanded_action = Some(*p);
                soft_drops = *sd;
                break;
            }
        }

        let placement = match expanded_action {
            Some(p) => p,
            None => return Ok(None),
        };

        let mut new_state = self.nodes[node_id].state;
        let piece = *self.nodes[node_id].queue.first().unwrap_or(&Piece::O);
        let info = new_state.advance(piece, placement);
        
        // reward from this step
        let step_reward = calculate_reward(&info);

        let next_queue = if self.nodes[node_id].queue.is_empty() {
            vec![]
        } else {
            self.nodes[node_id].queue[1..].to_vec()
        };

        let mut child = Node::new(new_state, next_queue);
        child.parent = Some(node_id);

        let value = if child.is_terminal {
            0.0 // or terminal reward
        } else {
            let placements: Vec<Placement> = child.valid_moves.iter().map(|(p, _)| *p).collect();
            let eval_result = self.evaluator.call1(py, (new_state, child.queue.clone(), placements.clone()))?;
            let (priors, network_value): (Vec<f32>, f32) = eval_result.extract(py)?;
            
            for (p, prob) in placements.into_iter().zip(priors.into_iter()) {
                child.priors.insert(p, prob);
            }
            network_value
        };

        // The backpropagated value is step_reward + gamma * value
        let q_value = step_reward + 0.99 * value;

        let child_index = self.nodes.len();
        self.nodes[node_id].children.insert(placement, child_index);
        self.nodes.push(child);

        Ok(Some((child_index, q_value)))
    }

    fn update(&mut self, value: f32, idx: usize) {
        let child = &mut self.nodes[idx];
        child.visits += 1;
        child.total_value += value;
    }
}

impl Node {
    pub fn new(state: GameState, queue: Vec<Piece>) -> Self {
        let valid_moves = {
            if let Some(piece) = queue.first() {
                find_moves(&state.board, *piece)
            } else {
                vec![]
            }
        }.into_iter().chain(find_moves(&state.board, state.hold)).collect::<Vec<_>>();

        let is_terminal = queue.is_empty() || valid_moves.is_empty();

        Node {
            parent: None,
            children: HashMap::new(),
            valid_moves,
            priors: HashMap::new(),
            queue,

            visits: 0,
            total_value: 0.0,
            state,

            is_terminal,
        }
    }
    fn value(&self) -> f32 {
        if self.visits == 0 {
            0.0
        } else {
            self.total_value / self.visits as f32
        }
    }
}
