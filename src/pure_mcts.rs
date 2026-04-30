use std::collections::HashMap;
use pyo3::prelude::*;
use rand::prelude::*;
use crate::data::{GameState, Piece, Placement, calculate_reward};
use crate::movegen::find_moves;

const C_PUNCT: f32 = 1.414; // Exploration constant for UCB1

struct Node {
    state: GameState,
    valid_moves: Vec<(Placement, u32)>,
    children: HashMap<Placement, usize>,
    queue: Vec<Piece>,
    visits: u32,
    total_value: f32,
    reward_from_parent: f32,
}

impl Node {
    fn new(state: GameState, queue: Vec<Piece>, reward_from_parent: f32) -> Self {
        let piece = queue.first().copied();
        let mut valid_moves = Vec::new();
        if let Some(p) = piece {
            valid_moves.extend(find_moves(&state.board, p));
            if state.hold != p {
                valid_moves.extend(find_moves(&state.board, state.hold));
            }
        }
        
        // Deduplicate valid_moves just in case
        let mut unique_moves = HashMap::new();
        for (m, sd) in valid_moves {
            unique_moves.insert(m, sd);
        }
        let valid_moves: Vec<_> = unique_moves.into_iter().collect();

        Node {
            state,
            valid_moves,
            children: HashMap::new(),
            queue,
            visits: 0,
            total_value: 0.0,
            reward_from_parent,
        }
    }
    
    fn is_terminal(&self) -> bool {
        self.valid_moves.is_empty() || self.queue.is_empty()
    }
    
    fn is_fully_expanded(&self) -> bool {
        self.children.len() == self.valid_moves.len()
    }
}

pub struct PureMCTS {
    nodes: Vec<Node>,
}

impl PureMCTS {
    pub fn new(root_state: GameState, queue: Vec<Piece>) -> Self {
        let root = Node::new(root_state, queue, 0.0);
        PureMCTS {
            nodes: vec![root],
        }
    }
    
    pub fn select(&self) -> (usize, Vec<usize>) {
        let mut current_node = 0;
        let mut path = vec![current_node];
        
        while self.nodes[current_node].is_fully_expanded() && !self.nodes[current_node].is_terminal() {
            let mut best_score = f32::NEG_INFINITY;
            let mut best_child = 0;
            
            for &child_idx in self.nodes[current_node].children.values() {
                let child = &self.nodes[child_idx];
                let ucb = if child.visits == 0 {
                    f32::INFINITY
                } else {
                    let q = child.total_value / child.visits as f32;
                    let u = C_PUNCT * ((self.nodes[current_node].visits as f32).ln() / child.visits as f32).sqrt();
                    q + u
                };
                
                if ucb > best_score {
                    best_score = ucb;
                    best_child = child_idx;
                }
            }
            
            current_node = best_child;
            path.push(current_node);
        }
        
        (current_node, path)
    }
    
    pub fn expand(&mut self, node_id: usize) -> usize {
        let node = &self.nodes[node_id];
        if node.is_terminal() {
            return node_id;
        }
        
        let mut expanded_action = None;
        for (p, _) in &node.valid_moves {
            if !node.children.contains_key(p) {
                expanded_action = Some(*p);
                break;
            }
        }
        
        let placement = expanded_action.unwrap();
        let mut new_state = node.state;
        let mut new_queue = node.queue.clone();
        
        let piece = new_queue.remove(0);
        let info = new_state.advance(piece, placement);
        let step_reward = calculate_reward(&info);
        
        let child_node = Node::new(new_state, new_queue, step_reward);
        let child_id = self.nodes.len();
        self.nodes.push(child_node);
        
        self.nodes[node_id].children.insert(placement, child_id);
        
        child_id
    }
    
    pub fn rollout(&self, node_id: usize) -> f32 {
        let node = &self.nodes[node_id];
        let mut state = node.state;
        let mut queue = node.queue.clone();
        let mut total_reward = 0.0;
        
        let mut rng = rand::rng();
        
        while !queue.is_empty() {
            let piece = queue.remove(0);
            let mut moves = find_moves(&state.board, piece);
            moves.extend(find_moves(&state.board, state.hold));
            
            if moves.is_empty() {
                return total_reward;
            }
            
            let (m, _) = *moves.choose(&mut rng).unwrap();
            let info = state.advance(piece, m);
            total_reward += calculate_reward(&info);
        }
        
        // Queue is exhausted: use bag prediction and average over all possible pieces
        let mut expected_reward = 0.0;
        let mut valid_pieces = 0;
        for piece in state.bag.iter() {
            let mut moves = find_moves(&state.board, piece);
            moves.extend(find_moves(&state.board, state.hold));
            
            if !moves.is_empty() {
                let mut piece_avg = 0.0;
                for (m, _) in &moves {
                    let mut temp_state = state;
                    let info = temp_state.advance(piece, *m);
                    piece_avg += calculate_reward(&info);
                }
                expected_reward += piece_avg / moves.len() as f32;
            }
            valid_pieces += 1;
        }
        
        if valid_pieces > 0 {
            total_reward += expected_reward / valid_pieces as f32;
        }
        
        total_reward
    }
    
    pub fn backpropagate(&mut self, path: &[usize], mut value: f32) {
        for &node_id in path.iter().rev() {
            self.nodes[node_id].visits += 1;
            self.nodes[node_id].total_value += value;
            value += self.nodes[node_id].reward_from_parent;
        }
    }
}

#[pyfunction]
pub fn pure_mcts_search(
    root: GameState,
    queue: Vec<Piece>,
    iteration: usize,
) -> PyResult<(Placement, HashMap<Placement, f32>)> {
    let mut tree = PureMCTS::new(root, queue);
    
    for _ in 0..iteration {
        let (node_idx, mut path) = tree.select();
        let mut leaf = node_idx;
        
        if !tree.nodes[leaf].is_terminal() {
            leaf = tree.expand(leaf);
            path.push(leaf);
        }
        
        let value = tree.rollout(leaf);
        tree.backpropagate(&path, value);
    }
    
    let root_node = &tree.nodes[0];
    let mut best_placement = None;
    let mut most_visits = 0;
    let mut policy = HashMap::new();
    
    let total_visits: u32 = root_node.children.values().map(|&idx| tree.nodes[idx].visits).sum();
    
    for (placement, &child_idx) in &root_node.children {
        let child_visits = tree.nodes[child_idx].visits;
        policy.insert(*placement, child_visits as f32 / total_visits.max(1) as f32);
        
        if child_visits > most_visits {
            most_visits = child_visits;
            best_placement = Some(*placement);
        }
    }
    
    Ok((best_placement.unwrap(), policy))
}
