use crate::data::*;
use crate::movegen::find_moves;
use pyo3::prelude::*;
use rand::seq::IndexedRandom;
use std::collections::HashMap;
use std::vec;

pub struct MCTS {
    nodes: Vec<Node>,
    evaluator: Py<PyAny>,
}
#[derive(Clone)]
pub struct Node {
    parent: Option<usize>,
    children: HashMap<Placement, (f32, usize)>, // placement -> (policy, index)
    queue: Vec<Piece>,

    visits: usize,
    total_score: f32,

    state: GameState,

    // for regularizing rewards
    min_score: f32,
    max_score: f32,
}

#[pyfunction]
pub fn mcts_search(
    py: Python<'_>,
    root: GameState,
    queue: Vec<Piece>,
    iteration: usize,
    evaluator: Py<PyAny>,
) -> Placement {
    let mut tree = MCTS::new(root, queue, evaluator);

    for _ in 0..iteration {
        let mut node_idx = 0;
        let mut path = vec![node_idx];

        // 1. Selection: Traverse down the tree using the UCB1 policy
        loop {
            let actions = tree.nodes[node_idx].actions();
            let node = &tree.nodes[node_idx];
            if !actions.is_empty() && node.children.len() == actions.len() {
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

impl MCTS {
    pub fn new(root: GameState, queue: Vec<Piece>, evaluator: Py<PyAny>) -> Self {
        // we make some assumptions here, namely, root gamestate already has hold piece
        let root = Node::new(root, queue);
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
        let actions = self.nodes[node_id].actions();
        let unexpanded: Vec<_> = actions
            .into_iter()
            .filter(|(p, _)| !self.nodes[node_id].children.contains_key(p))
            .collect();

        let (placement, _) = *unexpanded.choose(&mut rand::rng())?;

        let mut new_state = self.nodes[node_id].state;
        let piece = *self.nodes[node_id].queue.first().unwrap_or(&Piece::O);
        let _info = new_state.advance(piece, placement);

        let next_queue = if self.nodes[node_id].queue.is_empty() {
            vec![]
        } else {
            self.nodes[node_id].queue[1..].to_vec()
        };

        let eval_result = self
            .evaluator
            .call1(py, (new_state, next_queue.clone()))
            .ok()?;
        let (policy_score, quality_score): (f32, f32) = eval_result.extract(py).ok()?;

        let child_index = self.nodes.len();
        let mut child = Node::new(new_state, next_queue);
        child.parent = Some(node_id);

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
        Node {
            parent: None,
            children: HashMap::new(),
            queue,

            visits: 0,
            total_score: 0.0,

            state,

            min_score: f32::INFINITY,
            max_score: f32::NEG_INFINITY,
        }
    }
    fn value(&self) -> f32 {
        match self.visits {
            0 => 0.0,
            visits => self.total_score / visits as f32,
        }
    }
    fn actions(&self) -> Vec<(Placement, u32)> {
        let placements = {
            if let Some(piece) = self.queue.first() {
                find_moves(&self.state.board, *piece)
            } else {
                vec![]
            }
        };
        placements
            .into_iter()
            .chain(find_moves(&self.state.board, self.state.hold))
            .collect::<Vec<_>>()
    }
}

macro_rules! apply_combo {
    ($v:ident => $e:expr) => {
        lutify!(($e) for $v in [0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,16.0,17.0,18.0,19.0,20.0])
    };

}

fn calculate_reward(info: &PlacementInfo) -> f32 {
    // this is so ugly lol
    let mut r = 0.0f32;
    let c0_attack: [f32; 21] = apply_combo!(val => (1.0f32 + 1.25f32 * val).ln());
    if info.placement.spin == Spin::Full {
        r += 2.0 * info.lines_cleared as f32;
    } else {
        r += match info.lines_cleared {
            4 => 4.0,
            3 => 2.0,
            2 => 1.0,
            _ => 0.0,
        }
    }
    if info.b2b > 0 {
        r += 1.0;
    }
    if info.lines_cleared == 0 {
        r += c0_attack[info.combo as usize];
    } else {
        r *= 1.0 + 0.25 * info.combo as f32 // might want to quantize this reward
    }
    if info.b2b < -4 {
        r -= info.b2b as f32
    }
    if info.pc {
        r += 4.0
    }

    r
}
