use std::{
    sync::{
        RwLock,
        atomic::{self, AtomicBool, AtomicU64},
    },
    time::Instant,
};

use crate::data::{GameState, Piece, Placement};
use ahash::HashMap;
use bumpalo_herd::Herd;
use enum_map::EnumMap;
use enumset::EnumSet;
use once_cell::sync::Lazy;
use ouroboros::self_referencing;
use rand::{random, seq::IteratorRandom};

pub struct DAG {
    root: GameState,
    top_layer: Box<Layer>,
    last_advance: Instant,
    new_nodes: AtomicU64,
}

pub struct Child {
    placement: Placement,
    reward: f32,
    total_reward: f32,
}

pub struct ChildData {
    pub(crate) state: GameState,
    pub(crate) mv: Placement,
    pub(crate) eval: f32,
    pub(crate) reward: f32,
}

pub struct Node {
    prev: Vec<(GameState, Placement)>,
    eval: f32,
    expanding: AtomicBool,
    children: Option<EnumMap<Piece, Vec<Child>>>,
}

#[derive(Default)]
pub struct Layer {
    piece: Option<Piece>, // For pieces outside of queue, will have to do some bag prediction
    next_layer: Lazy<Box<Layer>>,
    states: RwLock<HashMap<GameState, Node>>,
}

pub struct Selection<'a> {
    layers: Vec<&'a Layer>,
    game_state: GameState,
}

impl DAG {
    pub fn root(&self) -> GameState {
        self.root
    }

    pub fn new(root: GameState, queue: impl IntoIterator<Item = Piece>) -> Self {
        let mut top_layer = Layer::default();
        top_layer.states.get_mut().unwrap().insert(
            root,
            Node {
                prev: vec![],
                eval: 0.0,
                children: None,
                expanding: AtomicBool::new(false),
            },
        );

        let mut layer = &mut top_layer;
        for piece in queue {
            layer.piece = Some(piece);
            layer = &mut layer.next_layer
        }

        Self {
            root,
            top_layer: Box::new(top_layer),
            last_advance: Instant::now(),
            new_nodes: AtomicU64::new(0),
        }
    }

    pub fn advance_root(&mut self, mv: Placement) {
        puffin::profile_function!();
        let now = Instant::now();
        eprintln!(
            "{:.0} nodes/second",
            *self.new_nodes.get_mut() as f64 / now.duration_since(self.last_advance).as_secs_f64()
        );
        self.last_advance = now;
        *self.new_nodes.get_mut() = 0;
        let top_layer = std::mem::take(&mut *self.top_layer);
        self.root.advance(
            top_layer.piece.expect("cannot advance without next piece"),
            mv,
        );
        Lazy::force(&top_layer.next_layer);
        self.top_layer = Lazy::into_value(top_layer.next_layer).unwrap();
        self.top_layer
            .states
            .get_mut()
            .unwrap()
            .entry(self.root)
            .or_insert(Node {
                prev: vec![],
                eval: 0.0,
                children: None,
                expanding: AtomicBool::new(false),
            });
        self.prune_unreachable();
    }

    fn prune_unreachable(&mut self) {
        let mut layer = self.top_layer.as_mut();
        let mut reachable = ahash::HashSet::default();
        reachable.insert(self.root);

        // Process top layer
        let states = layer.states.get_mut().unwrap();
        states.retain(|&k, _| k == self.root);
        if let Some(root_node) = states.get_mut(&self.root) {
            root_node.prev.clear();
        }

        // Process subsequent layers
        while let Some(next_layer) = once_cell::sync::Lazy::get_mut(&mut layer.next_layer) {
            layer = next_layer.as_mut();
            let mut next_reachable = ahash::HashSet::default();

            let states = layer.states.get_mut().unwrap();
            states.retain(|k, node| {
                node.prev
                    .retain(|(parent_state, _)| reachable.contains(parent_state));
                if node.prev.is_empty() {
                    false
                } else {
                    next_reachable.insert(*k);
                    true
                }
            });

            if next_reachable.is_empty() {
                break;
            }
            reachable = next_reachable;
        }
    }

    pub fn add_piece(&mut self, piece: Piece) {
        puffin::profile_function!();
        let mut layer = &mut self.top_layer;
        loop {
            if layer.piece.is_none() {
                layer.piece = Some(piece);
                return;
            }
            layer = &mut layer.next_layer;
        }
    }

    pub fn suggest(&self) -> Vec<Placement> {
        puffin::profile_function!();
        let states = self.top_layer.states.read().unwrap();
        let children = match &states.get(&self.root).unwrap().children {
            Some(c) => c,
            None => return vec![],
        };

        let mut candidates: Vec<&Child> = vec![];
        match self.top_layer.piece {
            Some(next) => {
                // Can be either next or hold
                candidates.extend(children[next].first());
                if next != self.root.hold {
                    candidates.extend(children[self.root.hold].first());
                }
            }
            None => {
                // Can be hold, or speculation
                for piece in self.root.bag {
                    candidates.extend(children[piece].first());
                }
                if !self.root.bag.contains(self.root.hold) {
                    candidates.extend(children[self.root.hold].first());
                }
            }
        }
        // sort reward by descending order
        candidates.sort_by(|a, b| b.total_reward.total_cmp(&a.total_reward));
        return candidates.into_iter().map(|c| c.placement).collect();
    }

    pub fn select(&self) -> Option<Selection<'_>> {
        puffin::profile_function!();
        let mut layers = vec![&*self.top_layer]; // queue for backpropagation
        let mut game_state = self.root;
        loop {
            let &layer = layers.last().unwrap();
            let guard = layer.states.read().unwrap();
            let node = guard.get(&game_state).unwrap();

            let children = match &node.children {
                None => {
                    // leaf node
                    if node.expanding.swap(true, atomic::Ordering::Acquire) {
                        return None; // another thread expanding
                    } else {
                        return Some(Selection { layers, game_state });
                    }
                }
                Some(children) => children,
            };

            // next piece or bag prediction
            let next_piece = layer.piece.unwrap_or_else(|| {
                let mut rng = rand::rng();
                if game_state.bag.is_empty() {
                    return EnumSet::<Piece>::all().iter().choose(&mut rng).unwrap();
                } else {
                    return game_state.bag.iter().choose(&mut rng).unwrap();
                }
            });
            // random selection
            let s: f32 = random();
            const EXPLORATION: f32 = 0.654;
            let len = children[next_piece].len();
            if len == 0 {
                // Game over or no moves
                return None;
            }
            let i = ((-s.ln() / EXPLORATION) % len as f32) as usize;
            self.new_nodes
                .fetch_add(1 as u64, atomic::Ordering::Relaxed);

            let choice = children[next_piece].get(i).unwrap().placement;

            game_state.advance(next_piece, choice);

            layers.push(&layer.next_layer)
        }
    }
}

impl<'a> Selection<'_> {
    pub fn state(&self) -> (GameState, Option<Piece>) {
        (self.game_state, self.layers.last().unwrap().piece)
    }
    // contains expansion, simulation, and backpropagation.
    pub fn expand(self, children: EnumMap<Piece, Vec<ChildData>>) {
        puffin::profile_function!();
        let mut layers = self.layers;
        let cur_layer = layers.pop().unwrap();

        // create child nodes
        let mut childs = EnumMap::<_, Vec<_>>::default();

        // lock parents before child to prevent deadlock
        let mut states = cur_layer.states.write().unwrap();
        let mut next_states = cur_layer.next_layer.states.write().unwrap();
        // generate children
        for (_, piece_children) in children {
            for child in piece_children {
                let child_node = next_states.entry(child.state).or_insert(Node {
                    prev: vec![],
                    eval: child.eval,
                    children: None,
                    expanding: AtomicBool::new(false),
                });
                child_node.prev.push((self.game_state, child.mv));
                childs[child.mv.location.piece].push(Child {
                    placement: child.mv,
                    reward: child.reward,
                    total_reward: child_node.eval + child.reward,
                })
            }
        }

        // sort each piece's possible placement by reward (descending)
        for piece_placement in childs.values_mut() {
            piece_placement.sort_by(|a, b| b.total_reward.partial_cmp(&a.total_reward).unwrap());
        }

        let mut priors = vec![];

        let node = states.get_mut(&self.game_state).unwrap();

        node.children = Some(childs);

        for (prior_state, mv) in node.prev.iter() {
            priors.push((*prior_state, *mv, self.game_state))
        }

        drop(next_states);
        drop(states);

        let mut prior_layer = cur_layer;
        while let Some(layer) = layers.pop() {
            let mut next_up = vec![];
            for (parent_state, parent_mv, child_state) in priors {
                let mut guard = layer.states.write().unwrap();
                let node = guard.get_mut(&parent_state).unwrap();
                let child_eval = prior_layer
                    .states
                    .read()
                    .unwrap()
                    .get(&child_state)
                    .unwrap()
                    .eval;

                let children = node.children.as_mut().unwrap();
                let list = &mut children[parent_mv.location.piece];

                let idx = list.iter().position(|c| c.placement == parent_mv).unwrap();
                list[idx].total_reward = list[idx].reward + child_eval;

                list.sort_unstable_by(|a, b| b.total_reward.partial_cmp(&a.total_reward).unwrap());

                let idx = list.iter().position(|c| c.placement == parent_mv).unwrap();
                if idx == 0 {
                    let next_possibilities = match layer.piece {
                        Some(p) => EnumSet::only(p),
                        None => parent_state.bag,
                    };

                    let best_for = |piece: Piece| -> f32 {
                        children[piece]
                            .first()
                            .map(|c| c.total_reward)
                            .unwrap_or(-f32::INFINITY)
                    };

                    let expectimax = next_possibilities
                        .iter()
                        .map(|p| best_for(p).max(best_for(parent_state.hold)))
                        .sum::<f32>()
                        / next_possibilities.len() as f32;

                    if node.eval != expectimax {
                        // if this changed the evaluation in any way
                        node.eval = expectimax;
                        for &(grandparent_state, grandparent_mv) in &node.prev {
                            next_up.push((grandparent_state, grandparent_mv, parent_state));
                        }
                    }
                }
            }
            priors = next_up;
            prior_layer = layer;

            if priors.is_empty() {
                break;
            }
        }
    }
}
