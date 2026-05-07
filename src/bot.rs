use std::process::Child;

use enum_map::EnumMap;
use enumset::EnumSet;

use crate::dag::{ChildData, DAG};
use crate::data::*;

use crate::movegen::find_moves;

pub struct Bot {
    dag: DAG,
}

impl Bot {
    pub fn new(root: GameState, queue: impl IntoIterator<Item = Piece>) -> Self {
        Self {
            dag: DAG::new(root, queue),
        }
    }

    pub fn advance(&mut self, mv: Placement) {
        // eprint!("Advancing placement {:?}", mv);
        self.dag.advance_root(mv)
    }

    // optional, will be removed ////////////////////
    pub fn state(&self) -> GameState {
        self.dag.root()
    }
    /////////////////////////////////////////////////

    pub fn add_piece(&mut self, piece: Piece) {
        self.dag.add_piece(piece)
    }

    pub fn suggest(&self) -> Vec<Placement> {
        self.dag.suggest()
    }

    pub fn expand(&self) {
        if let Some(selection) = self.dag.select() {
            let mut moves: EnumMap<Piece, Vec<(Placement, u32)>> = EnumMap::default();
            let (state, next) = selection.state();

            let next_possibilities = next.map(EnumSet::only).unwrap_or(state.bag);
            let mut children: EnumMap<Piece, Vec<ChildData>> = EnumMap::default();

            for piece in next_possibilities {
                moves[piece] = find_moves(&state.board, piece);
            }
            for piece in next_possibilities {
                let moves = moves[piece].iter().chain({
                    if piece == state.hold {
                        [].iter()
                    } else {
                        moves[state.hold].iter()
                    }
                });
                for &(mv, _) in moves {
                    let mut new_state = state;
                    let info = new_state.advance(piece, mv);

                    let (reward) = calculate_reward(&info);

                    children[piece].push(ChildData {
                        state: new_state,
                        mv,
                        reward,
                        eval: 0.0,
                    })
                }
            }

            selection.expand(children);
        }
    }
}
