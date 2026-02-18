use enumset::EnumSet;
use pyo3::prelude::*;

use crate::data::*;
use crate::mcts::mcts_search;
#[macro_use]
mod data;
mod mcts;
mod movegen;

fn main() {
    let server_addr = format!("127.0.0.1:{}", puffin_http::DEFAULT_PORT);
    let _puffin_server = puffin_http::Server::new(&server_addr).unwrap();
    eprintln!("Run this to view profiling data:  puffin_viewer {server_addr}");
    puffin::set_scopes_on(true);
    let root = GameState {
        board: Board { cols: [0; 10] },
        bag: EnumSet::all(),
        hold: Piece::I,
        b2b: 0,
        combo: 0,
    };
    let queue = vec![Piece::L, Piece::J, Piece::O, Piece::T, Piece::S];
    puffin::GlobalProfiler::lock().new_frame()
}
