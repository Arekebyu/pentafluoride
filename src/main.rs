use enumset::EnumSet;
use pentafluoride::mcts::{self, mcts_search};

use pentafluoride::data::*;
mod data;

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
    print!("{:?}", mcts_search(root, queue, 5000));
    puffin::GlobalProfiler::lock().new_frame()
}
