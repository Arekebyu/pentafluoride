use std::fs::File;
use std::time::Duration;
use pprof::ProfilerGuardBuilder;

use pentafluoride::data::{GameState, Piece};
use pentafluoride::pure_mcts::pure_mcts_search;

fn main() {
    println!("Initializing Pure MCTS Profile...");
    
    // Start profiling
    let guard = ProfilerGuardBuilder::default().frequency(1000).blocklist(&["libc", "libgcc", "pthread", "vdso"]).build().unwrap();

    let pieces = vec![Piece::L, Piece::J, Piece::O, Piece::T, Piece::S, Piece::Z, Piece::I];
    let mut state = GameState::gamestate([0; 10], Piece::L, 0, 0);
    
    let queue = vec![Piece::J, Piece::O, Piece::T, Piece::S, Piece::Z];

    println!("Running Pure MCTS for 5000 iterations to generate profile...");
    
    // We run the mcts search without PyO3 overhead to get pure rust stack
    // We don't care about the Python object returned, we just want it to do the work.
    let _ = pure_mcts_search(state, queue, 5000).unwrap();

    println!("Profile complete! Generating flamegraph...");
    
    if let Ok(report) = guard.report().build() {
        let file = File::create("/home/r/.gemini/antigravity/brain/1bdc7820-1a69-476a-b81a-7aeecb676601/artifacts/rust_flamegraph.svg").unwrap();
        report.flamegraph(file).unwrap();
        println!("Flamegraph generated at rust_flamegraph.svg");
    }
}
