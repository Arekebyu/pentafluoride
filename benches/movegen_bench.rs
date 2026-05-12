use criterion::{black_box, criterion_group, criterion_main, Criterion};
use pentafluoride::data::{Board, Piece};
use pentafluoride::movegen::find_moves;

fn bench_find_moves(c: &mut Criterion) {
    let empty_board = Board::default();
    let piece = Piece::T;

    c.bench_function("find_moves empty_board T", |b| {
        b.iter(|| find_moves(black_box(&empty_board), black_box(piece)))
    });

    // Create a partially filled board to simulate a real game state
    let mut complex_board = Board::default();
    for i in 0..10 {
        if i != 5 {
            // Fill bottom 4 rows for all columns except middle one
            complex_board.cols[i] = 0b1111;
        } else {
            // Fill bottom 2 rows for middle column
            complex_board.cols[i] = 0b0011;
        }
    }
    
    c.bench_function("find_moves complex_board T", |b| {
        b.iter(|| find_moves(black_box(&complex_board), black_box(piece)))
    });

    // Create a board with an overhang for t-spins
    let mut tspin_board = Board::default();
    tspin_board.cols[3] = 0b11;
    tspin_board.cols[4] = 0b01;
    tspin_board.cols[5] = 0b11;
    tspin_board.cols[4] |= 1 << 3; // Overhang

    c.bench_function("find_moves tspin_board T", |b| {
        b.iter(|| find_moves(black_box(&tspin_board), black_box(piece)))
    });
}

criterion_group!(benches, bench_find_moves);
criterion_main!(benches);
