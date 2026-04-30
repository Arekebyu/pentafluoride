use std::vec;

// modified from ColdClear 2 by MinusKelvin
use enum_map::Enum;
use enumset::{EnumSet, EnumSetType};
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

#[pyclass]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Deserialize)]
pub struct Board {
    #[pyo3(get, set)]
    pub cols: [u64; 10],
}

#[pyclass]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GameState {
    #[pyo3(get, set)]
    pub board: Board,
    pub bag: EnumSet<Piece>,
    #[pyo3(get, set)]
    pub hold: Piece,
    #[pyo3(get, set)]
    pub b2b: u32,
    #[pyo3(get, set)]
    pub combo: u8,
}

#[pyclass]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PieceLocation {
    #[serde(rename = "type")]
    #[pyo3(get, set)]
    pub piece: Piece,
    #[serde(rename = "orientation")]
    #[pyo3(get, set)]
    pub rotation: Rotation,
    #[pyo3(get, set)]
    pub x: i8,
    #[pyo3(get, set)]
    pub y: i8,
}

#[pyclass]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Placement {
    #[pyo3(get, set)]
    pub location: PieceLocation,
    #[pyo3(get, set)]
    pub spin: Spin,
}

#[pyclass]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PlacementInfo {
    pub placement: Placement,
    #[pyo3(get, set)]
    pub lines_cleared: u8,
    #[pyo3(get, set)]
    pub combo: u32,
    #[pyo3(get, set)]
    pub b2b: i32,
    #[pyo3(get, set)]
    pub pc: bool,
}

#[allow(clippy::derive_hash_xor_eq)]
#[pyclass]
#[derive(EnumSetType, Enum, Debug, Hash, Serialize, Deserialize)]
pub enum Piece {
    I = 0,
    O = 1,
    T = 2,
    L = 3,
    J = 4,
    S = 5,
    Z = 6,
}

#[pyclass]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Rotation {
    North = 0,
    West = 1,
    South = 2,
    East = 3,
}

#[pyclass]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Spin {
    None = 0,
    Mini = 1,
    Full = 2,
}

impl Piece {
    pub const fn cells(self) -> [(i8, i8); 4] {
        match self {
            Piece::I => [(-1, 0), (0, 0), (1, 0), (2, 0)],
            Piece::O => [(0, 0), (1, 0), (0, 1), (1, 1)],
            Piece::T => [(-1, 0), (0, 0), (1, 0), (0, 1)],
            Piece::L => [(-1, 0), (0, 0), (1, 0), (1, 1)],
            Piece::J => [(-1, 0), (0, 0), (1, 0), (-1, 1)],
            Piece::S => [(-1, 0), (0, 0), (0, 1), (1, 1)],
            Piece::Z => [(-1, 1), (0, 1), (0, 0), (1, 0)],
        }
    }
}

impl Rotation {
    pub const fn rotate_cell(self, (x, y): (i8, i8)) -> (i8, i8) {
        match self {
            Rotation::North => (x, y),
            Rotation::East => (y, -x),
            Rotation::South => (-x, -y),
            Rotation::West => (-y, x),
        }
    }

    pub const fn rotate_cells(self, cells: [(i8, i8); 4]) -> [(i8, i8); 4] {
        [
            self.rotate_cell(cells[0]),
            self.rotate_cell(cells[1]),
            self.rotate_cell(cells[2]),
            self.rotate_cell(cells[3]),
        ]
    }

    pub const fn cw(self) -> Self {
        match self {
            Rotation::North => Rotation::East,
            Rotation::East => Rotation::South,
            Rotation::South => Rotation::West,
            Rotation::West => Rotation::North,
        }
    }

    pub const fn ccw(self) -> Self {
        match self {
            Rotation::North => Rotation::West,
            Rotation::East => Rotation::North,
            Rotation::South => Rotation::East,
            Rotation::West => Rotation::South,
        }
    }

    pub const fn flip(self) -> Self {
        match self {
            Rotation::North => Rotation::South,
            Rotation::East => Rotation::West,
            Rotation::South => Rotation::North,
            Rotation::West => Rotation::East,
        }
    }
}

macro_rules! lutify {
    (($e:expr) for $v:ident in [$($val:expr),*]) => {
        [
            $(
                {
                    let $v = $val;
                    $e
                }
            ),*
        ]
    };
}

macro_rules! piece_lut {
    ($v:ident => $e:expr) => {
        lutify!(($e) for $v in [Piece::I, Piece::O, Piece::T, Piece::L, Piece::J, Piece::S, Piece::Z])
    };
}

macro_rules! rotation_lut {
    ($v:ident => $e:expr) => {
        lutify!(($e) for $v in [Rotation::North, Rotation::West, Rotation::South, Rotation::East])
    };
}

impl PieceLocation {
    pub const fn cells(&self) -> [(i8, i8); 4] {
        const LUT: [[[(i8, i8); 4]; 4]; 7] =
            piece_lut!(piece => rotation_lut!(rotation => rotation.rotate_cells(piece.cells())));
        self.translate_cells(LUT[self.piece as usize][self.rotation as usize])
    }

    const fn translate(&self, (x, y): (i8, i8)) -> (i8, i8) {
        (x + self.x, y + self.y)
    }

    const fn translate_cells(&self, cells: [(i8, i8); 4]) -> [(i8, i8); 4] {
        [
            self.translate(cells[0]),
            self.translate(cells[1]),
            self.translate(cells[2]),
            self.translate(cells[3]),
        ]
    }

    pub fn obstructed(&self, board: &Board) -> bool {
        self.cells().iter().any(|&cell| board.occupied(cell))
    }

    pub fn drop_distance(&self, board: &Board) -> i8 {
        self.cells()
            .iter()
            .map(|&(x, y)| board.distance_to_ground(x, y))
            .min()
            .unwrap()
    }

    pub fn above_stack(&self, board: &Board) -> bool {
        self.cells()
            .iter()
            .all(|&(x, y)| y >= 64 - board.cols[x as usize].leading_zeros() as i8)
    }

    pub fn canonical_form(&self) -> PieceLocation {
        match self.piece {
            Piece::T | Piece::J | Piece::L => *self,
            Piece::O => match self.rotation {
                Rotation::North => *self,
                Rotation::East => PieceLocation {
                    rotation: Rotation::North,
                    y: self.y - 1,
                    ..*self
                },
                Rotation::South => PieceLocation {
                    rotation: Rotation::North,
                    x: self.x - 1,
                    y: self.y - 1,
                    ..*self
                },
                Rotation::West => PieceLocation {
                    rotation: Rotation::North,
                    x: self.x - 1,
                    ..*self
                },
            },
            Piece::S | Piece::Z => match self.rotation {
                Rotation::North | Rotation::East => *self,
                Rotation::South => PieceLocation {
                    rotation: Rotation::North,
                    y: self.y - 1,
                    ..*self
                },
                Rotation::West => PieceLocation {
                    rotation: Rotation::East,
                    x: self.x - 1,
                    ..*self
                },
            },
            Piece::I => match self.rotation {
                Rotation::North | Rotation::East => *self,
                Rotation::South => PieceLocation {
                    rotation: Rotation::North,
                    x: self.x - 1,
                    ..*self
                },
                Rotation::West => PieceLocation {
                    rotation: Rotation::East,
                    y: self.y + 1,
                    ..*self
                },
            },
        }
    }
}

impl Board {
    pub const fn occupied(&self, (x, y): (i8, i8)) -> bool {
        if x < 0 || x >= 10 || y < 0 || y >= 40 {
            return true;
        }
        self.cols[x as usize] & 1 << y != 0
    }

    pub fn distance_to_ground(&self, x: i8, y: i8) -> i8 {
        debug_assert!((0..10).contains(&x));
        debug_assert!((0..64).contains(&y));
        if y == 0 {
            return 0;
        }
        (!self.cols[x as usize] << (64 - y)).leading_ones() as i8
    }

    pub fn place(&mut self, piece: PieceLocation) {
        for &(x, y) in &piece.cells() {
            debug_assert!((0..10).contains(&x));
            debug_assert!((0..40).contains(&y));
            self.cols[x as usize] |= 1 << y;
        }
    }

    pub fn line_clears(&self) -> u64 {
        self.cols.iter().fold(!0, |a, b| a & b)
    }

    pub fn remove_lines(&mut self, lines: u64) {
        for c in &mut self.cols {
            clear_lines(c, lines);
        }
    }
}

#[pymethods]
impl GameState {
    pub fn advance(&mut self, next: Piece, placement: Placement) -> PlacementInfo {
        self.bag.remove(next);
        if self.bag.is_empty() {
            self.bag = EnumSet::all();
        }
        if placement.location.piece != next {
            self.hold = next;
        }
        self.board.place(placement.location);
        let cleared_mask = self.board.line_clears();
        let mut b2b = self.b2b as i32;
        let mut pc = false;
        let prev_combo = self.combo;
        if cleared_mask != 0 {
            self.board.remove_lines(cleared_mask);
            let hard = cleared_mask.count_ones() == 4 || !matches!(placement.spin, Spin::None);
            if self.board.cols.iter().all(|&c| c == 0) {
                pc = true;
                b2b += 4;
            }
            if !(hard | pc) {
                self.b2b = 0;
                b2b = -b2b;
            } else {
                b2b += 1;
            }
            self.combo += 1;
        } else {
            self.combo = 0;
        }
        PlacementInfo {
            placement,
            lines_cleared: cleared_mask.count_ones() as u8,
            combo: prev_combo as u32,
            b2b,
            pc,
        }
    }
    #[new]
    pub fn gamestate(board: [u64; 10], hold: Piece, b2b: u32, combo: u8) -> Self {
        Self {
            board: Board { cols: board },
            bag: EnumSet::all(),
            hold,
            b2b,
            combo,
        }
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
fn clear_lines(col: &mut u64, lines: u64) {
    *col = unsafe {
        // SAFETY: #[cfg()] guard ensures that this instruction exists at compile time
        std::arch::x86_64::_pext_u64(*col, !lines)
    };
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "bmi2")))]
fn clear_lines(col: &mut u64, mut lines: u64) {
    while lines != 0 {
        let i = lines.trailing_zeros();
        let mask = (1 << i) - 1;
        *col = *col & mask | *col >> 1 & !mask;
        lines &= !(1 << i);
        lines >>= 1;
    }
}

#[pyfunction]
pub fn calculate_reward(info: &PlacementInfo) -> f32 {
    let mut r = 0.0;
    let lines = info.lines_cleared;
    let spin = info.placement.spin;
    
    let is_difficult = lines == 4 || (lines > 0 && spin != Spin::None);
    
    let mut base = 0.0;
    if spin == Spin::Full {
        base += match lines {
            0 => 4.0,   // T-Spin
            1 => 8.0,   // T-Spin Single
            2 => 12.0,  // T-Spin Double
            3 => 16.0,  // T-Spin Triple
            _ => 4.0,
        };
    } else if spin == Spin::Mini {
        base += match lines {
            0 => 1.0,   // Mini T-Spin
            1 => 2.0,   // Mini T-Spin Single
            _ => 1.0,
        };
    } else {
        base += match lines {
            1 => 1.0,   // Single
            2 => 3.0,   // Double
            3 => 5.0,   // Triple
            4 => 8.0,   // Tetris
            _ => 0.0,
        };
    }
    
    // B2B Bonus: +50% to difficult line clears (if b2b > 1)
    if is_difficult && info.b2b > 1 {
        base *= 1.5;
    }
    r += base;
    
    // Perfect Clear
    if info.pc {
        r += 30.0;
    }
    
    // Combos: +(0.5 * current combo)
    if lines > 0 && info.combo > 0 {
        r += 0.5 * (info.combo as f32);
    }
    
    // Lower board state bonus
    let y = info.placement.location.y as f32;
    r += 0.01 * (20.0 - y);
    
    r
}

#[pymethods]
impl Piece {
    #[getter]
    pub fn value(&self) -> u8 {
        *self as u8
    }
}

#[pymethods]
impl Rotation {
    #[getter]
    pub fn value(&self) -> u8 {
        *self as u8
    }
}

#[pymethods]
impl Spin {
    #[getter]
    pub fn value(&self) -> u8 {
        *self as u8
    }
}
