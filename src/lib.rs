use pyo3::prelude::*;

#[macro_use]
pub mod data;
pub mod mcts;
pub mod movegen;

#[pymodule]
fn pentafluoride(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(mcts::mcts_search, m)?)?;
    m.add_class::<data::Board>()?;
    m.add_class::<data::GameState>()?;
    m.add_class::<data::Piece>()?;
    m.add_class::<data::PieceLocation>()?;
    m.add_class::<data::Placement>()?;
    m.add_class::<data::Rotation>()?;
    m.add_class::<data::Spin>()?;
    Ok(())
}
