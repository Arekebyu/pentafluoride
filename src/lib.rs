use pyo3::prelude::*;

#[macro_use]
pub mod data;
pub mod ppo_mcts;
pub mod mcts;
pub mod pure_mcts;
pub mod movegen;

#[pymodule]
fn pentafluoride(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ppo_mcts::ppo_mcts_search, m)?)?;
    m.add_function(wrap_pyfunction!(ppo_mcts::ppo_mcts_generate_targets, m)?)?;
    m.add_function(wrap_pyfunction!(mcts::alphago_mcts_search, m)?)?;
    m.add_function(wrap_pyfunction!(pure_mcts::pure_mcts_search, m)?)?;
    m.add_class::<data::Board>()?;
    m.add_class::<data::GameState>()?;
    m.add_class::<data::Piece>()?;
    m.add_class::<data::PieceLocation>()?;
    m.add_class::<data::Placement>()?;
    m.add_class::<data::PlacementInfo>()?;
    m.add_class::<data::Rotation>()?;
    m.add_class::<data::Spin>()?;
    m.add_function(wrap_pyfunction!(movegen::find_moves, m)?)?;
    m.add_function(wrap_pyfunction!(data::calculate_reward, m)?)?;
    Ok(())
}
