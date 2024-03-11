pub mod alpha_beta;
pub mod board_evaluation;
pub mod piece_tables;

pub use board_evaluation::evaluate_board;
// pub use board_evaluation::evaluate_board_for_color;
pub use alpha_beta::alpha_beta;
pub use piece_tables::evaluate_pawn_position;
pub use piece_tables::evaluate_knight_position;
