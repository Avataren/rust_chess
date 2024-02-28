pub mod magic_constants;
mod magics_generator;
pub mod masks;
pub mod move_generator;
pub mod piece_conductor;
pub mod piece_patterns;

pub use piece_patterns::get_bishop_move_patterns;
pub use piece_patterns::get_king_move_patterns;
pub use piece_patterns::get_knight_move_patterns;
pub use piece_patterns::get_pawn_move_patterns;
pub use piece_patterns::get_rook_move_patterns;
