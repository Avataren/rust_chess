pub mod bitboard; // Declare the bitboard module
pub mod move_patterns;

pub use move_patterns::get_rook_move_patterns;
pub use move_patterns::get_bishop_move_patterns;
pub use bitboard::Bitboard; // Make Bitboard available for public use
