// src/lib.rs
pub mod chessboard;
pub mod fen;
pub mod zobrist;
// Re-export main structs for easy access
pub use chessboard::ChessBoard;
pub use fen::FENParser;