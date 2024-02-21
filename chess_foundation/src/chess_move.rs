#[derive(Debug, Clone, Copy)]
pub struct ChessMove {
    pub from_square: i32,
    pub to_square: i32,
}

impl ChessMove {
    // Primary constructor that directly takes file_index and rank_index
    pub const fn new(from_square: i32, to_square: i32) -> Self {
        Self {
            from_square,
            to_square,
        }
    }
}
