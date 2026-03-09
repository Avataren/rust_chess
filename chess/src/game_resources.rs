use bevy::prelude::Resource;
use chess_foundation::ChessMove;

#[derive(Resource)]
pub struct ValidMoves {
    pub moves: Vec<ChessMove>,
}

impl ValidMoves {
    pub fn new() -> Self {
        ValidMoves { moves: Vec::new() }
    }

    pub fn set_moves(&mut self, moves: Vec<ChessMove>) {
        self.moves = moves;
    }
}

#[derive(Resource, Default)]
pub struct LastMove {
    pub start_square: Option<u16>,
    pub target_square: Option<u16>,
}
