use bevy::ecs::system::Resource;
use chess_foundation::ChessMove;

#[derive(Resource)]
pub struct ValidMoves {
    pub moves: Vec<ChessMove>,
}

impl ValidMoves {
    pub fn new() -> Self {
        ValidMoves { moves: Vec::new() }
    }

    pub fn clear(&mut self) {
        self.moves.clear();
    }

    pub fn set_moves(&mut self, moves: Vec<ChessMove>) {
        self.moves = moves;
    }
}
