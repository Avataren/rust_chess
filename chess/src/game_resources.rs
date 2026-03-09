use bevy::prelude::*;
use chess_evaluation::OpeningBook;
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

#[derive(Resource, Default, PartialEq, Debug, Clone, Copy)]
pub enum GameOverState {
    #[default]
    Playing,
    PlayerWins,
    OpponentWins,
    Stalemate,
}

/// Holds a game-over outcome that should be applied after the current tween finishes.
#[derive(Resource, Default)]
pub struct PendingGameOver(pub Option<GameOverState>);

#[derive(Resource, Default, PartialEq, Debug, Clone, Copy)]
pub enum PlayerColor {
    #[default]
    White,
    Black,
}

#[derive(Resource, Default, PartialEq, Debug, Clone, Copy)]
pub enum GamePhase {
    #[default]
    StartScreen,
    Playing,
}

#[derive(Resource)]
pub struct OpeningBookRes {
    pub book: OpeningBook,
}

/// Tracks the last detected opening name, persisted until a new one is found.
#[derive(Resource, Default)]
pub struct CurrentOpening(pub String);
