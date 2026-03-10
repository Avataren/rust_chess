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
    Draw,
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

/// True while the AI search task is running.
#[derive(Resource, Default)]
pub struct IsAiThinking(pub bool);

#[derive(Resource, Default, PartialEq, Debug, Clone, Copy)]
pub enum Difficulty {
    Easy,
    #[default]
    Medium,
    Hard,
    VeryHard,
}

impl Difficulty {
    pub fn search_depth(self) -> i32 {
        let d = match self {
            Difficulty::Easy     => 2,
            Difficulty::Medium   => 4,
            Difficulty::Hard     => 7,
            Difficulty::VeryHard => 12,
        };
        // WASM runs the search on the browser's main thread — cap depth to
        // prevent page freezes.  The time limit below provides the real guard.
        #[cfg(target_arch = "wasm32")]
        { d.min(4) }
        #[cfg(not(target_arch = "wasm32"))]
        d
    }

    /// Time budget for the search. `None` means run to the full depth cap.
    pub fn time_limit(self) -> Option<std::time::Duration> {
        match self {
            Difficulty::Easy     => None,
            Difficulty::Medium   => Some(std::time::Duration::from_secs(3)),
            Difficulty::Hard     => Some(std::time::Duration::from_secs(8)),
            Difficulty::VeryHard => Some(std::time::Duration::from_secs(5)),
        }
    }
}
