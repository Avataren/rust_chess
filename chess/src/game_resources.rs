use bevy::prelude::*;
use chess_evaluation::OpeningBook;
use chess_foundation::ChessMove;
use std::collections::HashMap;
use std::sync::{Arc, atomic::AtomicBool};

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

/// Sound effect to play when the AI piece finishes its landing animation.
#[derive(Resource, Default)]
pub struct PendingMoveSound(pub Option<&'static str>);

#[derive(Resource, Default, PartialEq, Debug, Clone, Copy)]
pub enum Difficulty {
    VeryEasy,
    Easy,
    #[default]
    Medium,
    Hard,
    VeryHard,
}

impl Difficulty {
    pub fn search_depth(self) -> i32 {
        match self {
            Difficulty::VeryEasy => 1,
            Difficulty::Easy     => 2,
            Difficulty::Medium   => 4,
            Difficulty::Hard     => 7,
            Difficulty::VeryHard => 12,
        }
    }

    /// Time budget for the search. `None` means run to the full depth cap.
    pub fn time_limit(self) -> Option<std::time::Duration> {
        match self {
            Difficulty::VeryEasy => None,
            Difficulty::Easy     => None,
            Difficulty::Medium   => Some(std::time::Duration::from_secs(3)),
            Difficulty::Hard     => Some(std::time::Duration::from_secs(8)),
            Difficulty::VeryHard => Some(std::time::Duration::from_secs(5)),
        }
    }

    /// Whether this difficulty level uses background pondering.
    pub fn ponders(self) -> bool {
        matches!(self, Difficulty::Hard | Difficulty::VeryHard)
    }
}

/// State for background multi-ponder searches (thinking on the human's time).
/// Active only on Hard and VeryHard difficulty.
/// Native: ponders top 4 opponent candidate moves in parallel.
/// WASM:   ponders top 2 candidate moves.
#[derive(Resource, Default)]
pub struct PonderState {
    /// Stop flags — one per active ponder task.
    pub stops: Vec<Arc<AtomicBool>>,
    /// Completed ponder results keyed by the Zobrist hash of the pondered position.
    pub results: HashMap<u64, (i32, Option<ChessMove>, Option<ChessMove>)>,
    /// True while ponder tasks are in flight or have un-consumed results.
    pub ponder_active: bool,
    /// True while the main (non-ponder) search task is in flight.
    pub main_search_active: bool,
}
