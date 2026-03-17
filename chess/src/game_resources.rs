use bevy::prelude::*;
use chess_evaluation::OpeningBook;
use chess_foundation::ChessMove;
use std::collections::HashMap;
use std::sync::{Arc, atomic::AtomicBool};
use std::time::Duration;

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

// ── Time controls ─────────────────────────────────────────────────────────────

#[derive(Resource, Clone, Copy, PartialEq, Default)]
pub enum TimeControl {
    Bullet,
    #[default]
    Blitz,
    Rapid,
}

impl TimeControl {
    pub fn initial_time(self) -> Duration {
        match self {
            TimeControl::Bullet => Duration::from_secs(120),
            TimeControl::Blitz  => Duration::from_secs(300),
            TimeControl::Rapid  => Duration::from_secs(600),
        }
    }

    pub fn increment(self) -> Duration {
        match self {
            TimeControl::Bullet => Duration::from_secs(1),
            TimeControl::Blitz  => Duration::from_secs(3),
            TimeControl::Rapid  => Duration::from_secs(5),
        }
    }
}

// ── Strength levels ───────────────────────────────────────────────────────────

/// Five strength levels targeting approximate ELO bands.
/// ELO values are rough estimates — calibrate against Stockfish/lichess.
#[derive(Resource, Clone, Copy, PartialEq, Default)]
pub enum Strength {
    S1,
    S2,
    #[default]
    S3,
    S4,
    S5,
}

impl Strength {
    pub fn max_depth(self) -> i32 {
        match self {
            Strength::S1 => 2,
            Strength::S2 => 3,
            Strength::S3 => 6,
            Strength::S4 => 12,
            Strength::S5 => 50,
        }
    }

    /// Root-level eval noise in centipawns.  After searching to max_depth, each root
    /// move's static score has a random ±noise sample added before selecting the best.
    pub fn eval_noise_cp(self) -> i32 {
        match self {
            Strength::S1 => 150,  // ~old Casual — many blunders, easy for a beginner
            Strength::S2 => 60,   // occasionally drops a pawn, misses short tactics
            Strength::S3 => 20,   // solid but imperfect
            Strength::S4 => 5,    // near-engine strength, rare slip
            Strength::S5 => 0,
        }
    }

    /// Whether this strength level uses background pondering.
    pub fn ponders(self) -> bool {
        matches!(self, Strength::S4 | Strength::S5)
    }

}

// ── GameSettings (bundles time_control + strength) ───────────────────────────

/// Bundles `time_control` and `strength` into one resource so that
/// `handle_async_moves` stays within Bevy's 16-parameter limit.
#[derive(Resource)]
pub struct GameSettings {
    pub time_control: TimeControl,
    pub strength: Strength,
}

impl Default for GameSettings {
    fn default() -> Self {
        GameSettings {
            time_control: TimeControl::Blitz,
            strength: Strength::S3,
        }
    }
}

// ── GameClocks ────────────────────────────────────────────────────────────────

/// Live game clocks for both sides.
#[derive(Resource)]
pub struct GameClocks {
    pub white_remaining: Duration,
    pub black_remaining: Duration,
    pub increment: Duration,
    initial: Duration,
}

impl GameClocks {
    pub fn new(tc: TimeControl) -> Self {
        let initial = tc.initial_time();
        let increment = tc.increment();
        GameClocks {
            white_remaining: initial,
            black_remaining: initial,
            increment,
            initial,
        }
    }

    /// Update to a new time control and reset both clocks.
    pub fn set_time_control(&mut self, tc: TimeControl) {
        self.initial = tc.initial_time();
        self.increment = tc.increment();
        self.reset();
    }

    /// Restore both clocks to the initial time.
    pub fn reset(&mut self) {
        self.white_remaining = self.initial;
        self.black_remaining = self.initial;
    }

    /// Subtract `delta` from the active side's clock (saturating).
    pub fn tick(&mut self, is_white: bool, delta: Duration) {
        if is_white {
            self.white_remaining = self.white_remaining.saturating_sub(delta);
        } else {
            self.black_remaining = self.black_remaining.saturating_sub(delta);
        }
    }

    /// Add the increment to a side's clock after they make a move.
    pub fn add_increment(&mut self, is_white: bool) {
        if is_white {
            self.white_remaining += self.increment;
        } else {
            self.black_remaining += self.increment;
        }
    }

    pub fn remaining(&self, is_white: bool) -> Duration {
        if is_white { self.white_remaining } else { self.black_remaining }
    }

    pub fn is_flagged(&self, is_white: bool) -> bool {
        self.remaining(is_white).is_zero()
    }

    /// Recommended search time budget: remaining/40 + increment×0.75,
    /// clamped to [100 ms, remaining/2].
    pub fn move_budget(&self, is_white: bool) -> Duration {
        let remaining = self.remaining(is_white);
        if remaining.is_zero() {
            return Duration::from_millis(100);
        }
        let budget_secs = remaining.as_secs_f64() / 40.0
            + self.increment.as_secs_f64() * 0.75;
        let budget = Duration::from_secs_f64(budget_secs);
        let min = Duration::from_millis(100);
        let max = (remaining / 2).max(min);
        budget.clamp(min, max)
    }
}

// ── Ponder state ──────────────────────────────────────────────────────────────

/// State for background multi-ponder searches (thinking on the human's time).
/// Active only on Strong (S4) and Maximum (S5) strength.
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
