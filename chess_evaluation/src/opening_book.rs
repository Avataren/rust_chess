use chess_board::ChessBoard;
use chess_foundation::{piece::PieceType, ChessMove};
use move_generator::{
    move_generator::get_all_legal_moves_for_color, piece_conductor::PieceConductor,
};
use rand::seq::SliceRandom;
use std::collections::HashMap;

// Layout of the 781-entry Zobrist table:
//   0..768  : 12 piece-types × 64 squares  (piece_index * 64 + square)
//   768..772: castling rights (WK, WQ, BK, BQ)
//   772..780: en-passant files a-h
//   780     : side to move (XOR in when black is to move)
const ZOBRIST_SIZE: usize = 781;
const ZOBRIST_CASTLING_WK: usize = 768;
const ZOBRIST_CASTLING_WQ: usize = 769;
const ZOBRIST_CASTLING_BK: usize = 770;
const ZOBRIST_CASTLING_BQ: usize = 771;
const ZOBRIST_EP_OFFSET: usize = 772;
const ZOBRIST_SIDE_TO_MOVE: usize = 780;

fn make_zobrist_table() -> [u64; ZOBRIST_SIZE] {
    let mut table = [0u64; ZOBRIST_SIZE];
    let mut state: u64 = 0x9E3779B97F4A7C15; // golden-ratio-derived seed
    for v in table.iter_mut() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        *v = state;
    }
    table
}

// Piece index following Polyglot ordering:
// black_pawn=0, white_pawn=1, black_knight=2, white_knight=3, ..., white_king=11
fn piece_zobrist_index(piece_type: PieceType, is_white: bool) -> usize {
    let base = match piece_type {
        PieceType::Pawn => 0,
        PieceType::Knight => 2,
        PieceType::Bishop => 4,
        PieceType::Rook => 6,
        PieceType::Queen => 8,
        PieceType::King => 10,
        PieceType::None => panic!("None piece in Zobrist hash"),
    };
    base + if is_white { 1 } else { 0 }
}

fn zobrist_hash(board: &ChessBoard, table: &[u64; ZOBRIST_SIZE]) -> u64 {
    let mut hash = 0u64;

    for sq in 0..64u16 {
        if let Some(piece) = board.get_piece_at_square(sq) {
            let idx = piece_zobrist_index(piece.piece_type(), piece.is_white());
            hash ^= table[idx * 64 + sq as usize];
        }
    }

    let cr = board.castling_rights;
    if cr & 0b1000 != 0 { hash ^= table[ZOBRIST_CASTLING_WK]; }
    if cr & 0b0100 != 0 { hash ^= table[ZOBRIST_CASTLING_WQ]; }
    if cr & 0b0010 != 0 { hash ^= table[ZOBRIST_CASTLING_BK]; }
    if cr & 0b0001 != 0 { hash ^= table[ZOBRIST_CASTLING_BQ]; }

    // En passant: include the file hash when the previous move was a double pawn push.
    if let Some(last) = board.get_last_move() {
        if last.has_flag(ChessMove::PAWN_TWO_UP_FLAG) {
            let ep_file = last.target_square() % 8;
            hash ^= table[ZOBRIST_EP_OFFSET + ep_file as usize];
        }
    }

    if !board.is_white_active() {
        hash ^= table[ZOBRIST_SIDE_TO_MOVE];
    }

    hash
}

/// Opening lines expressed as UCI move strings (e.g. "e2e4").
/// Every position encountered while replaying a line gets the next move
/// recorded as a book response.
const OPENING_LINES: &[&[&str]] = &[
    // ── OPEN GAME: 1.e4 e5 ──────────────────────────────────────────────────
    // Ruy Lopez
    &["e2e4", "e7e5", "g1f3", "b8c6", "f1b5"],
    &["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6"],
    &["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6", "e1g1", "f8e7"],
    // Italian / Giuoco Piano
    &["e2e4", "e7e5", "g1f3", "b8c6", "f1c4"],
    &["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5"],
    &["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5", "c2c3", "g8f6"],
    // Two Knights
    &["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6"],
    // Petrov
    &["e2e4", "e7e5", "g1f3", "g8f6", "f3e5", "d7d6", "e5f3", "f6e4"],
    &["e2e4", "e7e5", "g1f3", "g8f6", "d2d4"],
    // Vienna
    &["e2e4", "e7e5", "b1c3", "g8f6", "f2f4"],
    &["e2e4", "e7e5", "b1c3", "b8c6", "g1f3"],

    // ── SICILIAN: 1.e4 c5 ───────────────────────────────────────────────────
    &["e2e4", "c7c5", "g1f3"],
    &["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4"],
    &["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3"],
    &["e2e4", "c7c5", "g1f3", "b8c6", "d2d4", "c5d4", "f3d4"],
    &["e2e4", "c7c5", "g1f3", "b8c6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3"],
    &["e2e4", "c7c5", "g1f3", "e7e6", "d2d4", "c5d4", "f3d4"],
    &["e2e4", "c7c5", "b1c3", "b8c6", "g1f3", "g7g6"],

    // ── FRENCH: 1.e4 e6 ─────────────────────────────────────────────────────
    &["e2e4", "e7e6", "d2d4"],
    &["e2e4", "e7e6", "d2d4", "d7d5", "b1c3"],
    &["e2e4", "e7e6", "d2d4", "d7d5", "b1c3", "g8f6", "e4e5"],
    &["e2e4", "e7e6", "d2d4", "d7d5", "b1c3", "f8b4"],
    &["e2e4", "e7e6", "d2d4", "d7d5", "e4e5", "c7c5"],
    &["e2e4", "e7e6", "d2d4", "d7d5", "b1d2", "g8f6"],

    // ── SCANDINAVIAN: 1.e4 d5 ───────────────────────────────────────────────
    &["e2e4", "d7d5", "e4d5"],
    &["e2e4", "d7d5", "e4d5", "d8d5", "b1c3"],
    &["e2e4", "d7d5", "e4d5", "d8d5", "b1c3", "d5a5"],
    &["e2e4", "d7d5", "e4d5", "g8f6"],

    // ── CARO-KANN: 1.e4 c6 ──────────────────────────────────────────────────
    &["e2e4", "c7c6", "d2d4", "d7d5"],
    &["e2e4", "c7c6", "d2d4", "d7d5", "b1c3", "d5e4"],
    &["e2e4", "c7c6", "d2d4", "d7d5", "e4e5", "c8f5"],

    // ── ALEKHINE: 1.e4 Nf6 ──────────────────────────────────────────────────
    &["e2e4", "g8f6", "e4e5", "f6d5", "d2d4"],

    // ── PIRC / MODERN: 1.e4 g6 ──────────────────────────────────────────────
    &["e2e4", "g7g6", "d2d4", "f8g7", "b1c3"],

    // ── QUEEN'S GAMBIT: 1.d4 d5 2.c4 ────────────────────────────────────────
    &["d2d4", "d7d5", "c2c4"],
    &["d2d4", "d7d5", "c2c4", "e7e6", "b1c3"],
    &["d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "g8f6", "g1f3"],
    &["d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "g8f6", "c1g5"],
    // Slav
    &["d2d4", "d7d5", "c2c4", "c7c6", "b1c3", "g8f6"],
    &["d2d4", "d7d5", "c2c4", "c7c6", "b1c3", "g8f6", "g1f3"],
    // QGA
    &["d2d4", "d7d5", "c2c4", "d5c4", "g1f3"],

    // ── KING'S INDIAN: 1.d4 Nf6 2.c4 g6 ─────────────────────────────────────
    &["d2d4", "g8f6", "c2c4"],
    &["d2d4", "g8f6", "c2c4", "g7g6", "b1c3"],
    &["d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "f8g7", "e2e4"],
    &["d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "f8g7", "e2e4", "d7d6", "g1f3"],

    // ── NIMZO-INDIAN: 1.d4 Nf6 2.c4 e6 3.Nc3 Bb4 ────────────────────────────
    &["d2d4", "g8f6", "c2c4", "e7e6", "b1c3"],
    &["d2d4", "g8f6", "c2c4", "e7e6", "b1c3", "f8b4"],
    &["d2d4", "g8f6", "c2c4", "e7e6", "b1c3", "f8b4", "e2e3"],
    &["d2d4", "g8f6", "c2c4", "e7e6", "b1c3", "f8b4", "d1c2"],

    // ── QUEEN'S INDIAN: 1.d4 Nf6 2.c4 e6 3.Nf3 b6 ───────────────────────────
    &["d2d4", "g8f6", "c2c4", "e7e6", "g1f3"],
    &["d2d4", "g8f6", "c2c4", "e7e6", "g1f3", "b7b6"],
    &["d2d4", "g8f6", "c2c4", "e7e6", "g1f3", "b7b6", "g2g3"],

    // ── GRÜNFELD: 1.d4 Nf6 2.c4 g6 3.Nc3 d5 ─────────────────────────────────
    &["d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "d7d5"],
    &["d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "d7d5", "c4d5"],

    // ── DUTCH DEFENSE: 1.d4 f5 ───────────────────────────────────────────────
    &["d2d4", "f7f5", "g2g3", "g8f6", "f1g2"],
    &["d2d4", "f7f5", "c2c4", "g8f6", "g2g3"],

    // ── LONDON SYSTEM ─────────────────────────────────────────────────────────
    &["d2d4", "d7d5", "g1f3", "g8f6", "c1f4"],
    &["d2d4", "d7d5", "g1f3", "g8f6", "c1f4", "e7e6", "e2e3"],
    &["d2d4", "g8f6", "g1f3", "d7d5", "c1f4", "e7e6"],
];

/// A compact opening book built from hard-coded GM-repertoire lines.
/// The book maps each encountered position (by Zobrist hash) to a list of
/// known responses. On a cache hit, one response is chosen uniformly at random.
#[derive(Clone)]
pub struct OpeningBook {
    table: [u64; ZOBRIST_SIZE],
    /// Zobrist hash → [(from_square, to_square)]
    positions: HashMap<u64, Vec<(u16, u16)>>,
}

impl OpeningBook {
    /// Build the book by replaying every opening line from the start position.
    /// Requires the move generator so that moves get correct flags (castling,
    /// en-passant, double-pawn-push) before `make_move` is called.
    pub fn build(conductor: &PieceConductor) -> Self {
        let table = make_zobrist_table();
        let mut positions: HashMap<u64, Vec<(u16, u16)>> = HashMap::new();

        for &line in OPENING_LINES {
            let mut board = ChessBoard::new();

            for &uci in line {
                let hash = zobrist_hash(&board, &table);
                let parsed = ChessMove::from_san(uci);
                let from = parsed.start_square();
                let to = parsed.target_square();

                let entry = positions.entry(hash).or_default();
                if !entry.contains(&(from, to)) {
                    entry.push((from, to));
                }

                // Advance the board using the legal move (picks up correct flags).
                let is_white = board.is_white_active();
                let legal = get_all_legal_moves_for_color(&mut board, conductor, is_white);
                if let Some(mut m) = legal
                    .into_iter()
                    .find(|m| m.start_square() == from && m.target_square() == to)
                {
                    board.make_move(&mut m);
                } else {
                    eprintln!("Opening book: move {uci} not legal — line truncated");
                    break;
                }
            }
        }

        let n_positions = positions.len();
        println!("Opening book built: {n_positions} positions");
        OpeningBook { table, positions }
    }

    /// Probe the book for the current position.
    /// Returns (from_square, to_square) if in book, otherwise None.
    pub fn probe(&self, board: &ChessBoard) -> Option<(u16, u16)> {
        let hash = zobrist_hash(board, &self.table);
        let moves = self.positions.get(&hash)?;
        moves.choose(&mut rand::thread_rng()).copied()
    }
}
