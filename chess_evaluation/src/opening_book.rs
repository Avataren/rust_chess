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

use crate::opening_book_data::{OPENING_LINES, NAMED_OPENINGS};


/// A compact opening book built from hard-coded GM-repertoire lines.
/// The book maps each encountered position (by Zobrist hash) to a list of
/// known responses. On a cache hit, one response is chosen uniformly at random.
#[derive(Clone)]
pub struct OpeningBook {
    table: [u64; ZOBRIST_SIZE],
    /// Zobrist hash → [(from_square, to_square)]
    positions: HashMap<u64, Vec<(u16, u16)>>,
    /// Zobrist hash → opening name for named landmark positions
    names: HashMap<u64, &'static str>,
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

        // Build named-position map: replay each named line, store the final hash.
        let mut names: HashMap<u64, &'static str> = HashMap::new();
        for &(name, line) in NAMED_OPENINGS {
            let mut board = ChessBoard::new();
            for &uci in line {
                let is_white = board.is_white_active();
                let parsed = ChessMove::from_san(uci);
                let from = parsed.start_square();
                let to = parsed.target_square();
                let legal = get_all_legal_moves_for_color(&mut board, conductor, is_white);
                if let Some(mut m) = legal
                    .into_iter()
                    .find(|m| m.start_square() == from && m.target_square() == to)
                {
                    board.make_move(&mut m);
                } else {
                    eprintln!("Named opening '{name}': move {uci} not legal");
                    break;
                }
            }
            names.insert(zobrist_hash(&board, &table), name);
        }

        let n_positions = positions.len();
        let n_names = names.len();
        println!("Opening book: {n_positions} book positions, {n_names} named openings");
        OpeningBook { table, positions, names }
    }

    /// Probe the book for the current position.
    /// Returns (from_square, to_square) if in book, otherwise None.
    pub fn probe(&self, board: &ChessBoard) -> Option<(u16, u16)> {
        let hash = zobrist_hash(board, &self.table);
        let moves = self.positions.get(&hash)?;
        moves.choose(&mut rand::thread_rng()).copied()
    }

    /// Return the opening name if the current position is a named landmark.
    /// Returns `None` between landmarks — callers should keep the last name.
    pub fn probe_name(&self, board: &ChessBoard) -> Option<&'static str> {
        let hash = zobrist_hash(board, &self.table);
        self.names.get(&hash).copied()
    }
}
