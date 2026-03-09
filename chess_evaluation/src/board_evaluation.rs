use chess_board::ChessBoard;
use chess_foundation::Bitboard;

use crate::piece_tables::{
    bishop_table_value, is_passed_pawn, king_table_value, knight_table_value,
    passed_pawn_bonus, pawn_table_value, queen_table_value, rook_table_value,
};

const PAWN_VALUE: i32 = 100;
const KNIGHT_VALUE: i32 = 320;
const BISHOP_VALUE: i32 = 330;
const ROOK_VALUE: i32 = 500;
const QUEEN_VALUE: i32 = 900;
const KING_VALUE: i32 = 20000;

fn count(bb: Bitboard) -> i32 {
    bb.count_ones() as i32
}

/// Compute a game-phase weight.
/// Returns 0 (full endgame) … 256 (full middlegame).
fn endgame_weight(chess_board: &ChessBoard) -> i32 {
    let queens  = count(chess_board.get_queens())  * 4;
    let rooks   = count(chess_board.get_rooks())   * 2;
    let bishops = count(chess_board.get_bishops()) * 1;
    let knights = count(chess_board.get_knights()) * 1;
    let phase   = (queens + rooks + bishops + knights).min(24); // 24 = opening
    // Map 24→0 (mg), 0→256 (eg)
    ((24 - phase) * 256) / 24
}

/// Iterate a bitboard, calling `f(square)` for each set bit.
#[inline]
fn for_each_sq(mut bb: Bitboard, mut f: impl FnMut(usize)) {
    while bb != Bitboard::default() {
        f(bb.pop_lsb());
    }
}

/// Manhattan distance of a king from the nearest centre square (d4/d5/e4/e5).
/// Returns 0 when on a centre square, up to 6 when in a corner.
fn king_center_distance(sq: usize) -> i32 {
    let rank = (sq / 8) as i32;
    let file = (sq % 8) as i32;
    let rank_dist = (rank - 3).abs().min((rank - 4).abs());
    let file_dist = (file - 3).abs().min((file - 4).abs());
    rank_dist + file_dist
}

/// Manhattan distance between two squares (max 14).
fn manhattan(sq1: usize, sq2: usize) -> i32 {
    let r1 = (sq1 / 8) as i32;
    let f1 = (sq1 % 8) as i32;
    let r2 = (sq2 / 8) as i32;
    let f2 = (sq2 % 8) as i32;
    (r1 - r2).abs() + (f1 - f2).abs()
}

/// Mop-up bonus for the winning side:
///  - push the losing king toward a corner (high center-distance = good)
///  - keep the winning king close to the losing king
///
/// `material_score` is from white's perspective.
/// Returns a value to ADD to the overall score (positive = good for white).
fn mop_up(
    material_score: i32,
    white_king_sq: usize,
    black_king_sq: usize,
    eg_weight: i32,
) -> i32 {
    // Only meaningful in the endgame and when one side is clearly ahead.
    if eg_weight < 80 || material_score.abs() < 150 {
        return 0;
    }

    // corner_push: reward driving the losing king to a corner/edge (0..=60)
    // proximity:   reward the winning king being adjacent to the losing king (0..=56)
    let (corner_push, proximity) = if material_score > 0 {
        // White is winning — push black king to a corner
        let corner_push = king_center_distance(black_king_sq) * 10;
        let proximity = (14 - manhattan(white_king_sq, black_king_sq)) * 4;
        (corner_push, proximity)
    } else {
        // Black is winning — push white king to a corner
        let corner_push = king_center_distance(white_king_sq) * 10;
        let proximity = (14 - manhattan(black_king_sq, white_king_sq)) * 4;
        (-(corner_push), -(proximity))
    };

    // Scale by how far into the endgame we are
    (corner_push + proximity) * eg_weight / 256
}

/// Evaluates the chess board and returns an absolute score:
/// positive = white is ahead, negative = black is ahead.
pub fn evaluate_board(chess_board: &ChessBoard) -> i32 {
    let white = chess_board.get_white();
    let black = chess_board.get_black();
    let pawns   = chess_board.get_pawns();
    let knights = chess_board.get_knights();
    let bishops = chess_board.get_bishops();
    let rooks   = chess_board.get_rooks();
    let queens  = chess_board.get_queens();
    let kings   = chess_board.get_kings();

    let eg_weight = endgame_weight(chess_board);

    let white_pawns_bb = (white & pawns).0;
    let black_pawns_bb = (black & pawns).0;

    let mut score = 0i32;

    // --- Material ---
    score += count(white & pawns)   * PAWN_VALUE;
    score += count(white & knights) * KNIGHT_VALUE;
    score += count(white & bishops) * BISHOP_VALUE;
    score += count(white & rooks)   * ROOK_VALUE;
    score += count(white & queens)  * QUEEN_VALUE;
    score += count(white & kings)   * KING_VALUE;

    score -= count(black & pawns)   * PAWN_VALUE;
    score -= count(black & knights) * KNIGHT_VALUE;
    score -= count(black & bishops) * BISHOP_VALUE;
    score -= count(black & rooks)   * ROOK_VALUE;
    score -= count(black & queens)  * QUEEN_VALUE;
    score -= count(black & kings)   * KING_VALUE;

    // --- White piece-square tables ---
    for_each_sq(white & pawns, |sq| {
        score += pawn_table_value(sq, true);
        if is_passed_pawn(sq, black_pawns_bb, true) {
            score += passed_pawn_bonus(sq, true);
        }
    });
    for_each_sq(white & knights, |sq| score += knight_table_value(sq, true));
    for_each_sq(white & bishops, |sq| score += bishop_table_value(sq, true));
    for_each_sq(white & rooks,   |sq| score += rook_table_value(sq, true));
    for_each_sq(white & queens,  |sq| score += queen_table_value(sq, true));
    for_each_sq(white & kings,   |sq| score += king_table_value(sq, true, eg_weight));

    // --- Black piece-square tables ---
    for_each_sq(black & pawns, |sq| {
        score -= pawn_table_value(sq, false);
        if is_passed_pawn(sq, white_pawns_bb, false) {
            score -= passed_pawn_bonus(sq, false);
        }
    });
    for_each_sq(black & knights, |sq| score -= knight_table_value(sq, false));
    for_each_sq(black & bishops, |sq| score -= bishop_table_value(sq, false));
    for_each_sq(black & rooks,   |sq| score -= rook_table_value(sq, false));
    for_each_sq(black & queens,  |sq| score -= queen_table_value(sq, false));
    let mut black_king_sq = 0usize;
    for_each_sq(black & kings, |sq| {
        score -= king_table_value(sq, false, eg_weight);
        black_king_sq = sq;
    });

    // --- Mop-up: drive the losing king to the corner in winning endgames ---
    let white_king_sq = (white & kings).0.trailing_zeros() as usize;
    // material_score excludes king values (both sides have one, they cancel out)
    let material_score = score - count(white & kings) * KING_VALUE + count(black & kings) * KING_VALUE;
    score += mop_up(material_score, white_king_sq, black_king_sq, eg_weight);

    score
}
