use chess_board::ChessBoard;
use chess_foundation::Bitboard;
use move_generator::piece_conductor::PieceConductor;
use std::cell::UnsafeCell;

use crate::piece_tables::{
    eg_bishop_table, eg_king_table, eg_knight_table, eg_pawn_table, eg_queen_table,
    eg_rook_table, is_passed_pawn, mg_bishop_table, mg_king_table, mg_knight_table,
    mg_pawn_table, mg_queen_table, mg_rook_table, passed_pawn_bonus_eg, passed_pawn_bonus_mg,
};

// ── Pawn hash table ──────────────────────────────────────────────────────────
//
// Pawn structure evaluation is expensive (PSTs + doubled/isolated + passed
// pawn detection) but depends only on the two pawn bitboards, which change
// infrequently during a search.  Caching behind a small per-thread table
// avoids recomputing the same pawn eval at every node.

const PAWN_TABLE_SIZE: usize = 1 << 14; // 16k entries (~1 MB per thread)

/// A single pawn hash entry.  Stores all pawn-only MG/EG contributions
/// (PSTs, structure penalties, passed pawn base bonuses) plus the passed-pawn
/// bitboards needed to compute king-proximity bonuses at lookup time.
#[derive(Clone, Copy)]
struct PawnHashEntry {
    key:          u64,
    pawn_mg:      i32,  // white pawn PSTs MG − black pawn PSTs MG
    pawn_eg:      i32,  // white pawn PSTs EG − black pawn PSTs EG
    struct_score: i32,  // black structure penalty − white structure penalty (white's POV)
    pass_w_mg:    i32,  // Σ passed_pawn_bonus_mg for white passers
    pass_w_eg:    i32,
    pass_b_mg:    i32,  // Σ passed_pawn_bonus_mg for black passers
    pass_b_eg:    i32,
    white_passers: u64,
    black_passers: u64,
}

impl PawnHashEntry {
    const EMPTY: Self = Self {
        key: 0, pawn_mg: 0, pawn_eg: 0, struct_score: 0,
        pass_w_mg: 0, pass_w_eg: 0, pass_b_mg: 0, pass_b_eg: 0,
        white_passers: 0, black_passers: 0,
    };
}

struct PawnHashTable {
    entries: Box<[PawnHashEntry; PAWN_TABLE_SIZE]>,
}

impl PawnHashTable {
    fn new() -> Self {
        Self { entries: Box::new([PawnHashEntry::EMPTY; PAWN_TABLE_SIZE]) }
    }

    #[inline(always)]
    fn probe(&self, key: u64) -> Option<&PawnHashEntry> {
        let e = &self.entries[key as usize & (PAWN_TABLE_SIZE - 1)];
        if e.key == key { Some(e) } else { None }
    }

    #[inline(always)]
    fn store(&mut self, entry: PawnHashEntry) {
        self.entries[entry.key as usize & (PAWN_TABLE_SIZE - 1)] = entry;
    }
}

// UnsafeCell allows mutation via a shared reference inside thread_local!
// This is sound because thread_local storage is never shared across threads.
struct UnsafePawnTable(UnsafeCell<PawnHashTable>);
unsafe impl Sync for UnsafePawnTable {}

thread_local! {
    static PAWN_TABLE: UnsafePawnTable = UnsafePawnTable(UnsafeCell::new(PawnHashTable::new()));
}

/// Compute a pawn-only Zobrist hash from the two pawn bitboards.
#[inline(always)]
fn pawn_key(white_pawns: u64, black_pawns: u64) -> u64 {
    white_pawns.wrapping_mul(0x9E3779B97F4A7C15)
        ^ black_pawns.wrapping_mul(0x517CC1B727220A95)
}

// PeSTO tapered piece values
const MG_PAWN_VALUE:   i32 =  82;
const MG_KNIGHT_VALUE: i32 = 337;
const MG_BISHOP_VALUE: i32 = 365;
const MG_ROOK_VALUE:   i32 = 477;
const MG_QUEEN_VALUE:  i32 = 1025;

const EG_PAWN_VALUE:   i32 =  94;
const EG_KNIGHT_VALUE: i32 = 281;
const EG_BISHOP_VALUE: i32 = 297;
const EG_ROOK_VALUE:   i32 = 512;
const EG_QUEEN_VALUE:  i32 = 936;

const ISOLATED_PAWN_PENALTY: i32 = 15;
const DOUBLED_PAWN_PENALTY:  i32 = 15;

const MG_BISHOP_PAIR_BONUS: i32 = 30;
const EG_BISHOP_PAIR_BONUS: i32 = 50;

const MG_ROOK_OPEN_FILE:      i32 = 20;
const EG_ROOK_OPEN_FILE:      i32 = 15;
const MG_ROOK_SEMI_OPEN_FILE: i32 = 10;
const EG_ROOK_SEMI_OPEN_FILE: i32 = 8;

// Stockfish-style king proximity: enemy king far = bonus, own king far = penalty.
// Rank weight: 5*rank - 13 (clamped ≥ 0), so ranks 0-2 contribute nothing.
// Asymmetric weights: enemy distance matters ~2× more than friendly (SF uses 4.75:2).
const PASSER_KING_ENEMY_WT:  i32 = 2; // cp per unit of enemy king distance (×rank_weight)
const PASSER_KING_FRIEND_WT: i32 = 1; // cp per unit of friendly king distance (×rank_weight)

const ROOK_BEHIND_PASSER_EG: i32 = 20;

// Rook on 7th rank bonus (MG / EG).  Only awarded when the enemy king is on
// the back rank, making the 7th-rank rook maximally threatening.
const MG_ROOK_ON_SEVENTH: i32 = 25;
const EG_ROOK_ON_SEVENTH: i32 = 35;

// Knight outpost bonus (MG / EG).  A knight on a square where it cannot be
// chased by an enemy pawn and is protected by a friendly pawn is very strong.
const MG_KNIGHT_OUTPOST: i32 = 30;
const EG_KNIGHT_OUTPOST: i32 = 20;

// Piece mobility weights (cp per reachable square).
// Calibrated against SF14 MobilityBonus tables (averaged over typical square counts):
//   Knight: max ~37cp MG (8 sq), Bishop: max ~96cp MG (13 sq), Rook: max ~67cp MG (14 sq)
//   Queen:  max ~119cp MG (27 sq) — included here with a modest per-square weight.
// The per-square weights deliberately undercut SF14's peaks: SF uses non-linear
// tables where the bonus saturates; our linear approximation needs a lower slope
// to avoid over-rewarding maximum mobility.
const KNIGHT_MOBILITY_WEIGHT: i32 = 4; // ~32cp at 8 squares
const BISHOP_MOBILITY_WEIGHT: i32 = 3; // ~39cp at 13 squares
const ROOK_MOBILITY_WEIGHT:   i32 = 2; // ~28cp at 14 squares (was 1; SF14 rook peaks at 67)
const QUEEN_MOBILITY_WEIGHT:  i32 = 2; // ~54cp at 27 squares — conservative to avoid instability

// King safety — pawn shield (only when castled) + attack counting.
const KING_SHIELD_MISSING:  i32 = 15;
const KING_SHIELD_ADVANCED: i32 =  5;

// Attack weight per piece type attacking king zone.
const KNIGHT_ATTACK_WEIGHT: i32 = 2;
const BISHOP_ATTACK_WEIGHT: i32 = 2;
const ROOK_ATTACK_WEIGHT:   i32 = 3;
const QUEEN_ATTACK_WEIGHT:  i32 = 5;

/// Non-linear safety penalty indexed by total attack weight.
/// Approximates SF14's quadratic king-danger curve (danger² / 4096).
/// Ramps slowly for a lone minor piece, steeply when queen + support arrives.
/// Values in centipawns, applied to MG score only.
#[rustfmt::skip]
const SAFETY_TABLE: [i32; 20] = [
//   0    1    2    3    4    5    6    7    8    9
     0,   0,   4,  14,  32,  62, 100, 155, 222, 300,
//  10   11   12   13   14   15   16   17   18   19
   390, 490, 590, 685, 770, 845, 905, 955, 995, 1025,
];

const FILE_MASKS: [u64; 8] = [
    0x0101010101010101, // a-file
    0x0202020202020202, // b-file
    0x0404040404040404, // c-file
    0x0808080808080808, // d-file
    0x1010101010101010, // e-file
    0x2020202020202020, // f-file
    0x4040404040404040, // g-file
    0x8080808080808080, // h-file
];

#[inline(always)]
fn count(bb: Bitboard) -> i32 {
    bb.count_ones() as i32
}

/// Game phase weight: 0 (full endgame) … 24 (full middlegame).
#[inline]
fn game_phase(chess_board: &ChessBoard) -> i32 {
    let queens  = count(chess_board.get_queens())  * 4;
    let rooks   = count(chess_board.get_rooks())   * 2;
    let bishops = count(chess_board.get_bishops()) * 1;
    let knights = count(chess_board.get_knights()) * 1;
    (queens + rooks + bishops + knights).min(24)
}

/// Iterate a bitboard, calling `f(square)` for each set bit.
#[inline]
fn for_each_sq(mut bb: Bitboard, mut f: impl FnMut(usize)) {
    while bb != Bitboard::default() {
        f(bb.pop_lsb());
    }
}

/// Manhattan distance between two squares (max 14).
#[inline]
fn manhattan(sq1: usize, sq2: usize) -> i32 {
    let r1 = (sq1 / 8) as i32;
    let f1 = (sq1 % 8) as i32;
    let r2 = (sq2 / 8) as i32;
    let f2 = (sq2 % 8) as i32;
    (r1 - r2).abs() + (f1 - f2).abs()
}

/// Chebyshev distance between two squares (king metric: max of rank/file deltas).
#[inline]
fn chebyshev(sq1: usize, sq2: usize) -> i32 {
    let r1 = (sq1 / 8) as i32; let f1 = (sq1 % 8) as i32;
    let r2 = (sq2 / 8) as i32; let f2 = (sq2 % 8) as i32;
    (r1 - r2).abs().max((f1 - f2).abs())
}

/// Manhattan distance of a king from the nearest centre square (d4/d5/e4/e5).
#[inline]
fn king_center_distance(sq: usize) -> i32 {
    let rank = (sq / 8) as i32;
    let file = (sq % 8) as i32;
    let rank_dist = (rank - 3).abs().min((rank - 4).abs());
    let file_dist = (file - 3).abs().min((file - 4).abs());
    rank_dist + file_dist
}

/// True if `sq` is protected by a friendly pawn.
/// For white: a pawn at (rank-1, file±1); for black: (rank+1, file±1).
#[inline]
fn is_protected_by_pawn(sq: usize, own_pawns: u64, is_white: bool) -> bool {
    let file = sq % 8;
    let rank = sq / 8;
    if is_white {
        if rank == 0 { return false; }
        let prev_rank_base = (rank - 1) * 8;
        let left  = if file > 0 { own_pawns & (1u64 << (prev_rank_base + file - 1)) } else { 0 };
        let right = if file < 7 { own_pawns & (1u64 << (prev_rank_base + file + 1)) } else { 0 };
        (left | right) != 0
    } else {
        if rank == 7 { return false; }
        let next_rank_base = (rank + 1) * 8;
        let left  = if file > 0 { own_pawns & (1u64 << (next_rank_base + file - 1)) } else { 0 };
        let right = if file < 7 { own_pawns & (1u64 << (next_rank_base + file + 1)) } else { 0 };
        (left | right) != 0
    }
}

/// Pawn attack squares for a side (squares that side's pawns currently attack).
///
/// Stockfish excludes these from the opponent's mobility area — squares
/// attacked by pawns are dangerous for enemy pieces and should not count as
/// "reachable" in the mobility score.
#[inline]
fn pawn_attacks(pawns: u64, is_white: bool) -> u64 {
    const NOT_A_FILE: u64 = 0xfefe_fefe_fefe_fefe;
    const NOT_H_FILE: u64 = 0x7f7f_7f7f_7f7f_7f7f;
    if is_white {
        ((pawns << 7) & NOT_H_FILE) | ((pawns << 9) & NOT_A_FILE)
    } else {
        ((pawns >> 7) & NOT_A_FILE) | ((pawns >> 9) & NOT_H_FILE)
    }
}

/// Piece mobility score for one side.
///
/// Counts reachable squares (excluding own pieces AND squares attacked by enemy
/// pawns) per piece type and multiplies by the type-specific weight.
/// Excluding pawn-attacked squares (Stockfish practice) prevents overvaluing
/// pieces that appear mobile but land on controlled squares.
fn mobility_score(
    conductor: &PieceConductor,
    knights: Bitboard,
    bishops: Bitboard,
    rooks:   Bitboard,
    queens:  Bitboard,
    own_pieces: Bitboard,
    occupied:   Bitboard,
    enemy_pawn_attacks: u64,
) -> i32 {
    // Safe squares: not occupied by own pieces, not attacked by enemy pawns.
    let safe = !(own_pieces.0 | enemy_pawn_attacks);
    let mut score = 0i32;

    for_each_sq(knights, |sq| {
        let attacks = conductor.knight_lut[sq].0 & safe;
        score += attacks.count_ones() as i32 * KNIGHT_MOBILITY_WEIGHT;
    });

    for_each_sq(bishops, |sq| {
        let attacks = conductor.get_bishop_attacks(sq, Bitboard(0), occupied).0 & safe;
        score += attacks.count_ones() as i32 * BISHOP_MOBILITY_WEIGHT;
    });

    for_each_sq(rooks, |sq| {
        let attacks = conductor.get_rook_attacks(sq, Bitboard(0), occupied).0 & safe;
        score += attacks.count_ones() as i32 * ROOK_MOBILITY_WEIGHT;
    });

    // Queen mobility: conservative weight (2cp/sq) to capture activity bonus
    // while avoiding the instability that heavier weights caused previously.
    for_each_sq(queens, |sq| {
        let rook_part   = conductor.get_rook_attacks(sq, Bitboard(0), occupied).0 & safe;
        let bishop_part = conductor.get_bishop_attacks(sq, Bitboard(0), occupied).0 & safe;
        let attacks = rook_part | bishop_part;
        score += attacks.count_ones() as i32 * QUEEN_MOBILITY_WEIGHT;
    });

    score
}

#[cfg(test)]
mod tests {
    use super::*;
    use chess_board::ChessBoard;
    use move_generator::piece_conductor::PieceConductor;

    fn eval(fen: &str) -> i32 {
        let mut board = ChessBoard::new();
        board.set_from_fen(fen);
        evaluate_board(&board, &PieceConductor::new())
    }

    // ── Symmetry ────────────────────────────────────────────────────────────

    /// Starting position is perfectly symmetric — score must be near zero.
    #[test]
    fn starting_position_is_near_zero() {
        let score = eval("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        assert!(score.abs() < 30,
            "Starting position must be near-zero, got {score}");
    }

    // ── Mobility sanity ─────────────────────────────────────────────────────

    /// A knight in the centre (c3) should score higher than one in the corner (a1).
    #[test]
    fn central_knight_scores_higher_than_corner_knight() {
        let central = eval("4k3/8/8/8/8/2N5/8/4K3 w - - 0 1");
        let corner  = eval("4k3/8/8/8/8/8/8/N3K3 w - - 0 1");
        assert!(central > corner,
            "Central knight ({central}) must score higher than corner knight ({corner})");
    }

    /// A bishop on a central active square scores higher than one in the corner.
    /// Equal material (bishop + king vs bishop + king), pure mobility/PST difference.
    #[test]
    fn open_diagonal_bishop_scores_higher() {
        let central = eval("4k3/8/8/8/2B5/8/8/4K3 w - - 0 1"); // Bc4: active, high mobility
        let corner  = eval("4k3/8/8/8/8/8/8/B3K3 w - - 0 1");  // Ba1: corner, low mobility
        assert!(central > corner,
            "Central bishop ({central}) should score higher than corner bishop ({corner})");
    }

    /// Mobility must be symmetric for mirror-image positions.
    #[test]
    fn mobility_is_symmetric_for_equal_positions() {
        let score = eval("4k3/8/2n5/8/8/2N5/8/4K3 w - - 0 1");
        assert!(score.abs() < 30,
            "Symmetric knight positions must give near-zero score, got {score}");
    }

    /// Mobility must not override material: a pawn-up position must score better
    /// than the same position minus the pawn, even if the pawnless side has
    /// slightly higher piece mobility.
    #[test]
    fn extra_pawn_beats_extra_mobility() {
        // White knight + pawn vs white knight only (black knight symmetrically placed)
        let with_pawn    = eval("4k3/8/2n5/8/2NP4/8/8/4K3 w - - 0 1");
        let without_pawn = eval("4k3/8/2n5/8/2N5/8/8/4K3 w - - 0 1");
        assert!(with_pawn > without_pawn,
            "Extra pawn ({with_pawn}) must beat no pawn ({without_pawn})");
    }

    /// Rook on an open file must score higher than the same material with the rook
    /// on a file blocked by its own pawn.  Equal material: rook + pawn + king each.
    #[test]
    fn rook_open_file_still_rewarded() {
        // Ra1, Pa4 — a-file is open for the rook
        let open   = eval("4k3/8/8/8/P7/8/8/R3K3 w - - 0 1");
        // Rb1, Pb2 — rook's file is blocked by its own pawn
        let closed = eval("4k3/8/8/8/8/8/1P6/1R2K3 w - - 0 1");
        assert!(open > closed,
            "Rook on open file ({open}) must score higher than rook blocked by own pawn ({closed})");
    }

    /// Mobility bonus per-position must be bounded: starting position mobility
    /// delta should not exceed ~100cp (sanity check against inflated weights).
    #[test]
    fn mobility_bonus_is_not_inflated() {
        // Compute eval with all pieces present; the score should be near-zero
        // for the starting position (symmetry), not skewed by mobility.
        let score = eval("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        assert!(score.abs() < 50,
            "Mobility must not inflate the starting position score beyond ±50cp, got {score}");
    }

    // ── Rook on 7th ─────────────────────────────────────────────────────────

    #[test]
    fn rook_on_7th_ranks_higher_than_6th() {
        let on_7th = eval("6k1/R7/8/8/8/8/8/4K3 w - - 0 1");
        let on_6th = eval("6k1/8/R7/8/8/8/8/4K3 w - - 0 1");
        assert!(on_7th > on_6th,
            "Rook on 7th rank ({on_7th}) must score higher than 6th ({on_6th})");
    }

    #[test]
    fn rook_on_7th_requires_king_on_8th() {
        let with_bonus    = eval("6k1/R7/8/8/8/8/8/4K3 w - - 0 1"); // black king on g8
        let without_bonus = eval("8/R7/8/8/4k3/8/8/4K3 w - - 0 1"); // black king on e5
        assert!(with_bonus > without_bonus,
            "7th-rank bonus ({with_bonus}) should only apply when king is on 8th ({without_bonus})");
    }

    // ── Knight outposts ─────────────────────────────────────────────────────

    #[test]
    fn knight_outpost_scores_higher_than_non_outpost() {
        // Knight on d5 (rank 4), protected by pawn on e4, no black pawns on c/d/e files
        let outpost     = eval("4k3/8/8/3N4/4P3/8/8/4K3 w - - 0 1");
        // Knight on d4 — below rank 3 threshold, no outpost awarded
        let non_outpost = eval("4k3/8/8/8/3NP3/8/8/4K3 w - - 0 1");
        assert!(outpost > non_outpost,
            "Outpost knight ({outpost}) must score higher than non-outpost ({non_outpost})");
    }

    #[test]
    fn outpost_not_awarded_when_enemy_pawn_attacks() {
        // Black has e6 pawn that attacks d5 — outpost not valid
        let chased  = eval("4k3/8/4p3/3N4/4P3/8/8/4K3 w - - 0 1");
        let outpost = eval("4k3/8/8/3N4/4P3/8/8/4K3 w - - 0 1");
        assert!(outpost > chased,
            "Outpost ({outpost}) must score higher when no enemy pawn attacks ({chased})");
    }

    #[test]
    fn outpost_requires_pawn_protection() {
        let unprotected = eval("4k3/8/8/3N4/8/8/8/4K3 w - - 0 1"); // no white pawn
        let protected   = eval("4k3/8/8/3N4/4P3/8/8/4K3 w - - 0 1"); // e4 protects d5
        assert!(protected > unprotected,
            "Protected outpost ({protected}) must score higher than unprotected ({unprotected})");
    }

    // ── Pawn hash cache correctness ──────────────────────────────────────────

    /// Calling evaluate_board twice on the same position must return the same
    /// score regardless of whether it's a cache miss or hit.
    #[test]
    fn pawn_hash_consistent_on_repeated_calls() {
        let c = PieceConductor::new();
        let mut board = ChessBoard::new();
        // Italian Game: typical middlegame pawn structure.
        board.set_from_fen("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQK2R b KQkq - 0 6");

        let s1 = evaluate_board(&board, &c); // cache miss — computes fresh
        let s2 = evaluate_board(&board, &c); // cache hit — reads stored value
        let s3 = evaluate_board(&board, &c); // hit again
        assert_eq!(s1, s2, "Cache hit must return same score as cache miss ({s1} vs {s2})");
        assert_eq!(s2, s3, "Repeated cache hits must be stable ({s2} vs {s3})");
    }

    /// Two positions with different pawn structures must produce different eval
    /// scores.  This catches hash collisions silently returning the wrong entry.
    #[test]
    fn pawn_hash_distinguishes_different_pawn_structures() {
        let c = PieceConductor::new();

        // Position A: symmetric starting pawns.
        let mut board_a = ChessBoard::new();
        board_a.set_from_fen("4k3/pppppppp/8/8/8/8/PPPPPPPP/4K3 w - - 0 1");

        // Position B: white has an advanced pawn on e5, black has none.
        let mut board_b = ChessBoard::new();
        board_b.set_from_fen("4k3/pppp1ppp/8/4P3/8/8/PPPP1PPP/4K3 w - - 0 1");

        let sa = evaluate_board(&board_a, &c);
        let sb = evaluate_board(&board_b, &c);
        // Scores must differ — they have different pawn structures.
        assert_ne!(sa, sb,
            "Different pawn structures must produce different eval scores (both got {sa})");
    }

    /// After changing the pawn structure (simulated by evaluating a different
    /// FEN after the first), the eval must reflect the new pawn configuration,
    /// not the cached result from the previous position.
    #[test]
    fn pawn_hash_updates_after_pawn_structure_change() {
        let c = PieceConductor::new();

        // First: position with doubled white pawns on the e-file.
        let mut board1 = ChessBoard::new();
        board1.set_from_fen("4k3/8/8/8/4P3/4P3/8/4K3 w - - 0 1");
        let score_doubled = evaluate_board(&board1, &c);

        // Then: same material but no doubled pawns.
        let mut board2 = ChessBoard::new();
        board2.set_from_fen("4k3/8/8/8/4P3/3P4/8/4K3 w - - 0 1");
        let score_normal = evaluate_board(&board2, &c);

        // Doubled pawns carry a penalty, so the doubled position must score lower.
        assert!(score_doubled < score_normal,
            "Doubled pawns ({score_doubled}) must score lower than normal ({score_normal})");

        // Now re-evaluate board1 — must still return the original (correct) value,
        // not the cached value from board2.
        let score_doubled_again = evaluate_board(&board1, &c);
        assert_eq!(score_doubled, score_doubled_again,
            "Re-evaluating the doubled-pawn position must return the same score \
             ({score_doubled} vs {score_doubled_again}), not a stale cache from the other position");
    }
}
/// Mop-up bonus: drive the losing king to a corner in winning endgames.
/// The bonus is scaled up as the 50-move clock rises so the engine urgently
/// makes progress rather than shuffling and drawing by the 50-move rule.
fn mop_up(
    material_score: i32,
    white_king_sq: usize,
    black_king_sq: usize,
    mg_phase: i32,
    halfmove_clock: u32,
) -> i32 {
    let eg_weight = ((24 - mg_phase) * 256) / 24;
    if eg_weight < 80 || material_score.abs() < 150 {
        return 0;
    }

    let (corner_push, proximity) = if material_score > 0 {
        let corner_push = king_center_distance(black_king_sq) * 10;
        let proximity = (14 - manhattan(white_king_sq, black_king_sq)) * 4;
        (corner_push, proximity)
    } else {
        let corner_push = king_center_distance(white_king_sq) * 10;
        let proximity = (14 - manhattan(black_king_sq, white_king_sq)) * 4;
        (-(corner_push), -(proximity))
    };

    // Scale urgency: normal weight up to clock=30, then ramp up to 3× by clock=90.
    let urgency = if halfmove_clock <= 30 {
        256
    } else {
        let extra = ((halfmove_clock - 30).min(60) as i32 * 512) / 60;
        256 + extra
    };

    (corner_push + proximity) * eg_weight / 256 * urgency / 256
}

/// Pawn structure penalty (doubled + isolated).
fn pawn_structure_penalty(pawns_bb: u64) -> i32 {
    let mut penalty = 0i32;
    for file in 0..8usize {
        let on_file = (pawns_bb & FILE_MASKS[file]).count_ones() as i32;
        if on_file == 0 {
            continue;
        }
        if on_file > 1 {
            penalty += (on_file - 1) * DOUBLED_PAWN_PENALTY;
        }
        let mut adjacent = 0u64;
        if file > 0 { adjacent |= FILE_MASKS[file - 1]; }
        if file < 7 { adjacent |= FILE_MASKS[file + 1]; }
        if pawns_bb & adjacent == 0 {
            penalty += on_file * ISOLATED_PAWN_PENALTY;
        }
    }
    penalty
}

/// Pawn shield penalty for one side (always positive = penalty amount).
///
/// Only applied when the king is on a wing (files a–c or f–h), indicating
/// it has castled or moved to safety. A king in the centre gets no shield
/// penalty — central pawns being advanced is normal opening play.
fn king_shield_penalty(king_sq: usize, friendly_pawns: u64, is_white: bool) -> i32 {
    let king_file = king_sq % 8;

    // Only check pawn shield when king is on a wing (likely castled).
    if king_file >= 3 && king_file <= 4 {
        return 0;
    }

    let center = king_file.max(1).min(6);

    // Ranks where a pawn still forms a tight shield.
    let shield_ranks: u64 = if is_white {
        0x0000_0000_00FF_FF00 // ranks 2–3 (squares 8–23)
    } else {
        0x00FF_FF00_0000_0000 // ranks 6–7 (squares 40–55)
    };

    let mut penalty = 0i32;
    for file in (center - 1)..=(center + 1) {
        let pawn_on_file = friendly_pawns & FILE_MASKS[file];
        if pawn_on_file == 0 {
            penalty += KING_SHIELD_MISSING;
        } else if pawn_on_file & shield_ranks == 0 {
            penalty += KING_SHIELD_ADVANCED;
        }
    }
    penalty
}

/// Attack-counting king safety.
///
/// Counts how many enemy pieces (knight, bishop, rook, queen) attack the
/// king zone (the 8 squares around the king + the king square itself).
/// Each piece type contributes a weight; the total indexes a non-linear
/// safety table. Only penalises when >= 2 attackers (a lone piece is not
/// usually dangerous enough to warrant a penalty).
fn king_attack_penalty(
    conductor: &PieceConductor,
    king_sq: usize,
    enemy_knights: Bitboard,
    enemy_bishops: Bitboard,
    enemy_rooks: Bitboard,
    enemy_queens: Bitboard,
    occupied: Bitboard,
) -> i32 {
    // King zone = king square + 8 surrounding squares.
    let zone = conductor.king_lut[king_sq] | Bitboard(1u64 << king_sq);

    let mut attack_weight = 0i32;
    let mut attacker_count = 0i32;

    // Knights
    for_each_sq(enemy_knights, |sq| {
        if (conductor.knight_lut[sq] & zone).0 != 0 {
            attack_weight += KNIGHT_ATTACK_WEIGHT;
            attacker_count += 1;
        }
    });

    // Bishops
    for_each_sq(enemy_bishops, |sq| {
        let attacks = conductor.get_bishop_attacks(sq, Bitboard(0), occupied);
        if (attacks & zone).0 != 0 {
            attack_weight += BISHOP_ATTACK_WEIGHT;
            attacker_count += 1;
        }
    });

    // Rooks
    for_each_sq(enemy_rooks, |sq| {
        let attacks = conductor.get_rook_attacks(sq, Bitboard(0), occupied);
        if (attacks & zone).0 != 0 {
            attack_weight += ROOK_ATTACK_WEIGHT;
            attacker_count += 1;
        }
    });

    // Queens
    for_each_sq(enemy_queens, |sq| {
        let rook_part   = conductor.get_rook_attacks(sq, Bitboard(0), occupied);
        let bishop_part = conductor.get_bishop_attacks(sq, Bitboard(0), occupied);
        if ((rook_part | bishop_part) & zone).0 != 0 {
            attack_weight += QUEEN_ATTACK_WEIGHT;
            attacker_count += 1;
        }
    });

    // Zero attackers = no danger. Even a single attacker can be significant
    // (e.g. queen near exposed king) so don't suppress it.
    if attacker_count == 0 {
        return 0;
    }

    SAFETY_TABLE[attack_weight.min(SAFETY_TABLE.len() as i32 - 1) as usize]
}

/// Evaluates the chess board and returns an absolute score:
/// positive = white is ahead, negative = black is ahead.
pub fn evaluate_board(chess_board: &ChessBoard, conductor: &PieceConductor) -> i32 {
    let white = chess_board.get_white();
    let black = chess_board.get_black();
    let pawns   = chess_board.get_pawns();
    let knights = chess_board.get_knights();
    let bishops = chess_board.get_bishops();
    let rooks   = chess_board.get_rooks();
    let queens  = chess_board.get_queens();
    let kings   = chess_board.get_kings();

    let mg_phase = game_phase(chess_board);
    let eg_phase = 24 - mg_phase;

    let white_pawns_bb = (white & pawns).0;
    let black_pawns_bb = (black & pawns).0;

    let white_king_bb = (white & kings).0;
    let black_king_bb = (black & kings).0;
    // Guard: if a king is missing (should not happen in legal chess), bail early.
    if white_king_bb == 0 || black_king_bb == 0 {
        return 0;
    }
    let white_king_sq = white_king_bb.trailing_zeros() as usize;
    let black_king_sq = black_king_bb.trailing_zeros() as usize;

    let mut mg = 0i32;
    let mut eg = 0i32;

    // --- Material (tapered) ---
    let n = count(white & pawns);   mg += n * MG_PAWN_VALUE;   eg += n * EG_PAWN_VALUE;
    let n = count(white & knights); mg += n * MG_KNIGHT_VALUE; eg += n * EG_KNIGHT_VALUE;
    let n = count(white & bishops); mg += n * MG_BISHOP_VALUE; eg += n * EG_BISHOP_VALUE;
    let n = count(white & rooks);   mg += n * MG_ROOK_VALUE;   eg += n * EG_ROOK_VALUE;
    let n = count(white & queens);  mg += n * MG_QUEEN_VALUE;  eg += n * EG_QUEEN_VALUE;

    let n = count(black & pawns);   mg -= n * MG_PAWN_VALUE;   eg -= n * EG_PAWN_VALUE;
    let n = count(black & knights); mg -= n * MG_KNIGHT_VALUE; eg -= n * EG_KNIGHT_VALUE;
    let n = count(black & bishops); mg -= n * MG_BISHOP_VALUE; eg -= n * EG_BISHOP_VALUE;
    let n = count(black & rooks);   mg -= n * MG_ROOK_VALUE;   eg -= n * EG_ROOK_VALUE;
    let n = count(black & queens);  mg -= n * MG_QUEEN_VALUE;  eg -= n * EG_QUEEN_VALUE;

    // Safety: if a king is missing (null-move search artifacts),
    // return raw material score to avoid out-of-bounds panics.
    if white_king_sq >= 64 || black_king_sq >= 64 {
        return (mg * mg_phase + eg * eg_phase) / 24;
    }

    // --- PSTs (non-pawn pieces computed fresh; pawns via pawn hash) ---
    for_each_sq(white & knights, |sq| { mg += mg_knight_table(sq, true); eg += eg_knight_table(sq, true); });
    for_each_sq(white & bishops, |sq| { mg += mg_bishop_table(sq, true); eg += eg_bishop_table(sq, true); });
    for_each_sq(white & rooks,   |sq| { mg += mg_rook_table(sq, true);   eg += eg_rook_table(sq, true); });
    for_each_sq(white & queens,  |sq| { mg += mg_queen_table(sq, true);  eg += eg_queen_table(sq, true); });
    for_each_sq(white & kings,   |sq| { mg += mg_king_table(sq, true);   eg += eg_king_table(sq, true); });

    for_each_sq(black & knights, |sq| { mg -= mg_knight_table(sq, false); eg -= eg_knight_table(sq, false); });
    for_each_sq(black & bishops, |sq| { mg -= mg_bishop_table(sq, false); eg -= eg_bishop_table(sq, false); });
    for_each_sq(black & rooks,   |sq| { mg -= mg_rook_table(sq, false);   eg -= eg_rook_table(sq, false); });
    for_each_sq(black & queens,  |sq| { mg -= mg_queen_table(sq, false);  eg -= eg_queen_table(sq, false); });
    for_each_sq(black & kings, |sq| {
        mg -= mg_king_table(sq, false);
        eg -= eg_king_table(sq, false);
    });

    // --- Pawn hash: PSTs + structure + passed pawn base bonuses ---
    // Probe the per-thread pawn hash table.  On a miss, compute everything
    // from scratch and store the result.  Hit rate is very high because pawn
    // structure rarely changes in a single search tree.
    let pkey = pawn_key(white_pawns_bb, black_pawns_bb);
    let ph = PAWN_TABLE.with(|t| {
        // SAFETY: thread_local, never aliased.
        let table = unsafe { &mut *t.0.get() };
        if let Some(e) = table.probe(pkey) {
            return *e;
        }
        // Cache miss — compute all pawn-only terms.
        let mut pmg = 0i32;
        let mut peg = 0i32;
        for_each_sq(white & pawns, |sq| { pmg += mg_pawn_table(sq, true);  peg += eg_pawn_table(sq, true); });
        for_each_sq(black & pawns, |sq| { pmg -= mg_pawn_table(sq, false); peg -= eg_pawn_table(sq, false); });

        let w_struct = pawn_structure_penalty(white_pawns_bb);
        let b_struct = pawn_structure_penalty(black_pawns_bb);

        let mut pw_mg = 0i32; let mut pw_eg = 0i32;
        let mut pb_mg = 0i32; let mut pb_eg = 0i32;
        let mut wpass = 0u64; let mut bpass = 0u64;
        for_each_sq(white & pawns, |sq| {
            if is_passed_pawn(sq, black_pawns_bb, true) {
                wpass |= 1u64 << sq;
                pw_mg += passed_pawn_bonus_mg(sq, true);
                pw_eg += passed_pawn_bonus_eg(sq, true);
            }
        });
        for_each_sq(black & pawns, |sq| {
            if is_passed_pawn(sq, white_pawns_bb, false) {
                bpass |= 1u64 << sq;
                pb_mg += passed_pawn_bonus_mg(sq, false);
                pb_eg += passed_pawn_bonus_eg(sq, false);
            }
        });

        let entry = PawnHashEntry {
            key:          pkey,
            pawn_mg:      pmg,
            pawn_eg:      peg,
            struct_score: b_struct as i32 - w_struct as i32,
            pass_w_mg:    pw_mg,
            pass_w_eg:    pw_eg,
            pass_b_mg:    pb_mg,
            pass_b_eg:    pb_eg,
            white_passers: wpass,
            black_passers: bpass,
        };
        table.store(entry);
        entry
    });

    mg += ph.pawn_mg;
    eg += ph.pawn_eg;

    // --- King safety (MG only — fades naturally in endgame blend) ---
    //
    // Pawn shield: penalise missing/advanced shield pawns when king is on a wing.
    mg -= king_shield_penalty(white_king_sq, white_pawns_bb, true);
    mg += king_shield_penalty(black_king_sq, black_pawns_bb, false);

    // Attack counting: penalise when multiple enemy pieces aim at the king zone.
    let occupied = chess_board.get_all_pieces();
    mg -= king_attack_penalty(
        conductor, white_king_sq,
        black & knights, black & bishops, black & rooks, black & queens,
        occupied,
    );
    mg += king_attack_penalty(
        conductor, black_king_sq,
        white & knights, white & bishops, white & rooks, white & queens,
        occupied,
    );

    // --- Bishop pair ---
    if count(white & bishops) >= 2 { mg += MG_BISHOP_PAIR_BONUS; eg += EG_BISHOP_PAIR_BONUS; }
    if count(black & bishops) >= 2 { mg -= MG_BISHOP_PAIR_BONUS; eg -= EG_BISHOP_PAIR_BONUS; }

    // --- Rook on open / semi-open file ---
    let all_pawns_bb = (pawns).0;
    for_each_sq(white & rooks, |sq| {
        let file_mask = FILE_MASKS[sq % 8];
        if all_pawns_bb & file_mask == 0 {
            mg += MG_ROOK_OPEN_FILE;      eg += EG_ROOK_OPEN_FILE;
        } else if white_pawns_bb & file_mask == 0 {
            mg += MG_ROOK_SEMI_OPEN_FILE; eg += EG_ROOK_SEMI_OPEN_FILE;
        }
    });
    for_each_sq(black & rooks, |sq| {
        let file_mask = FILE_MASKS[sq % 8];
        if all_pawns_bb & file_mask == 0 {
            mg -= MG_ROOK_OPEN_FILE;      eg -= EG_ROOK_OPEN_FILE;
        } else if black_pawns_bb & file_mask == 0 {
            mg -= MG_ROOK_SEMI_OPEN_FILE; eg -= EG_ROOK_SEMI_OPEN_FILE;
        }
    });

    // --- Rook behind passed pawn (EG) — uses cached passer bitboards ---
    for_each_sq(white & rooks, |sq| {
        let file = sq % 8;
        let rank = sq / 8;
        let above = 1u64.wrapping_shl((rank as u32 + 1) * 8).wrapping_sub(1);
        let ahead_mask = FILE_MASKS[file] & !above;
        if ph.white_passers & ahead_mask != 0 {
            eg += ROOK_BEHIND_PASSER_EG;
        }
    });
    for_each_sq(black & rooks, |sq| {
        let file = sq % 8;
        let rank = sq / 8;
        let below_mask = FILE_MASKS[file] & ((1u64 << (rank * 8)).wrapping_sub(1));
        if ph.black_passers & below_mask != 0 {
            eg -= ROOK_BEHIND_PASSER_EG;
        }
    });

    // --- Tapered blend ---
    let mut score = (mg * mg_phase + eg * eg_phase) / 24;

    // --- Mop-up: drive the losing king to a corner in winning endgames ---
    let material_score =
          count(white & pawns)   * MG_PAWN_VALUE   - count(black & pawns)   * MG_PAWN_VALUE
        + count(white & knights) * MG_KNIGHT_VALUE - count(black & knights) * MG_KNIGHT_VALUE
        + count(white & bishops) * MG_BISHOP_VALUE - count(black & bishops) * MG_BISHOP_VALUE
        + count(white & rooks)   * MG_ROOK_VALUE   - count(black & rooks)   * MG_ROOK_VALUE
        + count(white & queens)  * MG_QUEEN_VALUE  - count(black & queens)  * MG_QUEEN_VALUE;
    score += mop_up(material_score, white_king_sq, black_king_sq, mg_phase,
                    chess_board.get_halfmove_clock());


    // --- Passed pawns (cached base bonuses + fresh king-proximity) ---
    // Base bonuses come from the pawn hash; king proximity is computed fresh
    // since kings move during search.
    score += (ph.pass_w_mg * mg_phase + ph.pass_w_eg * eg_phase) / 24;
    score -= (ph.pass_b_mg * mg_phase + ph.pass_b_eg * eg_phase) / 24;

    let mut wpass = ph.white_passers;
    while wpass != 0 {
        let sq = wpass.trailing_zeros() as usize;
        wpass &= wpass - 1;
        let rank_weight = (5 * (sq / 8) as i32 - 13).max(0);
        if rank_weight > 0 {
            let friendly_dist = chebyshev(sq, white_king_sq).min(5);
            let enemy_dist    = chebyshev(sq, black_king_sq).min(5);
            score += (enemy_dist * PASSER_KING_ENEMY_WT
                    - friendly_dist * PASSER_KING_FRIEND_WT)
                    * rank_weight * eg_phase / 24;
        }
    }
    let mut bpass = ph.black_passers;
    while bpass != 0 {
        let sq = bpass.trailing_zeros() as usize;
        bpass &= bpass - 1;
        let black_rank = 7 - sq / 8;
        let rank_weight = (5 * black_rank as i32 - 13).max(0);
        if rank_weight > 0 {
            let friendly_dist = chebyshev(sq, black_king_sq).min(5);
            let enemy_dist    = chebyshev(sq, white_king_sq).min(5);
            score -= (enemy_dist * PASSER_KING_ENEMY_WT
                    - friendly_dist * PASSER_KING_FRIEND_WT)
                    * rank_weight * eg_phase / 24;
        }
    }

    // --- Pawn structure (cached) ---
    score += ph.struct_score;

    // --- Mobility ---
    // Weighted by mg_phase so mobility matters more in the middlegame when
    // piece activity is more important than in simplified endgames.
    let black_pawn_atk = pawn_attacks(black_pawns_bb, false);
    let white_pawn_atk = pawn_attacks(white_pawns_bb, true);

    let white_mob = mobility_score(
        conductor,
        white & knights, white & bishops, white & rooks, white & queens,
        white, occupied,
        black_pawn_atk, // exclude squares attacked by black pawns
    );
    let black_mob = mobility_score(
        conductor,
        black & knights, black & bishops, black & rooks, black & queens,
        black, occupied,
        white_pawn_atk, // exclude squares attacked by white pawns
    );
    score += (white_mob - black_mob) * mg_phase / 24;

    // --- Rook on 7th rank ---
    // Blend the bonus using the same mg_phase/eg_phase as material/PSTs.
    let black_on_8th = (black & kings).0 & 0xFF00_0000_0000_0000 != 0;
    for_each_sq(white & rooks, |sq| {
        if sq / 8 == 6 && black_on_8th {
            score += (MG_ROOK_ON_SEVENTH * mg_phase + EG_ROOK_ON_SEVENTH * eg_phase) / 24;
        }
    });
    let white_on_1st = (white & kings).0 & 0x0000_0000_0000_00FF != 0;
    for_each_sq(black & rooks, |sq| {
        if sq / 8 == 1 && white_on_1st {
            score -= (MG_ROOK_ON_SEVENTH * mg_phase + EG_ROOK_ON_SEVENTH * eg_phase) / 24;
        }
    });

    // --- Knight outposts ---
    // A knight on a square protected by a friendly pawn that no enemy pawn
    // can ever attack (reusing the passed-pawn mask for the outpost square).
    // Only awarded on ranks 4–6 for white (2–4 for black) where outposts matter.
    for_each_sq(white & knights, |sq| {
        let rank = sq / 8;
        if rank >= 3 && rank <= 5
            && is_passed_pawn(sq, black_pawns_bb, true)
            && is_protected_by_pawn(sq, white_pawns_bb, true)
        {
            score += (MG_KNIGHT_OUTPOST * mg_phase + EG_KNIGHT_OUTPOST * eg_phase) / 24;
        }
    });
    for_each_sq(black & knights, |sq| {
        let rank = sq / 8;
        if rank >= 2 && rank <= 4
            && is_passed_pawn(sq, white_pawns_bb, false)
            && is_protected_by_pawn(sq, black_pawns_bb, false)
        {
            score -= (MG_KNIGHT_OUTPOST * mg_phase + EG_KNIGHT_OUTPOST * eg_phase) / 24;
        }
    });

    score
}
