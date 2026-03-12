// PeSTO's Evaluation Function piece-square tables.
// Source: https://www.chessprogramming.org/PeSTO%27s_Evaluation_Function
//
// Convention: a1=0, h8=63 (rank 1 first, rank 8 last in each array).
// White pieces: table[square] directly.
// Black pieces: table[square ^ 56]  ← vertical mirror only (preserves file).
//   NOTE: the old `63 - square` mapping was a 180° rotation (wrong for
//   asymmetric tables); `sq ^ 56` is the correct rank-only flip.

// ── Middlegame tables ─────────────────────────────────────────────────────────

#[rustfmt::skip]
const MG_PAWN: [i32; 64] = [
    // rank 1 (impossible for pawns)
      0,   0,   0,   0,   0,   0,   0,   0,
    // rank 2
    -35,  -1, -20, -23, -15,  24,  38, -22,
    // rank 3
    -26,  -4,  -4, -10,   3,   3,  33, -12,
    // rank 4
    -27,  -2,  -5,  12,  17,   6,  10, -25,
    // rank 5
    -14,  13,   6,  21,  23,  12,  17, -23,
    // rank 6
     -6,   7,  26,  31,  65,  56,  25, -20,
    // rank 7 (near promotion)
     98, 134,  61,  95,  68, 126,  34, -11,
    // rank 8
      0,   0,   0,   0,   0,   0,   0,   0,
];

#[rustfmt::skip]
const MG_KNIGHT: [i32; 64] = [
    // rank 1
   -105, -21, -58, -33, -17, -28, -19, -23,
    // rank 2
    -29, -53, -12,  -3,  -1,  18, -14, -19,
    // rank 3
    -23,  -9,  12,  10,  19,  17,  25, -16,
    // rank 4
    -13,   4,  16,  13,  28,  19,  21,  -8,
    // rank 5
     -9,  17,  19,  53,  37,  69,  18,  22,
    // rank 6
    -47,  60,  37,  65,  84, 129,  73,  44,
    // rank 7
    -73, -41,  72,  36,  23,  62,   7, -17,
    // rank 8
   -167, -89, -34, -49,  61, -97, -15,-107,
];

#[rustfmt::skip]
const MG_BISHOP: [i32; 64] = [
    // rank 1
    -33,  -3, -14, -21, -13, -12, -39, -21,
    // rank 2
      4,  15,  16,   0,   7,  21,  33,   1,
    // rank 3
      0,  15,  15,  15,  14,  27,  18,  10,
    // rank 4
     -6,  13,  13,  26,  34,  12,  10,   4,
    // rank 5
     -4,   5,  19,  50,  37,  37,   7,  -2,
    // rank 6
    -16,  37,  43,  40,  35,  50,  37,  -2,
    // rank 7
    -26,  16, -18, -13,  30,  59,  18, -47,
    // rank 8
    -29,   4, -82, -37, -25, -42,   7,  -8,
];

#[rustfmt::skip]
const MG_ROOK: [i32; 64] = [
    // rank 1
    -19, -13,   1,  17,  16,   7, -37, -26,
    // rank 2
    -44, -16, -20,  -9,  -1,  11,  -6, -71,
    // rank 3
    -45, -25, -16, -17,   3,   0,  -5, -33,
    // rank 4
    -36, -26, -12,  -1,   9,  -7,   6, -23,
    // rank 5
    -24, -11,   7,  26,  24,  35,  -8, -20,
    // rank 6
     -5,  19,  26,  36,  17,  45,  61,  16,
    // rank 7 (powerful rank)
     27,  32,  58,  62,  80,  67,  26,  44,
    // rank 8
     32,  42,  32,  51,  63,   9,  31,  43,
];

#[rustfmt::skip]
const MG_QUEEN: [i32; 64] = [
    // rank 1
     -1, -18,  -9,  10, -15, -25, -31, -50,
    // rank 2
    -35,  -8,  11,   2,   8,  15,  -3,   1,
    // rank 3
    -14,   2, -11,  -2,  -5,   2,  14,   5,
    // rank 4
     -9, -26,  -9, -10,  -2,  -4,   3,  -3,
    // rank 5
    -27, -27, -16, -16,  -1,  17,  -2,   1,
    // rank 6
    -13, -17,   7,   8,  29,  56,  47,  57,
    // rank 7
    -24, -39,  -5,   1, -16,  57,  28,  54,
    // rank 8
    -28,   0,  29,  12,  59,  44,  43,  45,
];

#[rustfmt::skip]
const MG_KING: [i32; 64] = [
    // rank 1
    -15,  36,  12, -54,   8, -28,  24,  14,
    // rank 2
      1,   7,  -8, -64, -43, -16,   9,   8,
    // rank 3
    -14, -14, -22, -46, -44, -30, -15, -27,
    // rank 4
    -49,  -1, -27, -39, -46, -44, -33, -51,
    // rank 5
    -17, -20, -12, -27, -30, -25, -14, -36,
    // rank 6
     -9,  24,   2, -16, -20,   6,  22, -22,
    // rank 7
     29,  -1, -20,  -7,  -8,  -4, -38, -29,
    // rank 8
    -65,  23,  16, -15, -56, -34,   2,  13,
];

// ── Endgame tables ────────────────────────────────────────────────────────────

#[rustfmt::skip]
const EG_PAWN: [i32; 64] = [
    // rank 1
      0,   0,   0,   0,   0,   0,   0,   0,
    // rank 2
     13,   8,   8,  10,  13,   0,   2,  -7,
    // rank 3
      4,   7,  -6,   1,   0,  -5,  -1,  -8,
    // rank 4
     13,   9,  -3,  -7,  -7,  -8,   3,  -1,
    // rank 5
     32,  24,  13,   5,  -2,   4,  17,  17,
    // rank 6
     94, 100,  85,  67,  56,  53,  82,  84,
    // rank 7
    178, 173, 158, 134, 147, 132, 165, 187,
    // rank 8
      0,   0,   0,   0,   0,   0,   0,   0,
];

#[rustfmt::skip]
const EG_KNIGHT: [i32; 64] = [
    // rank 1
    -29, -51, -23, -15, -22, -18, -50, -64,
    // rank 2
    -42, -20, -10,  -5,  -2, -20, -23, -44,
    // rank 3
    -23,  -3,  -1,  15,  10,  -3, -20, -22,
    // rank 4
    -18,  -6,  16,  25,  16,  17,   4, -18,
    // rank 5
    -17,   3,  22,  22,  22,  11,   8, -18,
    // rank 6
    -24, -20,  10,   9,  -1,  -9, -19, -41,
    // rank 7
    -25,  -8, -25,  -2,  -9, -25, -24, -52,
    // rank 8
    -58, -38, -13, -28, -31, -27, -63, -99,
];

#[rustfmt::skip]
const EG_BISHOP: [i32; 64] = [
    // rank 1
    -23,  -9, -23,  -5,  -9, -16,  -5, -17,
    // rank 2
    -14, -18,  -7,  -1,   4,  -9, -15, -27,
    // rank 3
    -12,  -3,   8,  10,  13,   3,  -7, -15,
    // rank 4
     -6,   3,  13,  19,   7,  10,  -3,  -9,
    // rank 5
     -3,   9,  12,   9,  14,  10,   3,   2,
    // rank 6
      2,  -8,   0,  -1,  -2,   6,   0,   4,
    // rank 7
     -8,  -4,   7, -12,  -3, -13,  -4, -14,
    // rank 8
    -14, -21, -11,  -8,  -7,  -9, -17, -24,
];

#[rustfmt::skip]
const EG_ROOK: [i32; 64] = [
    // rank 1
     -9,   2,   3,  -1,  -5, -13,   4, -20,
    // rank 2
     -6,  -6,   0,   2,  -9,  -9, -11,  -3,
    // rank 3
     -4,   0,  -5,  -1,  -7, -12,  -8, -16,
    // rank 4
      3,   5,   8,   4,  -5,  -6,  -8, -11,
    // rank 5
      4,   3,  13,   1,   2,   1,  -1,   2,
    // rank 6
      7,   7,   7,   5,   4,  -3,  -5,  -3,
    // rank 7
     11,  13,  13,  11,  -3,   3,   8,   3,
    // rank 8
     13,  10,  18,  15,  12,  12,   8,   5,
];

#[rustfmt::skip]
const EG_QUEEN: [i32; 64] = [
    // rank 1
    -33, -28, -22, -43,  -5, -32, -20, -41,
    // rank 2
    -22, -23, -30, -16, -16, -23, -36, -32,
    // rank 3
    -16, -27,  15,   6,   9,  17,  10,   5,
    // rank 4
    -18,  28,  19,  47,  31,  34,  39,  23,
    // rank 5
      3,  22,  24,  45,  57,  40,  57,  36,
    // rank 6
    -20,   6,   9,  49,  47,  35,  19,   9,
    // rank 7
    -17,  20,  32,  41,  58,  25,  30,   0,
    // rank 8
     -9,  22,  22,  27,  27,  19,  10,  20,
];

#[rustfmt::skip]
const EG_KING: [i32; 64] = [
    // rank 1
    -53, -34, -21, -11, -28, -14, -24, -43,
    // rank 2
    -27, -11,   4,  13,  14,   4,  -5, -17,
    // rank 3
    -19,  -3,  11,  21,  23,  16,   7,  -9,
    // rank 4
    -18,  -4,  21,  24,  27,  23,   9, -11,
    // rank 5
     -8,  22,  24,  27,  26,  33,  26,   3,
    // rank 6
     10,  17,  23,  15,  20,  45,  44,  13,
    // rank 7
    -12,  17,  14,  17,  17,  38,  23,  11,
    // rank 8
    -74, -35, -18, -18, -11,  15,   4, -17,
];

// ── Passed-pawn bonus (rank-indexed, white perspective) ───────────────────────

const PASSED_PAWN_BONUS_MG: [i32; 8] = [0, 0,  5, 15, 25,  45,  70, 0];
const PASSED_PAWN_BONUS_EG: [i32; 8] = [0, 0, 20, 40, 65, 105, 160, 0];

// ── Index helper ──────────────────────────────────────────────────────────────

/// Correct vertical mirror for black: flip rank, keep file (`sq ^ 56`).
/// Old `63 - sq` was a 180° rotation (wrong for asymmetric tables).
#[inline]
fn idx(square: usize, is_white: bool) -> usize {
    if is_white { square } else { square ^ 56 }
}

// ── MG accessors ──────────────────────────────────────────────────────────────

pub fn mg_pawn_table(sq: usize, is_white: bool)   -> i32 { MG_PAWN[idx(sq, is_white)] }
pub fn mg_knight_table(sq: usize, is_white: bool) -> i32 { MG_KNIGHT[idx(sq, is_white)] }
pub fn mg_bishop_table(sq: usize, is_white: bool) -> i32 { MG_BISHOP[idx(sq, is_white)] }
pub fn mg_rook_table(sq: usize, is_white: bool)   -> i32 { MG_ROOK[idx(sq, is_white)] }
pub fn mg_queen_table(sq: usize, is_white: bool)  -> i32 { MG_QUEEN[idx(sq, is_white)] }
pub fn mg_king_table(sq: usize, is_white: bool)   -> i32 { MG_KING[idx(sq, is_white)] }

// ── EG accessors ──────────────────────────────────────────────────────────────

pub fn eg_pawn_table(sq: usize, is_white: bool)   -> i32 { EG_PAWN[idx(sq, is_white)] }
pub fn eg_knight_table(sq: usize, is_white: bool) -> i32 { EG_KNIGHT[idx(sq, is_white)] }
pub fn eg_bishop_table(sq: usize, is_white: bool) -> i32 { EG_BISHOP[idx(sq, is_white)] }
pub fn eg_rook_table(sq: usize, is_white: bool)   -> i32 { EG_ROOK[idx(sq, is_white)] }
pub fn eg_queen_table(sq: usize, is_white: bool)  -> i32 { EG_QUEEN[idx(sq, is_white)] }
pub fn eg_king_table(sq: usize, is_white: bool)   -> i32 { EG_KING[idx(sq, is_white)] }

// ── Passed-pawn helpers ───────────────────────────────────────────────────────

/// Returns true if the pawn at `square` has no enemy pawns blocking or
/// guarding its path to promotion.
pub fn is_passed_pawn(square: usize, enemy_pawns_bb: u64, is_white: bool) -> bool {
    let rank = square / 8;
    let file = square % 8;
    let mut mask: u64 = 0;

    if is_white {
        for r in (rank + 1)..8 {
            mask |= 1u64 << (r * 8 + file);
            if file > 0 { mask |= 1u64 << (r * 8 + file - 1); }
            if file < 7 { mask |= 1u64 << (r * 8 + file + 1); }
        }
    } else {
        for r in 0..rank {
            mask |= 1u64 << (r * 8 + file);
            if file > 0 { mask |= 1u64 << (r * 8 + file - 1); }
            if file < 7 { mask |= 1u64 << (r * 8 + file + 1); }
        }
    }

    (enemy_pawns_bb & mask) == 0
}

/// Middlegame rank-scaled bonus for a passed pawn.
pub fn passed_pawn_bonus_mg(square: usize, is_white: bool) -> i32 {
    let rank = if is_white { square / 8 } else { 7 - square / 8 };
    PASSED_PAWN_BONUS_MG[rank]
}

/// Endgame rank-scaled bonus for a passed pawn.
pub fn passed_pawn_bonus_eg(square: usize, is_white: bool) -> i32 {
    let rank = if is_white { square / 8 } else { 7 - square / 8 };
    PASSED_PAWN_BONUS_EG[rank]
}

/// Rank-scaled bonus for a passed pawn (backward-compat alias: returns EG value).
pub fn passed_pawn_bonus(square: usize, is_white: bool) -> i32 {
    passed_pawn_bonus_eg(square, is_white)
}

// ── Backward-compatible aliases (used by lib.rs re-exports) ───────────────────

pub fn pawn_table_value(sq: usize, is_white: bool) -> i32   { mg_pawn_table(sq, is_white) }
pub fn knight_table_value(sq: usize, is_white: bool) -> i32 { mg_knight_table(sq, is_white) }
pub fn bishop_table_value(sq: usize, is_white: bool) -> i32 { mg_bishop_table(sq, is_white) }
pub fn rook_table_value(sq: usize, is_white: bool) -> i32   { mg_rook_table(sq, is_white) }
pub fn queen_table_value(sq: usize, is_white: bool) -> i32  { mg_queen_table(sq, is_white) }
pub fn king_table_value(sq: usize, is_white: bool, eg_weight: i32) -> i32 {
    let mg = mg_king_table(sq, is_white);
    let eg = eg_king_table(sq, is_white);
    (mg * (256 - eg_weight) + eg * eg_weight) / 256
}
pub fn evaluate_pawn_position(sq: usize, is_white: bool) -> i32   { pawn_table_value(sq, is_white) }
pub fn evaluate_knight_position(sq: usize, is_white: bool) -> i32 { knight_table_value(sq, is_white) }
