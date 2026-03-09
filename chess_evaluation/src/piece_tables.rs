// All tables use a1=0, h8=63 convention (standard bitboard ordering).
// White pieces use table[square] directly.
// Black pieces use table[63 - square] (vertical mirror).
// Rows below are laid out rank 1 → rank 8 (bottom to top for white).

#[rustfmt::skip]
const PAWN_TABLE: [i32; 64] = [
    // rank 1 — impossible for pawns
     0,   0,   0,   0,   0,   0,   0,   0,
    // rank 2 — starting squares; slight penalty for blocking d/e pawns
     5,  10,  10, -20, -20,  10,  10,   5,
    // rank 3
     5,  -5, -10,   0,   0, -10,  -5,   5,
    // rank 4 — center control bonus
     0,   0,   0,  20,  20,   0,   0,   0,
    // rank 5
     5,   5,  10,  25,  25,  10,   5,   5,
    // rank 6
    10,  10,  20,  30,  30,  20,  10,  10,
    // rank 7 — near promotion
    50,  50,  50,  50,  50,  50,  50,  50,
    // rank 8 — handled by promotion logic, no extra bonus
     0,   0,   0,   0,   0,   0,   0,   0,
];

#[rustfmt::skip]
const KNIGHT_TABLE: [i32; 64] = [
    // rank 1
    -50, -40, -30, -30, -30, -30, -40, -50,
    // rank 2
    -40, -20,   0,   5,   5,   0, -20, -40,
    // rank 3
    -30,   0,  10,  15,  15,  10,   0, -30,
    // rank 4
    -30,   5,  15,  20,  20,  15,   5, -30,
    // rank 5
    -30,   0,  15,  20,  20,  15,   0, -30,
    // rank 6
    -30,   5,  10,  15,  15,  10,   5, -30,
    // rank 7
    -40, -20,   0,   0,   0,   0, -20, -40,
    // rank 8
    -50, -40, -30, -30, -30, -30, -40, -50,
];

#[rustfmt::skip]
const BISHOP_TABLE: [i32; 64] = [
    // rank 1
    -20, -10, -10, -10, -10, -10, -10, -20,
    // rank 2
    -10,   0,   0,   0,   0,   0,   0, -10,
    // rank 3
    -10,   0,   5,  10,  10,   5,   0, -10,
    // rank 4
    -10,   5,   5,  10,  10,   5,   5, -10,
    // rank 5
    -10,   0,  10,  10,  10,  10,   0, -10,
    // rank 6
    -10,  10,  10,  10,  10,  10,  10, -10,
    // rank 7
    -10,   5,   0,   0,   0,   0,   5, -10,
    // rank 8
    -20, -10, -10, -10, -10, -10, -10, -20,
];

#[rustfmt::skip]
const ROOK_TABLE: [i32; 64] = [
    // rank 1 — small bonus for central files to encourage connection
     0,   0,   0,   5,   5,   0,   0,   0,
    // rank 2
    -5,   0,   0,   0,   0,   0,   0,  -5,
    // rank 3
    -5,   0,   0,   0,   0,   0,   0,  -5,
    // rank 4
    -5,   0,   0,   0,   0,   0,   0,  -5,
    // rank 5
    -5,   0,   0,   0,   0,   0,   0,  -5,
    // rank 6
    -5,   0,   0,   0,   0,   0,   0,  -5,
    // rank 7 — 7th rank is very powerful
     5,  10,  10,  10,  10,  10,  10,   5,
    // rank 8
     0,   0,   0,   0,   0,   0,   0,   0,
];

#[rustfmt::skip]
const QUEEN_TABLE: [i32; 64] = [
    // rank 1 — queen belongs here in the opening, not developed prematurely
    -20, -10, -10,  -5,  -5, -10, -10, -20,
    // rank 2
    -10,   0,   5,   0,   0,   0,   0, -10,
    // rank 3
    -10,   5,   5,   5,   5,   5,   0, -10,
    // rank 4
      0,   0,   5,   5,   5,   5,   0,  -5,
    // rank 5
     -5,   0,   5,   5,   5,   5,   0,  -5,
    // rank 6
    -10,   0,   5,   5,   5,   5,   0, -10,
    // rank 7
    -10,   0,   0,   0,   0,   0,   0, -10,
    // rank 8
    -20, -10, -10,  -5,  -5, -10, -10, -20,
];

// King wants to castle and hide behind pawns in the middlegame.
#[rustfmt::skip]
const KING_MG_TABLE: [i32; 64] = [
    // rank 1 — castled positions (a1/h1 side) rewarded
     20,  30,  10,   0,   0,  10,  30,  20,
    // rank 2
     20,  20,   0,   0,   0,   0,  20,  20,
    // rank 3
    -10, -20, -20, -20, -20, -20, -20, -10,
    // rank 4
    -20, -30, -30, -40, -40, -30, -30, -20,
    // rank 5
    -30, -40, -40, -50, -50, -40, -40, -30,
    // rank 6
    -30, -40, -40, -50, -50, -40, -40, -30,
    // rank 7
    -30, -40, -40, -50, -50, -40, -40, -30,
    // rank 8
    -30, -40, -40, -50, -50, -40, -40, -30,
];

// King should centralise and be active in the endgame.
#[rustfmt::skip]
const KING_EG_TABLE: [i32; 64] = [
    // rank 1
    -50, -40, -30, -20, -20, -30, -40, -50,
    // rank 2
    -30, -20, -10,   0,   0, -10, -20, -30,
    // rank 3
    -30, -10,  20,  30,  30,  20, -10, -30,
    // rank 4
    -30, -10,  30,  40,  40,  30, -10, -30,
    // rank 5
    -30, -10,  30,  40,  40,  30, -10, -30,
    // rank 6
    -30, -10,  20,  30,  30,  20, -10, -30,
    // rank 7
    -30, -30,   0,   0,   0,   0, -30, -30,
    // rank 8
    -50, -30, -30, -30, -30, -30, -30, -50,
];

// Passed-pawn bonus indexed by rank (0 = rank 1, 7 = rank 8).
// White uses rank = square / 8.  Black uses rank = 7 - square / 8.
const PASSED_PAWN_BONUS: [i32; 8] = [0, 0, 10, 20, 35, 60, 100, 0];

#[inline]
fn idx(square: usize, is_white: bool) -> usize {
    if is_white { square } else { 63 - square }
}

pub fn pawn_table_value(square: usize, is_white: bool) -> i32 {
    PAWN_TABLE[idx(square, is_white)]
}

pub fn knight_table_value(square: usize, is_white: bool) -> i32 {
    KNIGHT_TABLE[idx(square, is_white)]
}

pub fn bishop_table_value(square: usize, is_white: bool) -> i32 {
    BISHOP_TABLE[idx(square, is_white)]
}

pub fn rook_table_value(square: usize, is_white: bool) -> i32 {
    ROOK_TABLE[idx(square, is_white)]
}

pub fn queen_table_value(square: usize, is_white: bool) -> i32 {
    QUEEN_TABLE[idx(square, is_white)]
}

/// `endgame_weight` — 0 = pure middlegame, 256 = pure endgame.
pub fn king_table_value(square: usize, is_white: bool, endgame_weight: i32) -> i32 {
    let sq = idx(square, is_white);
    let mg = KING_MG_TABLE[sq];
    let eg = KING_EG_TABLE[sq];
    (mg * (256 - endgame_weight) + eg * endgame_weight) / 256
}

/// Returns true if the pawn at `square` has no enemy pawns blocking or
/// guarding its path to promotion (i.e. it is a passed pawn).
pub fn is_passed_pawn(square: usize, enemy_pawns_bb: u64, is_white: bool) -> bool {
    let rank = square / 8;
    let file = square % 8;
    let mut mask: u64 = 0;

    if is_white {
        for r in (rank + 1)..8 {
            mask |= 1u64 << (r * 8 + file);
            if file > 0 {
                mask |= 1u64 << (r * 8 + file - 1);
            }
            if file < 7 {
                mask |= 1u64 << (r * 8 + file + 1);
            }
        }
    } else {
        for r in 0..rank {
            mask |= 1u64 << (r * 8 + file);
            if file > 0 {
                mask |= 1u64 << (r * 8 + file - 1);
            }
            if file < 7 {
                mask |= 1u64 << (r * 8 + file + 1);
            }
        }
    }

    (enemy_pawns_bb & mask) == 0
}

/// Bonus for a passed pawn at `square`.
pub fn passed_pawn_bonus(square: usize, is_white: bool) -> i32 {
    let rank = if is_white {
        square / 8
    } else {
        7 - square / 8
    };
    PASSED_PAWN_BONUS[rank]
}

// Keep old names around so lib.rs re-exports still compile.
pub fn evaluate_pawn_position(square: usize, is_white: bool) -> i32 {
    pawn_table_value(square, is_white)
}

pub fn evaluate_knight_position(square: usize, is_white: bool) -> i32 {
    knight_table_value(square, is_white)
}
