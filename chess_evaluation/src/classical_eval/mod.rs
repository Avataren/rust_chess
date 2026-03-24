//! Classical hand-crafted evaluation (HCE).
//! Only compiled when the `classical-eval` feature is enabled.

mod pawn_eval;
mod king_safety;

use chess_board::ChessBoard;
use chess_foundation::Bitboard;
use move_generator::piece_conductor::PieceConductor;

use crate::piece_tables::{
    eg_bishop_table, eg_king_table, eg_knight_table, eg_pawn_table, eg_queen_table,
    eg_rook_table, is_passed_pawn, mg_bishop_table, mg_king_table, mg_knight_table,
    mg_pawn_table, mg_queen_table, mg_rook_table, passed_pawn_bonus_eg, passed_pawn_bonus_mg,
};

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
const KING_SHIELD_MISSING:  i32 = 20; // cp per missing shield pawn (was 15)
const KING_SHIELD_ADVANCED: i32 =  7; // cp per advanced shield pawn (was 5)

// Attack weight per piece type attacking king zone.
const KNIGHT_ATTACK_WEIGHT: i32 = 2;
const BISHOP_ATTACK_WEIGHT: i32 = 2;
const ROOK_ATTACK_WEIGHT:   i32 = 3;
const QUEEN_ATTACK_WEIGHT:  i32 = 5;

// Greek Gift sacrifice bonus.
// Added to attack_weight when an enemy bishop has a clear diagonal to the
// h-file pawn of a kingside-castled king (king on g-file, bishop eyes h7/h2).
// This single constant serves BOTH defence (penalises walking into the pattern)
// and offence (rewards maintaining the attacking setup).
// weight=2 (bishop alone) → SAFETY_TABLE[2]=4 cp  (old, invisible)
// weight=7 (bishop+bonus) → SAFETY_TABLE[7]=155 cp (new, properly scary)
const GREEK_GIFT_BONUS: i32 = 5;

// h-file bitmask — used to identify h7/h2 sacrifice squares inside king zone.
const H_FILE: u64 = 0x8080808080808080;

// Open/semi-open file bonuses added to attack_weight.
// An open file toward the king dramatically amplifies existing piece attacks.
const OPEN_FILE_ATTACK_BONUS:      i32 = 3; // fully open file adjacent to king
const SEMI_OPEN_FILE_ATTACK_BONUS: i32 = 1; // semi-open file adjacent to king

/// Non-linear safety penalty indexed by total attack weight.
/// Approximates SF14's quadratic king-danger curve (danger² / 4096).
/// Ramps slowly for a lone minor piece, steeply when queen + support arrives.
/// Values in centipawns, applied to MG score only.
#[rustfmt::skip]
const SAFETY_TABLE: [i32; 24] = [
//   0    1    2    3    4    5    6    7    8    9
     0,   0,   4,  14,  32,  62, 100, 155, 222, 300,
//  10   11   12   13   14   15   16   17   18   19
   390, 490, 590, 685, 770, 845, 905, 955, 995,1025,
//  20   21   22   23
  1050,1070,1085,1095,
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

/// Evaluates the chess board and returns an absolute score:

pub fn evaluate(chess_board: &ChessBoard, conductor: &PieceConductor) -> i32 {
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
    let ph = pawn_eval::probe_or_fill(
        white_pawns_bb,
        black_pawns_bb,
        &|| {
            let mut pmg = 0i32;
            let mut peg = 0i32;
            for_each_sq(white & pawns, |sq| { pmg += mg_pawn_table(sq, true);  peg += eg_pawn_table(sq, true); });
            for_each_sq(black & pawns, |sq| { pmg -= mg_pawn_table(sq, false); peg -= eg_pawn_table(sq, false); });
            (pmg, peg)
        },
        &|| {
            let mut pw_mg = 0i32; let mut pw_eg = 0i32; let mut wpass = 0u64;
            for_each_sq(white & pawns, |sq| {
                if is_passed_pawn(sq, black_pawns_bb, true) {
                    wpass |= 1u64 << sq;
                    pw_mg += passed_pawn_bonus_mg(sq, true);
                    pw_eg += passed_pawn_bonus_eg(sq, true);
                }
            });
            (pw_mg, pw_eg, wpass)
        },
        &|| {
            let mut pb_mg = 0i32; let mut pb_eg = 0i32; let mut bpass = 0u64;
            for_each_sq(black & pawns, |sq| {
                if is_passed_pawn(sq, white_pawns_bb, false) {
                    bpass |= 1u64 << sq;
                    pb_mg += passed_pawn_bonus_mg(sq, false);
                    pb_eg += passed_pawn_bonus_eg(sq, false);
                }
            });
            (pb_mg, pb_eg, bpass)
        },
    );

    mg += ph.pawn_mg;
    eg += ph.pawn_eg;

    // --- King safety (MG only — fades naturally in endgame blend) ---
    //
    // Pawn shield: penalise missing/advanced shield pawns when king is on a wing.
    mg -= king_safety::king_shield_penalty(white_king_sq, white_pawns_bb, true);
    mg += king_safety::king_shield_penalty(black_king_sq, black_pawns_bb, false);

    // Attack counting: penalise when multiple enemy pieces aim at the king zone.
    // Open files near the king and enemy queen presence amplify the danger.
    let occupied = chess_board.get_all_pieces();
    mg -= king_safety::king_attack_penalty(
        conductor, white_king_sq,
        black & knights, black & bishops, black & rooks, black & queens,
        occupied,
        white_pawns_bb, black_pawns_bb,
    );
    mg += king_safety::king_attack_penalty(
        conductor, black_king_sq,
        white & knights, white & bishops, white & rooks, white & queens,
        occupied,
        black_pawns_bb, white_pawns_bb,
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
