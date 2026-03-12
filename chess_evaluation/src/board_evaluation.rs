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

// King safety — pawn shield (only when castled) + attack counting.
const KING_SHIELD_MISSING:  i32 = 15;
const KING_SHIELD_ADVANCED: i32 =  5;

// Attack weight per piece type attacking king zone.
const KNIGHT_ATTACK_WEIGHT: i32 = 2;
const BISHOP_ATTACK_WEIGHT: i32 = 2;
const ROOK_ATTACK_WEIGHT:   i32 = 3;
const QUEEN_ATTACK_WEIGHT:  i32 = 5;

/// Non-linear safety penalty indexed by total attack weight.
/// Ramps slowly for 1–2 minor pieces, steeply once queen + support arrive.
/// Values in centipawns, applied to MG score only.
#[rustfmt::skip]
const SAFETY_TABLE: [i32; 20] = [
//   0    1    2    3    4    5    6    7    8    9
     0,   0,   1,   3,   6,  12,  20,  30,  43,  58,
//  10   11   12   13   14   15   16   17   18   19
    75,  95, 117, 141, 168, 197, 228, 261, 296, 333,
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

fn count(bb: Bitboard) -> i32 {
    bb.count_ones() as i32
}

/// Game phase weight: 0 (full endgame) … 24 (full middlegame).
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
fn manhattan(sq1: usize, sq2: usize) -> i32 {
    let r1 = (sq1 / 8) as i32;
    let f1 = (sq1 % 8) as i32;
    let r2 = (sq2 / 8) as i32;
    let f2 = (sq2 % 8) as i32;
    (r1 - r2).abs() + (f1 - f2).abs()
}

/// Chebyshev distance between two squares (king metric: max of rank/file deltas).
fn chebyshev(sq1: usize, sq2: usize) -> i32 {
    let r1 = (sq1 / 8) as i32; let f1 = (sq1 % 8) as i32;
    let r2 = (sq2 / 8) as i32; let f2 = (sq2 % 8) as i32;
    (r1 - r2).abs().max((f1 - f2).abs())
}

/// Manhattan distance of a king from the nearest centre square (d4/d5/e4/e5).
fn king_center_distance(sq: usize) -> i32 {
    let rank = (sq / 8) as i32;
    let file = (sq % 8) as i32;
    let rank_dist = (rank - 3).abs().min((rank - 4).abs());
    let file_dist = (file - 3).abs().min((file - 4).abs());
    rank_dist + file_dist
}

/// Mop-up bonus: drive the losing king to a corner in winning endgames.
fn mop_up(
    material_score: i32,
    white_king_sq: usize,
    black_king_sq: usize,
    mg_phase: i32,
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

    (corner_push + proximity) * eg_weight / 256
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

    // A lone attacker rarely constitutes a real threat.
    if attacker_count < 2 {
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

    // --- White PSTs ---
    for_each_sq(white & pawns, |sq| {
        mg += mg_pawn_table(sq, true);
        eg += eg_pawn_table(sq, true);
    });
    for_each_sq(white & knights, |sq| { mg += mg_knight_table(sq, true); eg += eg_knight_table(sq, true); });
    for_each_sq(white & bishops, |sq| { mg += mg_bishop_table(sq, true); eg += eg_bishop_table(sq, true); });
    for_each_sq(white & rooks,   |sq| { mg += mg_rook_table(sq, true);   eg += eg_rook_table(sq, true); });
    for_each_sq(white & queens,  |sq| { mg += mg_queen_table(sq, true);  eg += eg_queen_table(sq, true); });
    for_each_sq(white & kings,   |sq| { mg += mg_king_table(sq, true);   eg += eg_king_table(sq, true); });

    // --- Black PSTs ---
    for_each_sq(black & pawns, |sq| {
        mg -= mg_pawn_table(sq, false);
        eg -= eg_pawn_table(sq, false);
    });
    for_each_sq(black & knights, |sq| { mg -= mg_knight_table(sq, false); eg -= eg_knight_table(sq, false); });
    for_each_sq(black & bishops, |sq| { mg -= mg_bishop_table(sq, false); eg -= eg_bishop_table(sq, false); });
    for_each_sq(black & rooks,   |sq| { mg -= mg_rook_table(sq, false);   eg -= eg_rook_table(sq, false); });
    for_each_sq(black & queens,  |sq| { mg -= mg_queen_table(sq, false);  eg -= eg_queen_table(sq, false); });
    for_each_sq(black & kings, |sq| {
        mg -= mg_king_table(sq, false);
        eg -= eg_king_table(sq, false);
    });

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

    // --- Rook behind passed pawn (EG) ---
    for_each_sq(white & rooks, |sq| {
        let file = sq % 8;
        let rank = sq / 8;
        // Squares on same file with higher rank (ahead for white)
        let above = 1u64.wrapping_shl((rank as u32 + 1) * 8).wrapping_sub(1);
        let ahead_mask = FILE_MASKS[file] & !above;
        let mut ahead_pawns = white_pawns_bb & ahead_mask;
        while ahead_pawns != 0 {
            let pawn_sq = ahead_pawns.trailing_zeros() as usize;
            ahead_pawns &= ahead_pawns - 1;
            if is_passed_pawn(pawn_sq, black_pawns_bb, true) {
                eg += ROOK_BEHIND_PASSER_EG;
                break;
            }
        }
    });
    for_each_sq(black & rooks, |sq| {
        let file = sq % 8;
        let rank = sq / 8;
        // Squares on same file with lower rank (ahead for black)
        let below_mask = FILE_MASKS[file] & ((1u64 << (rank * 8)).wrapping_sub(1));
        let mut ahead_pawns = black_pawns_bb & below_mask;
        while ahead_pawns != 0 {
            let pawn_sq = ahead_pawns.trailing_zeros() as usize;
            ahead_pawns &= ahead_pawns - 1;
            if is_passed_pawn(pawn_sq, white_pawns_bb, false) {
                eg -= ROOK_BEHIND_PASSER_EG;
                break;
            }
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
    score += mop_up(material_score, white_king_sq, black_king_sq, mg_phase);


    // --- Passed pawns (tapered bonus + rank-weighted king proximity) ---
    for_each_sq(white & pawns, |sq| {
        if is_passed_pawn(sq, black_pawns_bb, true) {
            let bonus = (passed_pawn_bonus_mg(sq, true) * mg_phase
                       + passed_pawn_bonus_eg(sq, true) * eg_phase) / 24;
            score += bonus;
            // Rank weight: 0 for ranks 0-2, grows with rank (Stockfish: 5r-13).
            let rank_weight = (5 * (sq / 8) as i32 - 13).max(0);
            if rank_weight > 0 {
                let friendly_dist = chebyshev(sq, white_king_sq).min(5);
                let enemy_dist    = chebyshev(sq, black_king_sq).min(5);
                let proximity = (enemy_dist * PASSER_KING_ENEMY_WT
                               - friendly_dist * PASSER_KING_FRIEND_WT)
                               * rank_weight * eg_phase / 24;
                score += proximity;
            }
        }
    });
    for_each_sq(black & pawns, |sq| {
        if is_passed_pawn(sq, white_pawns_bb, false) {
            let bonus = (passed_pawn_bonus_mg(sq, false) * mg_phase
                       + passed_pawn_bonus_eg(sq, false) * eg_phase) / 24;
            score -= bonus;
            let black_rank = 7 - sq / 8; // relative rank from black's perspective
            let rank_weight = (5 * black_rank as i32 - 13).max(0);
            if rank_weight > 0 {
                let friendly_dist = chebyshev(sq, black_king_sq).min(5);
                let enemy_dist    = chebyshev(sq, white_king_sq).min(5);
                let proximity = (enemy_dist * PASSER_KING_ENEMY_WT
                               - friendly_dist * PASSER_KING_FRIEND_WT)
                               * rank_weight * eg_phase / 24;
                score -= proximity;
            }
        }
    });

    // --- Pawn structure ---
    score -= pawn_structure_penalty(white_pawns_bb);
    score += pawn_structure_penalty(black_pawns_bb);

    score
}
