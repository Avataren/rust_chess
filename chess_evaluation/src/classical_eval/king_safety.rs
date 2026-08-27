//! King-safety evaluation: pawn shield and attack-counting.

use chess_foundation::Bitboard;
use move_generator::piece_conductor::PieceConductor;

/// Pawn shield penalty for one side (always positive = penalty amount).
///
/// Only applied when the king is on a wing (files a–c or f–h), indicating
/// it has castled or moved to safety. A king in the centre gets no shield
/// penalty — central pawns being advanced is normal opening play.
pub(super) fn king_shield_penalty(king_sq: usize, friendly_pawns: u64, is_white: bool) -> i32 {
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
        let pawn_on_file = friendly_pawns & super::FILE_MASKS[file];
        if pawn_on_file == 0 {
            penalty += super::KING_SHIELD_MISSING;
        } else if pawn_on_file & shield_ranks == 0 {
            penalty += super::KING_SHIELD_ADVANCED;
        }
    }
    penalty
}

/// Attack-counting king safety.
///
/// Counts how many enemy pieces (knight, bishop, rook, queen) attack the
/// king zone (the 8 squares around the king + the king square itself).
/// Each piece type contributes a weight; the total indexes a non-linear
/// safety table.
///
/// Additional weight is added for open/semi-open files adjacent to the king:
/// an open file acts as a highway for rooks and queens and dramatically
/// amplifies existing piece attacks.
///
/// The total is scaled down by ~50% when the enemy has no queen, since
/// mating attacks without a queen are much rarer.
///
/// `attacker_lead` is the attacking (enemy) side's non-pawn material minus this
/// king's side. When negative — the attacker has sacrificed — the penalty is
/// scaled down: a static eval can't confirm a sac works, so a phantom +150 must
/// not paper over a real material deficit (unsound Nxc5 / …Bxh3 sacs).
pub(super) fn king_attack_penalty(
    conductor: &PieceConductor,
    king_sq: usize,
    enemy_knights: Bitboard,
    enemy_bishops: Bitboard,
    enemy_rooks: Bitboard,
    enemy_queens: Bitboard,
    occupied: Bitboard,
    friendly_pawns: u64,
    enemy_pawns: u64,
    is_white_king: bool,
    attacker_lead: i32,
) -> i32 {
    // Base king zone = king square + 8 surrounding squares.
    let base = Bitboard(conductor.king_lut[king_sq].0 | (1u64 << king_sq));
    let king_file = king_sq % 8;

    // Extended zone adds the infiltration row one rank toward the enemy
    // (f3/g3/h3 vs a g1 king) — the squares an attacking knight / rook / queen
    // uses to swarm a king without ever landing on the 8 base squares (e.g.
    // Nf4 + Qg4 vs Kg1 read as "no attackers" under the base zone alone).
    // Bishops keep the base zone: a bishop on g4/g5 is usually a pin, and
    // counting its f3/h3 ray produces phantom king-danger.
    let zone = Bitboard(base.0 | if is_white_king { base.0 << 8 } else { base.0 >> 8 });

    let mut attack_weight = 0i32;
    let mut attacker_count = 0i32;

    // Knights
    super::for_each_sq(enemy_knights, |sq| {
        if (conductor.knight_lut[sq] & zone).0 != 0 {
            attack_weight += super::KNIGHT_ATTACK_WEIGHT;
            attacker_count += 1;
        }
    });

    // Bishops — base zone only (see note above).
    super::for_each_sq(enemy_bishops, |sq| {
        let attacks = conductor.get_bishop_attacks(sq, Bitboard(0), occupied);
        if (attacks & base).0 != 0 {
            attack_weight += super::BISHOP_ATTACK_WEIGHT;
            attacker_count += 1;

            // Greek Gift bonus: bishop has a clear diagonal to the h-file pawn
            // of a kingside-castled king (king on g-file = file 6).
            if king_file == 6 && (attacks.0 & base.0 & super::H_FILE) != 0 {
                attack_weight += super::GREEK_GIFT_BONUS;
            }
        }
    });

    // Rooks
    super::for_each_sq(enemy_rooks, |sq| {
        let attacks = conductor.get_rook_attacks(sq, Bitboard(0), occupied);
        if (attacks & zone).0 != 0 {
            attack_weight += super::ROOK_ATTACK_WEIGHT;
            attacker_count += 1;
        }
    });

    // Queens
    super::for_each_sq(enemy_queens, |sq| {
        let rook_part   = conductor.get_rook_attacks(sq, Bitboard(0), occupied);
        let bishop_part = conductor.get_bishop_attacks(sq, Bitboard(0), occupied);
        if ((rook_part | bishop_part) & zone).0 != 0 {
            attack_weight += super::QUEEN_ATTACK_WEIGHT;
            attacker_count += 1;
        }
    });

    if attacker_count == 0 {
        return 0;
    }

    // Open / semi-open files near king amplify attack danger.
    let king_file = king_sq % 8;
    let all_pawns = friendly_pawns | enemy_pawns;
    for f in king_file.saturating_sub(1)..=(king_file + 1).min(7) {
        let fmask = super::FILE_MASKS[f];
        if all_pawns & fmask == 0 {
            attack_weight += super::OPEN_FILE_ATTACK_BONUS;
        } else if friendly_pawns & fmask == 0 {
            attack_weight += super::SEMI_OPEN_FILE_ATTACK_BONUS;
        }
    }

    // Scale down heavily when the enemy has no queen.
    let has_enemy_queen = enemy_queens.0 != 0;
    let weight = if has_enemy_queen { attack_weight } else { attack_weight / 2 };

    let raw = super::SAFETY_TABLE[weight.min(super::SAFETY_TABLE.len() as i32 - 1) as usize];

    // Discount an attack the attacker paid material for. attacker_lead < 0 means
    // a sac is on the board: minor-for-pawn (~−200) keeps ~76%, a clean piece
    // (~−330) ~60%, a full rook (~−500) ~40%, floor 25%.
    if attacker_lead >= 0 {
        raw
    } else {
        let keep = (1000 - (-attacker_lead).min(700) * 6 / 5).max(250);
        raw * keep / 1000
    }
}
