/// Static Exchange Evaluation (SEE)
///
/// Evaluates the material outcome of a capture sequence on a square using the
/// "least-valuable attacker" method. Both sides alternate capturing with their
/// cheapest available piece; either side can stop the exchange if continuing
/// would be losing.
///
/// Used for:
///   - Move ordering: put winning/even captures (SEE ≥ 0) before losing ones
///   - Qsearch pruning: skip captures where SEE < 0
use chess_board::ChessBoard;
use chess_foundation::{Bitboard, piece::PieceType};
use move_generator::piece_conductor::PieceConductor;

// ── Piece values for exchange evaluation ─────────────────────────────────────

pub const SEE_PAWN:   i32 = 100;
pub const SEE_KNIGHT: i32 = 300;
pub const SEE_BISHOP: i32 = 325;
pub const SEE_ROOK:   i32 = 500;
pub const SEE_QUEEN:  i32 = 900;
pub const SEE_KING:   i32 = 20_000;

pub fn see_piece_value(pt: PieceType) -> i32 {
    match pt {
        PieceType::Pawn   => SEE_PAWN,
        PieceType::Knight => SEE_KNIGHT,
        PieceType::Bishop => SEE_BISHOP,
        PieceType::Rook   => SEE_ROOK,
        PieceType::Queen  => SEE_QUEEN,
        PieceType::King   => SEE_KING,
        PieceType::None   => 0,
    }
}

fn sq_see_value(board: &ChessBoard, sq: usize) -> i32 {
    board.get_piece_type(sq as u16)
        .map(see_piece_value)
        .unwrap_or(0)
}

// ── Attacker generation ───────────────────────────────────────────────────────

/// All pieces (both colors) attacking `sq` given the current occupancy `occ`.
/// Passing updated `occ` after each capture reveals X-ray attackers correctly.
pub fn attackers_of(
    board: &ChessBoard,
    conductor: &PieceConductor,
    sq: usize,
    occ: Bitboard,
) -> Bitboard {
    let mut atk = Bitboard(0);
    let file = sq % 8;

    // Pawn attacks: find which pawns attack `sq`.
    // White pawn at (sq-7) attacks sq if sq not on h-file; at (sq-9) if not on a-file.
    // Black pawn at (sq+7) attacks sq if sq not on a-file; at (sq+9) if not on h-file.
    let wp = board.get_white() & board.get_pawns();
    let bp = board.get_black() & board.get_pawns();
    if sq >= 7 && file != 7 { atk |= Bitboard(1 << (sq - 7)) & wp; }
    if sq >= 9 && file != 0 { atk |= Bitboard(1 << (sq - 9)) & wp; }
    if sq + 7 < 64 && file != 0 { atk |= Bitboard(1 << (sq + 7)) & bp; }
    if sq + 9 < 64 && file != 7 { atk |= Bitboard(1 << (sq + 9)) & bp; }

    // Knight and king (pre-computed LUTs, occupancy-independent).
    atk |= conductor.knight_lut[sq] & board.get_knights() & occ;
    atk |= conductor.king_lut[sq]   & board.get_kings()   & occ;

    // Sliding pieces — use current `occ` so removed pieces reveal X-rays.
    let diag = conductor.get_bishop_attacks(sq, occ, occ);
    let orth = conductor.get_rook_attacks(sq, occ, occ);
    atk |= diag & (board.get_bishops() | board.get_queens()) & occ;
    atk |= orth & (board.get_rooks()   | board.get_queens()) & occ;

    atk
}

/// Find the least-valuable piece in `candidates` (must be non-empty).
/// Returns (square, SEE value).
fn lva(board: &ChessBoard, candidates: Bitboard) -> (usize, i32) {
    let order: [(Bitboard, i32); 6] = [
        (board.get_pawns(),   SEE_PAWN),
        (board.get_knights(), SEE_KNIGHT),
        (board.get_bishops(), SEE_BISHOP),
        (board.get_rooks(),   SEE_ROOK),
        (board.get_queens(),  SEE_QUEEN),
        (board.get_kings(),   SEE_KING),
    ];
    for (pieces, value) in order {
        let overlap = candidates & pieces;
        if !overlap.is_empty() {
            return (overlap.0.trailing_zeros() as usize, value);
        }
    }
    unreachable!("lva called on empty candidates")
}

// ── SEE entry point ───────────────────────────────────────────────────────────

/// Returns the net material gain (centipawns, from `is_white`'s perspective) of
/// the capture sequence starting with the piece on `from_sq` capturing on `to_sq`.
///
/// Positive = exchange favors the moving side.
/// Negative = exchange loses material.
/// Zero     = even exchange.
///
/// Algorithm (gain-array method):
///   gain[0]   = value of initial target (piece on to_sq)
///   gain[d]   = value of the piece just "sitting" on to_sq (about to be recaptured)
///   Backward: val = 0; for i = (d-1)..=0: val = gain[i] - max(0, val)
///   Return val.
///
/// The max(0, val) allows each side to opt out of a losing recapture.
pub fn see(
    board: &ChessBoard,
    conductor: &PieceConductor,
    from_sq: usize,
    to_sq: usize,
    is_white: bool,
) -> i32 {
    const MAX_DEPTH: usize = 32;
    let mut gain = [0i32; MAX_DEPTH];
    let mut d = 0usize;

    gain[0] = sq_see_value(board, to_sq);

    // Remove the initial attacker from occupancy (reveals potential X-rays behind it).
    let mut occ = board.get_all_pieces();
    occ.0 &= !(1u64 << from_sq);

    // The piece now "sitting" on to_sq — what the opponent would capture next.
    let mut sitting = sq_see_value(board, from_sq);
    let mut side_white = !is_white;

    loop {
        d += 1;
        if d >= MAX_DEPTH { break; }

        let all_atk = attackers_of(board, conductor, to_sq, occ);
        let side_bb  = if side_white { board.get_white() } else { board.get_black() };
        let candidates = all_atk & side_bb & occ;

        if candidates.is_empty() { break; }

        let (lva_sq, lva_val) = lva(board, candidates);

        // Record what the LVA captures (the sitting piece).
        gain[d] = sitting;

        // LVA piece is now sitting on to_sq; remove it from occupancy.
        sitting  = lva_val;
        occ.0   &= !(1u64 << lva_sq);
        side_white = !side_white;
    }

    // Backward minimax: each side can decline to recapture if it would lose.
    // val = gain[i] - max(0, val): the max(0,...) lets the side stop if continuing is negative.
    let mut val = 0i32;
    let mut i = d; // d is the count of recaptures; gain[0..d] are valid.
    while i > 0 {
        i -= 1;
        val = gain[i] - val.max(0);
    }
    val
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use chess_board::ChessBoard;
    use move_generator::piece_conductor::PieceConductor;

    fn board(fen: &str) -> ChessBoard {
        let mut b = ChessBoard::new();
        b.set_from_fen(fen);
        b
    }
    fn cond() -> PieceConductor { PieceConductor::new() }

    /// Pawn captures undefended queen: straightforward gain.
    #[test]
    fn pawn_captures_undefended_queen() {
        // d5=35, e6=44. No black defenders on e6.
        let b = board("4k3/8/4q3/3P4/8/8/8/4K3 w - - 0 1");
        let score = see(&b, &cond(), 35, 44, true);
        assert_eq!(score, SEE_QUEEN, "pawn x undefended queen = {SEE_QUEEN}, got {score}");
    }

    /// Pawn captures queen defended by another queen: still winning (gain queen, lose pawn).
    #[test]
    fn pawn_captures_queen_defended_by_queen() {
        // d5=35, e6=44. Black queen d7 defends e6.
        // Sequence: Pxe6(+900), Qxd5(-100). Net for white = +800.
        let b = board("4k3/3q4/4q3/3P4/8/8/8/4K3 w - - 0 1");
        let score = see(&b, &cond(), 35, 44, true);
        assert!(score > 700 && score <= 900,
            "pawn x queen defended by queen ≈ +800, got {score}");
    }

    /// Queen captures pawn defended by rook: losing exchange (-800).
    #[test]
    fn queen_captures_pawn_defended_by_rook() {
        // d4=27, d5=35. Black rook d8 defends d5.
        // Sequence: Qxd5(+100), Rxd5(-900). Net = -800.
        let b = board("3r4/8/8/3p4/3Q4/8/8/4K2k w - - 0 1");
        let score = see(&b, &cond(), 27, 35, true);
        assert!(score < -700, "queen x pawn defended by rook ≈ -800, got {score}");
    }

    /// Equal exchange: rook captures rook defended by king.
    #[test]
    fn equal_rook_exchange() {
        // d1=3, d8=59. Black king e8 defends d8.
        // Sequence: Rxd8(+500), Kxd8(-500). Net = 0.
        let b = board("3rk3/8/8/8/8/8/8/3R3K w - - 0 1");
        let score = see(&b, &cond(), 3, 59, true);
        assert_eq!(score, 0, "equal rook exchange = 0, got {score}");
    }

    /// Capturing undefended piece is pure gain.
    #[test]
    fn undefended_capture_is_pure_gain() {
        // White rook d1=3 captures black queen d8=59, no black defenders.
        let b = board("3q4/8/8/8/8/8/8/3R3K w - - 0 1");
        let score = see(&b, &cond(), 3, 59, true);
        assert_eq!(score, SEE_QUEEN, "undefended queen = {SEE_QUEEN}, got {score}");
    }

    /// Knight captures pawn defended by pawn: losing (-200).
    #[test]
    fn knight_captures_pawn_defended_by_pawn() {
        // White knight c3=18 captures black pawn d5=35, defended by black pawn e6=44.
        // Sequence: Nxd5(+100), exd5(-300). Net = -200.
        let b = board("4k3/8/4p3/3p4/8/2N5/8/4K3 w - - 0 1");
        let score = see(&b, &cond(), 18, 35, true);
        assert!(score < -150, "knight x pawn defended by pawn ≈ -200, got {score}");
    }

    /// Symmetry: same position from black's perspective.
    #[test]
    fn see_symmetric_for_black() {
        // Black knight captures white pawn on d4, defended by white pawn e3.
        // c6=42 (black knight), d4=27 (white pawn), e3=20 (white pawn).
        let b = board("4k3/8/2n5/8/3P4/4P3/8/4K3 b - - 0 1");
        let score = see(&b, &cond(), 42, 27, false);
        assert!(score < -150, "black knight x pawn defended by pawn ≈ -200, got {score}");
    }

    /// X-ray: rook behind a rook is revealed after first exchange.
    #[test]
    fn xray_rook_behind_rook() {
        // White: Ra1=0, Rb1=1 (second rook behind on b-file... actually use same file).
        // White Ra1 and Ra2 on a-file, black Ra8 on a-file (defended by nothing).
        // Ra1xa8: opponent has nothing, gain = SEE_ROOK.
        // More interesting: white Ra1 takes black Ra8; black Rb8 takes back; white Ra2 takes Rb8.
        // Net: +500 - 500 + 500 = +500. White ends up with +500 net from sequence.
        // a1=0, a8=56, a2=8, b8=57
        let b = board("rr6/8/8/8/8/8/R7/R6K w - - 0 1");
        // White Ra1(0) x black Ra8(56): black Rb8(57) recaptures; white Ra2(8) recaptures.
        let score = see(&b, &cond(), 0, 56, true);
        assert!(score > 400, "rook x rook with x-ray backup ≈ +500, got {score}");
    }
}
