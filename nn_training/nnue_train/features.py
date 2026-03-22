from __future__ import annotations

import numpy as np
import chess

# ── Original 12×64 features ───────────────────────────────────────────────

PIECE_OFFSETS = {
    (chess.PAWN, True): 0,
    (chess.KNIGHT, True): 64,
    (chess.BISHOP, True): 128,
    (chess.ROOK, True): 192,
    (chess.QUEEN, True): 256,
    (chess.KING, True): 320,
    (chess.PAWN, False): 384,
    (chess.KNIGHT, False): 448,
    (chess.BISHOP, False): 512,
    (chess.ROOK, False): 576,
    (chess.QUEEN, False): 640,
    (chess.KING, False): 704,
}

FEATURE_DIM = 768


def encode_board_12x64(board: chess.Board) -> np.ndarray:
    """Encode board to a 12x64 one-hot flat feature vector.

    Features are from white's board orientation; if black is to move we mirror
    and swap colors so the model learns side-to-move normalized positions.
    """
    x = np.zeros(FEATURE_DIM, dtype=np.float32)

    if board.turn == chess.BLACK:
        board = board.mirror()

    for square, piece in board.piece_map().items():
        offset = PIECE_OFFSETS[(piece.piece_type, piece.color)]
        x[offset + square] = 1.0

    return x


# ── King-bucketed HalfKP features ─────────────────────────────────────────
#
# Feature space: 12 piece types × 64 squares × 32 king buckets = 24,576
#
# King bucket layout (from side-to-move's perspective after board mirror):
#   Files are collapsed to 4 buckets (queenside mirror): file 0-3 → 0,1,2,3
#                                                         file 4-7 → 3,2,1,0
#   Ranks split into 4 quarters: rank // 2  →  0,0,1,1,2,2,3,3
#   bucket = rank_quarter * 4 + file_bucket  →  0..15 per quarter = 0..31 total
#
# Layout per feature:
#   feature_index = piece_slot * 64 * 32 + piece_square * 32 + king_bucket
# where piece_slot is 0..11 (same ordering as 12x64 above).

KING_BUCKETS = 32          # rank quarters (4) × mirrored files (4)
NUM_PIECE_SLOTS = 12       # 6 piece types × 2 colours
HALFKP_FEATURE_DIM = NUM_PIECE_SLOTS * 64 * KING_BUCKETS  # 24,576

# Precompute king bucket for each of the 64 squares (from side-to-move view).
# Files 0-3 map directly; files 4-7 are mirrored onto 3-0.
# Ranks split into quarters (rank // 2) instead of halves for finer resolution.
def _build_king_bucket_table() -> list[int]:
    table = []
    for sq in range(64):
        file = sq % 8
        rank = sq // 8
        file_bucket = file if file <= 3 else 7 - file
        rank_quarter = rank // 2
        table.append(rank_quarter * 4 + file_bucket)
    return table

KING_BUCKET = _build_king_bucket_table()

# Piece type → slot index 0..11 (ours 0..5, theirs 6..11)
_PIECE_SLOT = {
    (chess.PAWN,   True):  0,
    (chess.KNIGHT, True):  1,
    (chess.BISHOP, True):  2,
    (chess.ROOK,   True):  3,
    (chess.QUEEN,  True):  4,
    (chess.KING,   True):  5,
    (chess.PAWN,   False): 6,
    (chess.KNIGHT, False): 7,
    (chess.BISHOP, False): 8,
    (chess.ROOK,   False): 9,
    (chess.QUEEN,  False): 10,
    (chess.KING,   False): 11,
}


def encode_board_halfkp_dual(board: chess.Board) -> tuple[np.ndarray, np.ndarray]:
    """Encode board from both white and black absolute perspectives.

    Unlike encode_board_halfkp, no side-to-move mirroring is applied here.
    The white perspective always has white king as 'ours'; the black perspective
    mirrors the board first (so black king becomes 'ours') then encodes identically.

    Horizontal mirroring: when the king is on files e-h (file index 4-7), all
    piece square files are flipped (sq ^ 7 flips bits 0-2, preserving rank bits).
    This ensures that king-on-a1 and king-on-h1 see identical feature distributions.

    Returns (white_indices, black_indices), each an int64 array of length 32
    padded with sentinel HALFKP_FEATURE_DIM (12288).

    CP targets for dual-perspective training must be white-absolute
    (positive = good for white), not side-to-move perspective.
    """
    SENTINEL = HALFKP_FEATURE_DIM  # 12288

    def _encode_white_pov(b: chess.Board) -> np.ndarray:
        """Encode from white's view: white king = 'ours', no side-to-move flip.
        Applies horizontal file-mirroring when white king is on files e-h."""
        king_squares = b.pieces(chess.KING, chess.WHITE)
        if not king_squares:
            return np.full(32, SENTINEL, dtype=np.int64)
        king_sq = next(iter(king_squares))
        bucket = KING_BUCKET[king_sq]
        mirror = chess.square_file(king_sq) >= 4

        result = np.full(32, SENTINEL, dtype=np.int64)
        count = 0
        for square, piece in b.piece_map().items():
            if count >= 32:
                break
            slot = _PIECE_SLOT[(piece.piece_type, piece.color)]
            mapped_sq = square ^ 7 if mirror else square
            result[count] = slot * 64 * KING_BUCKETS + mapped_sq * KING_BUCKETS + bucket
            count += 1
        return result

    # White perspective: board as-is (white king = ours)
    white_indices = _encode_white_pov(board)
    # Black perspective: mirror board (black king becomes white king = ours)
    black_indices = _encode_white_pov(board.mirror())
    return white_indices, black_indices


def encode_board_halfkp(board: chess.Board) -> np.ndarray:
    """Encode board to a 12,288-dim sparse feature vector.

    Same side-to-move normalization as encode_board_12x64: mirror the board
    when black is to move so the model always sees it from the mover's view.

    Feature index = piece_slot * 64 * 16 + piece_square * 16 + king_bucket
    where king_bucket is derived from the side-to-move king's position.
    """
    x = np.zeros(HALFKP_FEATURE_DIM, dtype=np.float32)

    if board.turn == chess.BLACK:
        board = board.mirror()

    # Find the side-to-move king (always white after mirror)
    king_squares = board.pieces(chess.KING, chess.WHITE)
    if not king_squares:
        return x
    king_sq = next(iter(king_squares))
    bucket = KING_BUCKET[king_sq]

    for square, piece in board.piece_map().items():
        slot = _PIECE_SLOT[(piece.piece_type, piece.color)]
        idx = slot * 64 * KING_BUCKETS + square * KING_BUCKETS + bucket
        x[idx] = 1.0

    return x


# ── WDL target ────────────────────────────────────────────────────────────

def cp_to_wdl_target(cp: float) -> np.ndarray:
    """Map centipawn score to soft WDL distribution.

    Logistic mapping around 0cp. Draw probability is highest near equality.
    """
    p_win = 1.0 / (1.0 + np.exp(-cp / 180.0))
    p_loss = 1.0 - p_win
    draw = max(0.0, 1.0 - abs(cp) / 800.0)

    # Renormalize to sum=1, nudging win/loss toward non-draw mass.
    non_draw = 1.0 - draw
    p_win *= non_draw
    p_loss *= non_draw

    s = p_win + draw + p_loss
    return np.array([p_win / s, draw / s, p_loss / s], dtype=np.float32)
