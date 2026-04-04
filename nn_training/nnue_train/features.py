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


# ── HalfKAv2 features ─────────────────────────────────────────────────────
#
# Feature space: 11 piece types × 64 squares × 64 exact king squares = 45,056
#
# Key differences from HalfKP:
#   - Own king is EXCLUDED from the piece set (11 slots, not 12)
#   - Opponent king IS included as slot 10
#   - King square is exact (0-63) rather than a coarse bucket
#
# Feature index: slot * 64 * 64 + piece_square * 64 + king_square
#
# Horizontal mirroring: same rule as HalfKP — when the king is on files e-h
# (file index 4-7), both king_sq and piece_sq are file-flipped (XOR 7) so
# the model sees left-right mirror positions as identical.

HALFKAV2_NUM_PIECE_SLOTS = 11   # all pieces except own king
HALFKAV2_FEATURE_DIM = HALFKAV2_NUM_PIECE_SLOTS * 64 * 64  # 45,056

# Piece slot (from "own side" perspective after board mirror for black POV).
# Slots 0–4: own non-king pieces (P, N, B, R, Q).
# Slots 5–10: their pieces including king (P, N, B, R, Q, K).
# Own king is intentionally absent — it is not a HalfKAv2 feature.
_PIECE_SLOT_HALFKAV2 = {
    (chess.PAWN,   True):  0,   # own pawn
    (chess.KNIGHT, True):  1,   # own knight
    (chess.BISHOP, True):  2,   # own bishop
    (chess.ROOK,   True):  3,   # own rook
    (chess.QUEEN,  True):  4,   # own queen
    # (chess.KING,  True): excluded — own king never a feature in HalfKAv2
    (chess.PAWN,   False): 5,   # their pawn
    (chess.KNIGHT, False): 6,   # their knight
    (chess.BISHOP, False): 7,   # their bishop
    (chess.ROOK,   False): 8,   # their rook
    (chess.QUEEN,  False): 9,   # their queen
    (chess.KING,   False): 10,  # their king
}


def encode_board_halfkav2_dual(board: chess.Board) -> tuple[np.ndarray, np.ndarray]:
    """Encode board using HalfKAv2 dual-perspective features.

    Feature: (piece_slot, piece_square, own_king_square)
    All pieces EXCEPT own king are encoded; opponent king is slot 10.
    Feature dim = 11 × 64 × 64 = 45,056.
    Formula: slot * 64 * 64 + mapped_sq * 64 + king_sq

    Horizontal mirroring: when king is on files e-h, both king_sq and all
    piece squares are file-flipped (XOR 7) to maintain left-right symmetry.

    Returns (white_indices, black_indices), each int64[32] padded with
    HALFKAV2_FEATURE_DIM (45,056) as sentinel for unused slots.

    CP targets must be white-absolute (same convention as HalfKP dual).
    """
    SENTINEL = HALFKAV2_FEATURE_DIM  # 45,056

    def _encode_white_pov(b: chess.Board) -> np.ndarray:
        """Encode from white's perspective: white king = 'ours' (excluded)."""
        king_squares = b.pieces(chess.KING, chess.WHITE)
        if not king_squares:
            return np.full(32, SENTINEL, dtype=np.int64)
        king_sq = next(iter(king_squares))
        mirror = chess.square_file(king_sq) >= 4
        if mirror:
            king_sq = king_sq ^ 7  # flip file bits; rank bits unchanged

        result = np.full(32, SENTINEL, dtype=np.int64)
        count = 0
        for square, piece in b.piece_map().items():
            # Own (white) king: not a feature in HalfKAv2
            if piece.piece_type == chess.KING and piece.color == chess.WHITE:
                continue
            if count >= 32:
                break
            slot = _PIECE_SLOT_HALFKAV2[(piece.piece_type, piece.color)]
            mapped_sq = square ^ 7 if mirror else square
            result[count] = slot * 64 * 64 + mapped_sq * 64 + king_sq
            count += 1
        return result

    # White perspective: board as-is (white king = ours)
    white_indices = _encode_white_pov(board)
    # Black perspective: mirror board (black king becomes white king = ours)
    black_indices = _encode_white_pov(board.mirror())
    return white_indices, black_indices


# ── WDL target ────────────────────────────────────────────────────────────

def cp_to_wdl_target(cp: float, ply: int = 40) -> np.ndarray:
    """Map centipawn score to soft WDL distribution.

    Uses Stockfish's calibrated win-rate model (from nnue-pytorch), adapted
    from SF internal units to centipawns via the empirical scale factor 2.96
    (verified: a position stored as 713 SF units = +241cp by live Stockfish).

    The ply parameter makes the model phase-aware:
      - Same CP value means higher win probability in the endgame than opening
      - Defaults to ply=40 (early middlegame) when not known
    """
    a, b = _wdl_params(float(ply))
    win  = 1.0 / (1.0 + np.exp(np.clip(-(cp - b) / a, -500, 500)))
    loss = 1.0 / (1.0 + np.exp(np.clip(-(-cp - b) / a, -500, 500)))
    draw = max(0.0, 1.0 - win - loss)
    return np.array([win, draw, loss], dtype=np.float32)


def cp_to_wdl_batch(cp: np.ndarray, ply: np.ndarray | None = None) -> np.ndarray:
    """Vectorised cp_to_wdl_target. Returns (N, 3) float32.

    ply: per-position ply array. If None, defaults to 40 for all positions.
    Use piece-count proxy when ply is unavailable:
        ply_est = np.clip((32 - piece_count) * 4, 0, 240)
    """
    cp = np.asarray(cp, dtype=np.float32)
    if ply is None:
        ply_f = np.full_like(cp, 40.0)
    else:
        ply_f = np.clip(np.asarray(ply, dtype=np.float32), 0.0, 240.0)
    a, b = _wdl_params(ply_f)
    win  = 1.0 / (1.0 + np.exp(np.clip(-(cp - b) / a, -500, 500)))
    loss = 1.0 / (1.0 + np.exp(np.clip(-(-cp - b) / a, -500, 500)))
    draw = np.clip(1.0 - win - loss, 0.0, None)
    return np.stack([win, draw, loss], axis=1).astype(np.float32)


# ── Stockfish win-rate model (internal) ───────────────────────────────────────
#
# Polynomial coefficients from nnue-pytorch (nnue_training_data_formats.h).
# Originally calibrated to Stockfish internal units; adapted to centipawns
# by dividing by _SF_TO_CP = 2.96 (empirically measured from test80 data:
# a position stored as 713 SF units ↔ +241 cp displayed by Stockfish 15/16).
#
# Effect: same CP eval → higher win probability later in the game (smaller a).
#   +100 cp at ply  0 → ~67% win
#   +100 cp at ply 40 → ~70% win
#   +100 cp at ply 100 → ~74% win

_SF_TO_CP = 2.96

_A_COEFFS = np.array([
    -3.68389304 / _SF_TO_CP,
     30.07065921 / _SF_TO_CP,
    -60.52878723 / _SF_TO_CP,
     149.53378557 / _SF_TO_CP,
], dtype=np.float64)

_B_COEFFS = np.array([
    -2.0181857  / _SF_TO_CP,
     15.85685038 / _SF_TO_CP,
    -29.83452023 / _SF_TO_CP,
     47.59078827 / _SF_TO_CP,
], dtype=np.float64)


def _wdl_params(ply):
    """Return (a, b) WDL model parameters in centipawns. ply may be scalar or array."""
    m = np.clip(ply, 0.0, 240.0) / 64.0
    a = ((_A_COEFFS[0] * m + _A_COEFFS[1]) * m + _A_COEFFS[2]) * m + _A_COEFFS[3]
    b = ((_B_COEFFS[0] * m + _B_COEFFS[1]) * m + _B_COEFFS[2]) * m + _B_COEFFS[3]
    return a.astype(np.float32), b.astype(np.float32)
