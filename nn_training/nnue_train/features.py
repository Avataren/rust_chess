from __future__ import annotations

import numpy as np
import chess

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
