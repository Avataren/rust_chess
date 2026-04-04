"""
Unit tests for the HalfKAv2 dual-perspective encoder.

Covers:
  - encode_board_halfkav2_dual: feature counts, index range, symmetry,
    side-to-move invariance, no duplicates, exact index cross-validation
  - HALFKAV2_FEATURE_DIM constant value
  - Cross-perspective symmetry on symmetric boards
"""
import sys
from pathlib import Path

import chess
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nnue_train.features import (
    HALFKAV2_FEATURE_DIM,
    encode_board_halfkav2_dual,
)

SENTINEL = HALFKAV2_FEATURE_DIM  # 45,056


# ── Helpers ───────────────────────────────────────────────────────────────────

def active_indices(arr: np.ndarray) -> np.ndarray:
    """Return sorted non-sentinel indices from a dual index array."""
    return np.sort(arr[arr != SENTINEL])


# ── Constants ─────────────────────────────────────────────────────────────────

class TestConstants:

    def test_feature_dim_value(self):
        assert HALFKAV2_FEATURE_DIM == 45_056, \
            f"Expected 45056 = 11 * 64 * 64, got {HALFKAV2_FEATURE_DIM}"

    def test_feature_dim_formula(self):
        assert HALFKAV2_FEATURE_DIM == 11 * 64 * 64


# ── encode_board_halfkav2_dual ────────────────────────────────────────────────

class TestHalfKAv2Encoding:

    def test_output_dtype_and_shape(self):
        board = chess.Board()
        w, b = encode_board_halfkav2_dual(board)
        assert w.dtype == np.int64
        assert b.dtype == np.int64
        assert w.shape == (32,)
        assert b.shape == (32,)

    def test_feature_count_starting_position(self):
        # 32 pieces total, own king excluded per perspective → 31 active features each
        board = chess.Board()
        w, b = encode_board_halfkav2_dual(board)
        assert np.sum(w != SENTINEL) == 31, \
            f"White perspective: expected 31 active features, got {np.sum(w != SENTINEL)}"
        assert np.sum(b != SENTINEL) == 31, \
            f"Black perspective: expected 31 active features, got {np.sum(b != SENTINEL)}"

    def test_indices_in_range(self):
        board = chess.Board()
        w, b = encode_board_halfkav2_dual(board)
        for idx in w[w != SENTINEL]:
            assert 0 <= idx < HALFKAV2_FEATURE_DIM, f"White index {idx} out of range"
        for idx in b[b != SENTINEL]:
            assert 0 <= idx < HALFKAV2_FEATURE_DIM, f"Black index {idx} out of range"

    def test_sentinel_value_equals_feature_dim(self):
        # Sentinel must equal feature_dim so the model's embedding clamp works correctly
        assert SENTINEL == HALFKAV2_FEATURE_DIM

    def test_symmetric_starting_position(self):
        """Starting position is symmetric: both perspectives should have the same features."""
        board = chess.Board()
        w, b = encode_board_halfkav2_dual(board)
        assert np.array_equal(active_indices(w), active_indices(b)), \
            "Symmetric board should produce identical feature sets for both perspectives"

    def test_independent_of_side_to_move(self):
        """Dual encoding is absolute — toggling the turn must not change features."""
        board_w = chess.Board()
        board_b = chess.Board()
        board_b.turn = chess.BLACK

        ww, wb = encode_board_halfkav2_dual(board_w)
        bw, bb = encode_board_halfkav2_dual(board_b)

        assert np.array_equal(ww, bw), "White perspective must not depend on side to move"
        assert np.array_equal(wb, bb), "Black perspective must not depend on side to move"

    def test_no_duplicate_indices(self):
        board = chess.Board()
        w, b = encode_board_halfkav2_dual(board)
        w_active = w[w != SENTINEL]
        b_active = b[b != SENTINEL]
        assert len(np.unique(w_active)) == len(w_active), "Duplicate indices in white perspective"
        assert len(np.unique(b_active)) == len(b_active), "Duplicate indices in black perspective"

    def test_asymmetric_position_differs_across_perspectives(self):
        """After a move the position is no longer symmetric."""
        board = chess.Board()
        board.push_san("e4")
        w, b = encode_board_halfkav2_dual(board)
        assert not np.array_equal(active_indices(w), active_indices(b)), \
            "Asymmetric position should produce different features per perspective"

    def test_different_positions_give_different_features(self):
        start = chess.Board()
        after_e4 = chess.Board()
        after_e4.push_san("e4")

        w_start, _ = encode_board_halfkav2_dual(start)
        w_e4, _    = encode_board_halfkav2_dual(after_e4)
        assert not np.array_equal(active_indices(w_start), active_indices(w_e4))

    def test_no_own_king_in_features(self):
        """Own king must never appear as a feature (it is excluded from HalfKAv2)."""
        # There should be exactly 31 features (not 32) from the starting 32-piece position
        board = chess.Board()
        w, b = encode_board_halfkav2_dual(board)
        assert np.sum(w != SENTINEL) == 31, "Own king must be excluded (31 features, not 32)"
        assert np.sum(b != SENTINEL) == 31

    def test_feature_count_minimal_position(self):
        # K vs K: only 1 feature each (their king); own king excluded, so 1 piece left
        board = chess.Board("8/8/4k3/8/8/4K3/8/8 w - - 0 1")
        w, b = encode_board_halfkav2_dual(board)
        assert np.sum(w != SENTINEL) == 1, \
            f"KvK: white perspective should have 1 feature (their king), got {np.sum(w != SENTINEL)}"
        assert np.sum(b != SENTINEL) == 1, \
            f"KvK: black perspective should have 1 feature (their king), got {np.sum(b != SENTINEL)}"

    def test_indices_in_range_various_positions(self):
        fens = [
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
            "r3r1k1/pp3pbp/1qp3p1/2B5/2BP2b1/Q1n2N2/P4PPP/3RR1K1 w - - 0 1",
            "7k/8/8/8/8/8/P7/K7 w - - 0 1",
            "k7/8/8/8/8/8/7P/7K w - - 0 1",
        ]
        for fen in fens:
            board = chess.Board(fen)
            w, b = encode_board_halfkav2_dual(board)
            for idx in w[w != SENTINEL]:
                assert 0 <= idx < HALFKAV2_FEATURE_DIM, \
                    f"FEN {fen}: white index {idx} out of range"
            for idx in b[b != SENTINEL]:
                assert 0 <= idx < HALFKAV2_FEATURE_DIM, \
                    f"FEN {fen}: black index {idx} out of range"


# ── Exact index cross-validation ──────────────────────────────────────────────
#
# These tests pin concrete expected feature indices computed by hand.
# They serve as regression guards and cross-validate the Python encoder against
# the Rust encoder (which has matching constants in neural_eval.rs).

class TestExactIndices:

    def test_white_pov_opponent_king_starting(self):
        """
        Starting position, white perspective:
          White king on e1 = sq 4, file=4 >= 4 → mirror=True, king_sq = 4^7 = 3.
          Black king on e8 = sq 60, slot=10 (their king).
          mapped_sq = 60 ^ 7 = 59 (file-mirrored because mirror=True).
          Expected index = 10 * 64 * 64 + 59 * 64 + 3 = 44739.
        """
        board = chess.Board()
        w, _ = encode_board_halfkav2_dual(board)
        expected = 10 * 64 * 64 + 59 * 64 + 3  # 44739
        assert expected in w, \
            f"Opponent king index {expected} not found in white perspective features"

    def test_black_pov_opponent_king_starting(self):
        """
        Starting position, black perspective (board.mirror()):
          After mirror: black king lands on e1 = sq 4; file=4 >= 4 → mirror=True, king_sq = 4^7 = 3.
          Their king (original white king, now black in mirrored board) on e8 = sq 60.
          slot=10 (their king), mapped_sq = 60^7 = 59.
          Expected index = 10 * 64 * 64 + 59 * 64 + 3 = 44739.
        """
        board = chess.Board()
        _, b = encode_board_halfkav2_dual(board)
        expected = 10 * 64 * 64 + 59 * 64 + 3  # 44739
        assert expected in b, \
            f"Opponent king index {expected} not found in black perspective features"

    def test_white_pawn_a2_starting(self):
        """
        Starting position, white perspective:
          White king on e1, mirror=True, king_sq=3.
          White pawn on a2 = sq 8. slot=0 (own pawn).
          mapped_sq = 8 ^ 7 = 15 (file-mirrored).
          Expected index = 0 * 64 * 64 + 15 * 64 + 3 = 963.
        """
        board = chess.Board()
        w, _ = encode_board_halfkav2_dual(board)
        expected = 0 * 64 * 64 + 15 * 64 + 3  # 963
        assert expected in w, \
            f"White pawn a2 index {expected} not found in white perspective features"

    def test_king_on_a1_no_mirror(self):
        """
        Custom position: white king on a1 = sq 0, file=0 < 4 → mirror=False, king_sq=0.
        White pawn on a2 = sq 8. slot=0 (own pawn).
        mapped_sq = 8 (no mirror).
        Expected index = 0 * 64 * 64 + 8 * 64 + 0 = 512.
        """
        board = chess.Board("7k/8/8/8/8/8/P7/K7 w - - 0 1")
        w, _ = encode_board_halfkav2_dual(board)
        expected = 0 * 64 * 64 + 8 * 64 + 0  # 512
        assert expected in w, \
            f"White pawn a2 index {expected} not found when king on a1"

    def test_king_on_h1_mirror(self):
        """
        Custom position: white king on h1 = sq 7, file=7 >= 4 → mirror=True, king_sq=7^7=0.
        White pawn on h2 = sq 15. slot=0 (own pawn).
        mapped_sq = 15 ^ 7 = 8 (mirrored to a2).
        Expected index = 0 * 64 * 64 + 8 * 64 + 0 = 512.
        (Same as king-on-a1 case — mirror symmetry!)
        """
        board = chess.Board("k7/8/8/8/8/8/7P/7K w - - 0 1")
        w, _ = encode_board_halfkav2_dual(board)
        expected = 0 * 64 * 64 + 8 * 64 + 0  # 512 — same as a1/a2 case
        assert expected in w, \
            f"White pawn h2 index {expected} (mirrored) not found when king on h1"

    def test_mirror_symmetry_a1_vs_h1_king(self):
        """
        Horizontally mirrored positions must produce identical feature sets.
        King+Pawn on a1/a2 vs King+Pawn on h1/h2 should give the same features.
        """
        board_a = chess.Board("7k/8/8/8/8/8/P7/K7 w - - 0 1")
        board_h = chess.Board("k7/8/8/8/8/8/7P/7K w - - 0 1")
        w_a, _ = encode_board_halfkav2_dual(board_a)
        w_h, _ = encode_board_halfkav2_dual(board_h)
        assert np.array_equal(active_indices(w_a), active_indices(w_h)), \
            "Horizontally mirrored positions should produce identical feature sets"
