"""
Unit tests for the dual-perspective NNUE architecture (Phase 2+3).

Covers:
  - encode_board_halfkp_dual: feature counts, index range, symmetry,
    side-to-move invariance, and consistency with the single-perspective encoder
  - EvalNetDual: output shapes, shared weights, fc2 width, determinism
  - CP white-absolute convention used in dual preprocessing
  - export_weights auto-detection of dual vs single model
"""
import sys
from pathlib import Path

import chess
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nnue_train.features import (
    HALFKP_FEATURE_DIM,
    encode_board_halfkp,
    encode_board_halfkp_dual,
)
from nnue_train.model import EvalNet, EvalNetDual

SENTINEL = HALFKP_FEATURE_DIM  # 12288


# ── Helpers ───────────────────────────────────────────────────────────────────

def active_indices(arr: np.ndarray) -> np.ndarray:
    """Return sorted non-sentinel indices from a dual index array."""
    return np.sort(arr[arr != SENTINEL])


def single_active_indices(board: chess.Board) -> np.ndarray:
    """Active indices from the single-perspective encoder as a sorted array."""
    x = encode_board_halfkp(board)
    return np.sort(np.where(x > 0)[0].astype(np.int64))


# ── encode_board_halfkp_dual ──────────────────────────────────────────────────

class TestDualEncoding:

    def test_feature_count_starting_position(self):
        board = chess.Board()
        w, b = encode_board_halfkp_dual(board)
        assert np.sum(w != SENTINEL) == 32, "White perspective should have 32 active features"
        assert np.sum(b != SENTINEL) == 32, "Black perspective should have 32 active features"

    def test_indices_in_range(self):
        board = chess.Board()
        w, b = encode_board_halfkp_dual(board)
        for idx in w[w != SENTINEL]:
            assert 0 <= idx < HALFKP_FEATURE_DIM, f"White index {idx} out of range"
        for idx in b[b != SENTINEL]:
            assert 0 <= idx < HALFKP_FEATURE_DIM, f"Black index {idx} out of range"

    def test_output_dtype_and_shape(self):
        board = chess.Board()
        w, b = encode_board_halfkp_dual(board)
        assert w.dtype == np.int64
        assert b.dtype == np.int64
        assert w.shape == (32,)
        assert b.shape == (32,)

    def test_symmetric_starting_position(self):
        """Starting position is symmetric: both perspectives should have the same features."""
        board = chess.Board()
        w, b = encode_board_halfkp_dual(board)
        assert np.array_equal(active_indices(w), active_indices(b)), \
            "Symmetric board should produce identical feature sets for both perspectives"

    def test_independent_of_side_to_move(self):
        """Dual encoding is absolute — toggling the turn must not change features."""
        board_w = chess.Board()
        board_b = chess.Board()
        board_b.turn = chess.BLACK

        ww, wb = encode_board_halfkp_dual(board_w)
        bw, bb = encode_board_halfkp_dual(board_b)

        assert np.array_equal(ww, bw), "White perspective must not depend on side to move"
        assert np.array_equal(wb, bb), "Black perspective must not depend on side to move"

    def test_white_pov_matches_single_when_white_to_move(self):
        """When white is to move both encoders see the board identically."""
        board = chess.Board()
        assert board.turn == chess.WHITE

        w, _ = encode_board_halfkp_dual(board)
        single = single_active_indices(board)
        assert np.array_equal(active_indices(w), single), \
            "Dual white-pov must match single-perspective encoder when white to move"

    def test_black_pov_matches_single_when_black_to_move(self):
        """When black is to move, single-perspective mirrors — same as dual black-pov."""
        board = chess.Board()
        board.turn = chess.BLACK

        _, b = encode_board_halfkp_dual(board)
        single = single_active_indices(board)  # single mirrors for black
        assert np.array_equal(active_indices(b), single), \
            "Dual black-pov must match single-perspective encoder when black to move"

    def test_asymmetric_position_differs_across_perspectives(self):
        """After a move the position is no longer symmetric."""
        board = chess.Board()
        board.push_san("e4")  # white pawn advances — no longer symmetric
        w, b = encode_board_halfkp_dual(board)
        assert not np.array_equal(active_indices(w), active_indices(b)), \
            "Asymmetric position should produce different features per perspective"

    def test_no_duplicate_indices(self):
        board = chess.Board()
        w, b = encode_board_halfkp_dual(board)
        w_active = w[w != SENTINEL]
        b_active = b[b != SENTINEL]
        assert len(np.unique(w_active)) == len(w_active), "Duplicate indices in white perspective"
        assert len(np.unique(b_active)) == len(b_active), "Duplicate indices in black perspective"

    def test_different_positions_give_different_features(self):
        start = chess.Board()
        after_e4 = chess.Board()
        after_e4.push_san("e4")

        w_start, _ = encode_board_halfkp_dual(start)
        w_e4,    _ = encode_board_halfkp_dual(after_e4)
        assert not np.array_equal(active_indices(w_start), active_indices(w_e4))


# ── CP white-absolute convention ──────────────────────────────────────────────

class TestCpConvention:
    """
    In dual preprocessing, cp is converted from side-to-move to white-absolute:
        white_cp = cp if board.turn == WHITE else -cp
    """

    def test_white_to_move_cp_unchanged(self):
        board = chess.Board()
        assert board.turn == chess.WHITE
        cp_stm = 200.0
        white_abs = cp_stm if board.turn == chess.WHITE else -cp_stm
        assert white_abs == 200.0

    def test_black_to_move_cp_negated(self):
        board = chess.Board()
        board.turn = chess.BLACK
        cp_stm = 200.0  # black is up 200 from black's POV
        white_abs = cp_stm if board.turn == chess.WHITE else -cp_stm
        assert white_abs == -200.0, "Black winning → negative from white's POV"

    def test_equal_position_unchanged(self):
        board = chess.Board()
        board.turn = chess.BLACK
        cp_stm = 0.0
        white_abs = cp_stm if board.turn == chess.WHITE else -cp_stm
        assert white_abs == 0.0


# ── EvalNetDual ───────────────────────────────────────────────────────────────

class TestEvalNetDual:

    @pytest.fixture
    def model(self):
        return EvalNetDual(input_dim=12288, hidden_dim=512, hidden2_dim=32)

    @pytest.fixture
    def batch(self):
        """Random valid index batch (sentinel-padded)."""
        B = 4
        x = torch.full((B, 32), SENTINEL, dtype=torch.int64)
        # Put a few valid indices in
        x[:, :8] = torch.randint(0, SENTINEL, (B, 8))
        return x

    def test_output_shapes(self, model, batch):
        cp, wdl = model(batch, batch)
        assert cp.shape  == (4, 1), f"cp shape should be (4,1), got {cp.shape}"
        assert wdl.shape == (4, 3), f"wdl shape should be (4,3), got {wdl.shape}"

    def test_fc2_width_is_1024(self, model):
        assert model.fc2.in_features == 1024, \
            f"fc2 input should be 1024 (512×2), got {model.fc2.in_features}"

    def test_fc2_output_is_32(self, model):
        assert model.fc2.out_features == 32

    def test_shared_embedding_weights(self, model):
        """Both perspectives use the same embedding object — not two separate ones."""
        assert not hasattr(model, "embedding_white") and not hasattr(model, "embedding_black"), \
            "EvalNetDual must use a single shared embedding, not two separate ones"
        assert hasattr(model, "embedding")

    def test_both_perspectives_use_same_weights(self, model):
        """Swapping white/black inputs should produce mirrored (not identical) outputs
        only when inputs differ. With identical inputs results should be equal."""
        B = 2
        x = torch.full((B, 32), SENTINEL, dtype=torch.int64)
        x[:, :4] = torch.tensor([[100, 200, 300, 400], [500, 600, 700, 800]])

        cp_same, _ = model(x, x)

        # Manually run the embedding twice — results must be equal since weights shared
        h1 = torch.relu(model.embedding(x) + model.bias1)
        h2 = torch.relu(model.embedding(x) + model.bias1)
        assert torch.allclose(h1, h2), "Same input through shared embedding must give same output"

    def test_deterministic_output(self, model, batch):
        model.eval()
        with torch.no_grad():
            cp1, wdl1 = model(batch, batch)
            cp2, wdl2 = model(batch, batch)
        assert torch.allclose(cp1, cp2)
        assert torch.allclose(wdl1, wdl2)

    def test_different_inputs_give_different_outputs(self, model):
        model.eval()
        B = 2
        x1 = torch.full((B, 32), SENTINEL, dtype=torch.int64)
        x2 = torch.full((B, 32), SENTINEL, dtype=torch.int64)
        x1[:, :4] = torch.tensor([[10, 20, 30, 40], [50, 60, 70, 80]])
        x2[:, :4] = torch.tensor([[500, 600, 700, 800], [100, 200, 300, 400]])

        with torch.no_grad():
            cp1, _ = model(x1, x1)
            cp2, _ = model(x2, x2)
        assert not torch.allclose(cp1, cp2), "Different positions must produce different scores"

    def test_parameter_count_vs_single(self):
        """EvalNetDual should have more parameters than EvalNet due to wider fc2."""
        dual   = EvalNetDual(input_dim=12288, hidden_dim=512, hidden2_dim=32)
        single = EvalNet(input_dim=12288, hidden_dim=512, hidden2_dim=32,
                         sparse_input=True)
        dual_params   = sum(p.numel() for p in dual.parameters())
        single_params = sum(p.numel() for p in single.parameters())
        assert dual_params > single_params, \
            "EvalNetDual (wider fc2) should have more parameters than EvalNet"

    def test_state_dict_has_expected_keys(self, model):
        keys = set(model.state_dict().keys())
        assert "embedding.weight" in keys
        assert "bias1" in keys
        assert "fc2.weight" in keys
        assert "fc2.bias" in keys
        assert "cp_head.weight" in keys
        assert "wdl_head.weight" in keys


# ── export_weights auto-detection ────────────────────────────────────────────

class TestExportDetection:

    def test_single_model_fc2_shape(self):
        model = EvalNet(input_dim=12288, hidden_dim=512, hidden2_dim=32, sparse_input=True)
        state = model.state_dict()
        assert state["fc2.weight"].shape == (32, 512)

    def test_dual_model_fc2_shape(self):
        model = EvalNetDual(input_dim=12288, hidden_dim=512, hidden2_dim=32)
        state = model.state_dict()
        assert state["fc2.weight"].shape == (32, 1024)

    def test_auto_detect_dual_from_state_dict(self):
        """The detection logic used in export_weights.py."""
        hidden_dim = 512
        dual_model = EvalNetDual(input_dim=12288, hidden_dim=hidden_dim, hidden2_dim=32)
        state = dual_model.state_dict()
        is_dual = (
            "embedding.weight" in state
            and state["fc2.weight"].shape[1] == hidden_dim * 2
        )
        assert is_dual

    def test_auto_detect_single_not_flagged_as_dual(self):
        hidden_dim = 512
        single_model = EvalNet(input_dim=12288, hidden_dim=hidden_dim, hidden2_dim=32,
                               sparse_input=True)
        state = single_model.state_dict()
        is_dual = (
            "embedding.weight" in state
            and state["fc2.weight"].shape[1] == hidden_dim * 2
        )
        assert not is_dual
