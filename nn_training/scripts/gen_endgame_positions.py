#!/usr/bin/env python3
"""Generate random legal chess positions with a given piece count range.

Outputs one FEN per line, suitable for labeling with generate_data.py.
For positions with ≤7 pieces, Stockfish will use Syzygy tablebases
and return perfect evaluations in milliseconds.

Usage:
  python scripts/gen_endgame_positions.py \
      --min-pieces 2 --max-pieces 5 \
      --count 3500000 \
      --output data/new_fens/synthetic_b0.fens

  python scripts/gen_endgame_positions.py \
      --min-pieces 2 --max-pieces 7 \
      --count 1000000 --seed 42 \
      --output data/new_fens/synthetic_endgame.fens
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import chess
from tqdm import tqdm

# ── Piece pool (excluding kings, which are always present) ────────────────────

PIECE_TYPES = [
    chess.PAWN, chess.KNIGHT, chess.BISHOP,
    chess.ROOK, chess.QUEEN,
]


def _random_piece_config(min_extra: int, max_extra: int, rng: random.Random) -> list[tuple[chess.PieceType, chess.Color]]:
    """Return a random list of (piece_type, color) to place beyond the two kings.

    Avoids obviously drawn positions (bare kings) unless min_extra=0.
    Balanced between sides: white gets ceil(n/2), black gets floor(n/2).
    """
    n = rng.randint(min_extra, max_extra)
    if n == 0:
        return []

    pieces = []
    # Alternate colors to keep material roughly balanced
    for i in range(n):
        color = chess.WHITE if i % 2 == 0 else chess.BLACK
        pt = rng.choice(PIECE_TYPES)
        pieces.append((pt, color))
    rng.shuffle(pieces)
    return pieces


def _try_generate_position(
    min_pieces: int,
    max_pieces: int,
    rng: random.Random,
    max_attempts: int = 200,
) -> chess.Board | None:
    """Try to generate a legal position with piece count in [min_pieces, max_pieces]."""
    # total pieces = 2 kings + extra
    min_extra = max(0, min_pieces - 2)
    max_extra = max(0, max_pieces - 2)

    for _ in range(max_attempts):
        board = chess.Board(fen=None)  # empty board
        squares = list(range(64))
        rng.shuffle(squares)
        sq_iter = iter(squares)

        # Place kings — must not be adjacent
        wk = next(sq_iter)
        bk = next(sq_iter)
        # Ensure kings aren't adjacent
        attempts = 0
        while chess.square_distance(wk, bk) < 2:
            bk = next(sq_iter, None)
            if bk is None:
                break
            attempts += 1
        if bk is None or chess.square_distance(wk, bk) < 2:
            continue

        board.set_piece_at(wk, chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(bk, chess.Piece(chess.KING, chess.BLACK))

        # Extra pieces
        extra = _random_piece_config(min_extra, max_extra, rng)
        ok = True
        for pt, color in extra:
            sq = next(sq_iter, None)
            if sq is None:
                ok = False
                break
            # Pawns can't be on rank 1 or 8
            if pt == chess.PAWN and chess.square_rank(sq) in (0, 7):
                ok = False
                break
            board.set_piece_at(sq, chess.Piece(pt, color))

        if not ok:
            continue

        # Set random side to move
        board.turn = rng.choice([chess.WHITE, chess.BLACK])
        board.castling_rights = chess.BB_EMPTY  # no castling in endgames
        board.ep_square = None

        # Validate: side NOT to move must not be in check,
        # and position must have legal moves
        board.turn = not board.turn  # check opposite side
        if board.is_check():
            board.turn = not board.turn
            continue
        board.turn = not board.turn

        if board.is_valid() and not board.is_game_over():
            return board

    return None


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate random legal endgame positions as FENs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--min-pieces", type=int, default=2)
    ap.add_argument("--max-pieces", type=int, default=5)
    ap.add_argument("--count",      type=int, default=1_000_000,
                    help="Number of positions to generate")
    ap.add_argument("--output",     required=True,
                    help="Output FEN file (one FEN per line)")
    ap.add_argument("--seed",       type=int, default=None,
                    help="Random seed for reproducibility")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    generated = 0
    failed = 0

    with open(out, "w") as f:
        pbar = tqdm(total=args.count, desc="generating", unit="pos", dynamic_ncols=True)
        while generated < args.count:
            board = _try_generate_position(args.min_pieces, args.max_pieces, rng)
            if board is None:
                failed += 1
                if failed > args.count * 2:
                    print(f"\nToo many failures ({failed}). "
                          f"Try wider piece range or fewer positions.", file=sys.stderr)
                    break
                continue
            f.write(board.fen() + "\n")
            generated += 1
            pbar.update(1)
        pbar.close()

    print(f"Generated {generated:,} positions → {out}  (failures: {failed:,})")


if __name__ == "__main__":
    main()
