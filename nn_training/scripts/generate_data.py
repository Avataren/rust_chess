#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import random
from pathlib import Path

import chess
import chess.engine
import chess.pgn
from tqdm import tqdm


def evaluate_with_stockfish(
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    depth: int,
) -> float:
    info = engine.analyse(board, chess.engine.Limit(depth=depth))
    score = info["score"].pov(board.turn)
    cp = score.score(mate_score=10000)
    return float(cp)


def _iter_pgn_positions(pgn_path: Path, max_positions: int, plies_min: int, min_elo: int):
    """Yield sampled chess.Board positions from a PGN file.

    Uses a line-level state machine so move text is only parsed for games
    that pass the Elo filter — skipped games cost only a readline, not a
    full python-chess parse.  This is ~50-100x faster than read_game() on
    large files with a strict Elo threshold.
    """
    # States
    HEADER = 0   # reading PGN header tags
    COLLECT = 1  # reading move text for a qualifying game
    SKIP = 2     # skipping move text for a disqualified game

    state = HEADER
    w_elo = 0
    b_elo = 0
    game_lines: list[str] = []
    found = 0
    scanned = 0
    skipped = 0

    with pgn_path.open("r", encoding="utf-8", errors="ignore", buffering=1 << 20) as f:
        for raw_line in f:
            if found >= max_positions:
                break

            stripped = raw_line.strip()

            # ── HEADER state ──────────────────────────────────────────────
            if state == HEADER:
                if stripped.startswith("["):
                    game_lines.append(raw_line)
                    if stripped.startswith("[WhiteElo "):
                        try:
                            w_elo = int(stripped.split('"')[1])
                        except (IndexError, ValueError):
                            w_elo = 0
                    elif stripped.startswith("[BlackElo "):
                        try:
                            b_elo = int(stripped.split('"')[1])
                        except (IndexError, ValueError):
                            b_elo = 0
                elif stripped == "":
                    # Blank line: end of header block — decide whether to keep
                    if min_elo > 0 and (w_elo < min_elo or b_elo < min_elo):
                        state = SKIP
                        game_lines = []
                        w_elo = b_elo = 0
                        skipped += 1
                    else:
                        state = COLLECT
                        # blank line is part of valid PGN; include it
                        game_lines.append(raw_line)

            # ── COLLECT state ─────────────────────────────────────────────
            elif state == COLLECT:
                if stripped == "":
                    # Blank line: end of move text — parse and sample
                    scanned += 1
                    try:
                        game = chess.pgn.read_game(io.StringIO("".join(game_lines)))
                        if game is not None:
                            board = game.board()
                            states: list[chess.Board] = []
                            for i, mv in enumerate(game.mainline_moves()):
                                board.push(mv)
                                if i >= plies_min:
                                    states.append(board.copy())
                            if states:
                                yield random.choice(states)
                                found += 1
                    except Exception:
                        pass
                    state = HEADER
                    game_lines = []
                    w_elo = b_elo = 0
                else:
                    game_lines.append(raw_line)

            # ── SKIP state ────────────────────────────────────────────────
            elif state == SKIP:
                if stripped == "":
                    # End of skipped game's move text — back to header mode
                    state = HEADER

    # Handle file that ends without a trailing blank line
    if state == COLLECT and game_lines and found < max_positions:
        try:
            game = chess.pgn.read_game(io.StringIO("".join(game_lines)))
            if game is not None:
                board = game.board()
                states = []
                for i, mv in enumerate(game.mainline_moves()):
                    board.push(mv)
                    if i >= plies_min:
                        states.append(board.copy())
                if states:
                    yield random.choice(states)
        except Exception:
            pass

    print(f"PGN scan: {found} positions collected, {skipped} games skipped (Elo < {min_elo})")


def selfplay_positions(
    engine: chess.engine.SimpleEngine,
    games: int,
    movetime_ms: int,
    min_ply: int,
    max_ply: int,
    positions_per_game: int = 1,
) -> list[chess.Board]:
    out: list[chess.Board] = []
    for _ in tqdm(range(games), desc="selfplay"):
        board = chess.Board()
        states: list[chess.Board] = []
        for ply in range(max_ply):
            if board.is_game_over():
                break
            result = engine.play(board, chess.engine.Limit(time=movetime_ms / 1000.0))
            board.push(result.move)
            if ply >= min_ply:
                states.append(board.copy())
        if states:
            k = min(positions_per_game, len(states))
            out.extend(random.sample(states, k))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--label-engine",
        required=True,
        help="Path to UCI engine used for evaluation labels (typically Stockfish)",
    )
    ap.add_argument(
        "--selfplay-engine",
        help=(
            "Optional path to UCI engine used for self-play move generation. "
            "If omitted, the label engine is used."
        ),
    )
    ap.add_argument("--output", required=True, help="Output JSONL path")
    ap.add_argument("--pgn", help="Optional PGN source")
    ap.add_argument("--min-elo", type=int, default=2200,
                    help="Minimum Elo for both players when sampling from PGN (default 2200, 0 to disable)")
    ap.add_argument("--max-positions", type=int, default=200000)
    ap.add_argument("--eval-depth", type=int, default=12)
    ap.add_argument("--selfplay-games", type=int, default=50000)
    ap.add_argument("--selfplay-movetime-ms", type=int, default=20)
    ap.add_argument("--positions-per-game", type=int, default=1,
                    help="Positions sampled per self-play game (default 1). "
                         "Increase to get more data from fewer games.")
    args = ap.parse_args()

    random.seed(42)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    label_engine = chess.engine.SimpleEngine.popen_uci(args.label_engine)
    selfplay_engine = (
        chess.engine.SimpleEngine.popen_uci(args.selfplay_engine)
        if args.selfplay_engine
        else label_engine
    )

    pgn_cap = args.max_positions if args.selfplay_games == 0 else args.max_positions // 2

    written = 0
    with output.open("w", encoding="utf-8") as out_f:

        # ── PGN path: stream scan → label → write ────────────────────────
        if args.pgn and pgn_cap > 0:
            pbar = tqdm(total=pgn_cap, desc="pgn-label")
            for board in _iter_pgn_positions(
                Path(args.pgn), pgn_cap, plies_min=12, min_elo=args.min_elo
            ):
                cp = evaluate_with_stockfish(label_engine, board, depth=args.eval_depth)
                out_f.write(json.dumps({"fen": board.fen(), "cp": cp}) + "\n")
                written += 1
                pbar.update(1)
            pbar.close()

        # ── Self-play path ────────────────────────────────────────────────
        selfplay_cap = args.max_positions - written
        if args.selfplay_games > 0 and selfplay_cap > 0:
            boards = selfplay_positions(
                selfplay_engine,
                games=args.selfplay_games,
                movetime_ms=args.selfplay_movetime_ms,
                min_ply=12,
                max_ply=180,
                positions_per_game=args.positions_per_game,
            )
            random.shuffle(boards)
            boards = boards[:selfplay_cap]
            for board in tqdm(boards, desc="selfplay-label"):
                cp = evaluate_with_stockfish(label_engine, board, depth=args.eval_depth)
                out_f.write(json.dumps({"fen": board.fen(), "cp": cp}) + "\n")
                written += 1

    if selfplay_engine is not label_engine:
        selfplay_engine.quit()
    label_engine.quit()
    print(f"Wrote {written} samples to {output}")


if __name__ == "__main__":
    main()
