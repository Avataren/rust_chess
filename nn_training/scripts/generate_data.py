#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import random
from multiprocessing import Pool
from pathlib import Path

import chess
import chess.engine
import chess.pgn
from tqdm import tqdm


# ── Per-worker Stockfish state ─────────────────────────────────────────────

_engine: chess.engine.SimpleEngine | None = None
_eval_depth: int = 12


def _worker_init(stockfish_path: str, depth: int) -> None:
    global _engine, _eval_depth
    _engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    _eval_depth = depth


def _label_fen(fen: str) -> dict | None:
    try:
        board = chess.Board(fen)
        info = _engine.analyse(board, chess.engine.Limit(depth=_eval_depth))
        score = info["score"].pov(board.turn)
        cp = float(score.score(mate_score=10000))
        return {"fen": fen, "cp": cp}
    except Exception:
        return None


# ── PGN position iterator ──────────────────────────────────────────────────

def _iter_pgn_positions(pgn_path: Path, max_positions: int, plies_min: int, min_elo: int, max_elo: int = 0):
    """Yield sampled chess.Board positions from a PGN file.

    Uses a line-level state machine so move text is only parsed for games
    that pass the Elo filter — skipped games cost only a readline, not a
    full python-chess parse.  This is ~50-100x faster than read_game() on
    large files with a strict Elo threshold.
    """
    HEADER = 0
    COLLECT = 1
    SKIP = 2

    state = HEADER
    w_elo = 0
    b_elo = 0
    game_lines: list[str] = []
    found = 0
    skipped = 0

    with pgn_path.open("r", encoding="utf-8", errors="ignore", buffering=1 << 20) as f:
        for raw_line in f:
            if found >= max_positions:
                break

            stripped = raw_line.strip()

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
                    elo_fail = (min_elo > 0 and (w_elo < min_elo or b_elo < min_elo)) or \
                               (max_elo > 0 and (w_elo > max_elo or b_elo > max_elo))
                    if elo_fail:
                        state = SKIP
                        game_lines = []
                        w_elo = b_elo = 0
                        skipped += 1
                    else:
                        state = COLLECT
                        game_lines.append(raw_line)

            elif state == COLLECT:
                if stripped == "":
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

            elif state == SKIP:
                if stripped == "":
                    state = HEADER

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

    print(f"PGN scan: {found} positions collected, {skipped} games skipped (Elo filter)")


# ── Self-play position generator ───────────────────────────────────────────

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


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label-engine", required=True,
                    help="Path to Stockfish (or any UCI engine) for labeling")
    ap.add_argument("--selfplay-engine",
                    help="Optional UCI engine for self-play move generation (default: label engine)")
    ap.add_argument("--selfplay-engine-opt", action="append", default=[], metavar="NAME=VALUE",
                    help="UCI setoption for the self-play engine (repeatable)")
    ap.add_argument("--output", required=True, help="Output JSONL path")
    ap.add_argument("--pgn", help="Optional PGN source file")
    ap.add_argument("--fens", help="Optional pre-extracted FENs file (one FEN per line) — skips PGN scan")
    ap.add_argument("--min-elo", type=int, default=2200,
                    help="Minimum Elo for both players (default 2200, 0 to disable)")
    ap.add_argument("--max-elo", type=int, default=0,
                    help="Maximum Elo for both players (default 0 = no limit)")
    ap.add_argument("--max-positions", type=int, default=200000)
    ap.add_argument("--eval-depth", type=int, default=12)
    ap.add_argument("--workers", type=int, default=1,
                    help="Number of parallel Stockfish labeling workers (default 1)")
    ap.add_argument("--selfplay-games", type=int, default=50000)
    ap.add_argument("--selfplay-movetime-ms", type=int, default=20)
    ap.add_argument("--positions-per-game", type=int, default=1,
                    help="Positions sampled per self-play game (default 1)")
    args = ap.parse_args()

    random.seed(42)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    # ── Collect FENs first (fast, no engine needed) ────────────────────────
    fens: list[str] = []

    pgn_cap = args.max_positions if args.selfplay_games == 0 else args.max_positions // 2

    if args.fens:
        print(f"Loading pre-extracted FENs from {args.fens}...")
        with open(args.fens, "r") as f:
            for line in tqdm(f, desc="loading-fens"):
                fen = line.strip()
                if fen:
                    fens.append(fen)
                    if len(fens) >= pgn_cap:
                        break
    elif args.pgn and pgn_cap > 0:
        print(f"Scanning PGN for up to {pgn_cap} positions...")
        pbar = tqdm(total=pgn_cap, desc="pgn-scan")
        for board in _iter_pgn_positions(
            Path(args.pgn), pgn_cap, plies_min=12,
            min_elo=args.min_elo, max_elo=args.max_elo,
        ):
            fens.append(board.fen())
            pbar.update(1)
        pbar.close()

    if args.selfplay_games > 0:
        selfplay_engine = chess.engine.SimpleEngine.popen_uci(args.label_engine)
        if args.selfplay_engine:
            selfplay_engine.quit()
            selfplay_engine = chess.engine.SimpleEngine.popen_uci(args.selfplay_engine)
        if args.selfplay_engine_opt:
            opts = {}
            for opt in args.selfplay_engine_opt:
                if "=" in opt:
                    name, value = opt.split("=", 1)
                    opts[name.strip()] = value.strip()
            selfplay_engine.configure(opts)

        selfplay_cap = args.max_positions - len(fens)
        if selfplay_cap > 0:
            boards = selfplay_positions(
                selfplay_engine,
                games=args.selfplay_games,
                movetime_ms=args.selfplay_movetime_ms,
                min_ply=12,
                max_ply=180,
                positions_per_game=args.positions_per_game,
            )
            random.shuffle(boards)
            fens.extend(b.fen() for b in boards[:selfplay_cap])
        selfplay_engine.quit()

    print(f"Collected {len(fens)} positions — labeling with {args.workers} worker(s) at depth {args.eval_depth}...")

    # ── Label in parallel ──────────────────────────────────────────────────
    written = 0
    chunksize = max(1, len(fens) // (args.workers * 8))

    with output.open("w", encoding="utf-8") as out_f:
        with Pool(
            processes=args.workers,
            initializer=_worker_init,
            initargs=(args.label_engine, args.eval_depth),
        ) as pool:
            for result in tqdm(
                pool.imap(_label_fen, fens, chunksize=chunksize),
                total=len(fens),
                desc="labeling",
            ):
                if result is not None:
                    out_f.write(json.dumps(result) + "\n")
                    written += 1

    print(f"Wrote {written} samples to {output}")


if __name__ == "__main__":
    main()
