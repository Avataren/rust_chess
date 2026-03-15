#!/usr/bin/env python3
from __future__ import annotations

import argparse
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


def sample_positions_from_pgn(
    pgn_path: Path,
    max_positions: int,
    plies_min: int,
    min_elo: int = 0,
) -> list[chess.Board]:
    boards: list[chess.Board] = []
    skipped = 0
    with pgn_path.open("r", encoding="utf-8", errors="ignore") as f:
        while len(boards) < max_positions:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            if min_elo > 0:
                try:
                    white_elo = int(game.headers.get("WhiteElo", 0))
                    black_elo = int(game.headers.get("BlackElo", 0))
                except ValueError:
                    skipped += 1
                    continue
                if white_elo < min_elo or black_elo < min_elo:
                    skipped += 1
                    continue
            board = game.board()
            line = []
            for mv in game.mainline_moves():
                board.push(mv)
                line.append(board.copy())
            if len(line) <= plies_min:
                continue
            take = random.choice(line[plies_min:])
            boards.append(take)
    if skipped:
        print(f"Skipped {skipped} games below Elo threshold ({min_elo})")
    return boards


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

    boards: list[chess.Board] = []
    if args.pgn:
        # When self-play is disabled, allow PGN to fill the full quota.
        pgn_cap = args.max_positions if args.selfplay_games == 0 else args.max_positions // 2
        boards.extend(sample_positions_from_pgn(Path(args.pgn), pgn_cap, plies_min=12, min_elo=args.min_elo))

    boards.extend(
        selfplay_positions(
            selfplay_engine,
            games=args.selfplay_games,
            movetime_ms=args.selfplay_movetime_ms,
            min_ply=12,
            max_ply=180,
            positions_per_game=args.positions_per_game,
        )
    )

    random.shuffle(boards)
    boards = boards[: args.max_positions]

    with output.open("w", encoding="utf-8") as f:
        for board in tqdm(boards, desc="label-eval"):
            cp = evaluate_with_stockfish(label_engine, board, depth=args.eval_depth)
            f.write(json.dumps({"fen": board.fen(), "cp": cp}) + "\n")

    if selfplay_engine is not label_engine:
        selfplay_engine.quit()
    label_engine.quit()
    print(f"Wrote {len(boards)} samples to {output}")


if __name__ == "__main__":
    main()
