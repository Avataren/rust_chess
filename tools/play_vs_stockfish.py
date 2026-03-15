#!/usr/bin/env python3
"""
Play XavChess vs Stockfish at a fixed ELO rating.
Usage:
    python3 tools/play_vs_stockfish.py [--games N] [--elo ELO] [--threads T] [--movetime MS]
"""
import subprocess
import sys
import time
import argparse
import math
import io
import threading
from typing import Optional

STOCKFISH = "/usr/bin/stockfish"
ENGINE    = "/home/avataren/src/rust_chess/target/release/chess_uci"

OPENINGS = [
    "startpos",
    "startpos moves e2e4",
    "startpos moves d2d4",
    "startpos moves e2e4 e7e5",
    "startpos moves e2e4 c7c5",
    "startpos moves d2d4 d7d5",
    "startpos moves e2e4 e7e6",
    "startpos moves e2e4 e7e5 g1f3",
    "startpos moves d2d4 g8f6",
    "startpos moves e2e4 e7e5 g1f3 b8c6",
    "startpos moves e2e4 e7e5 g1f3 b8c6 f1b5",
    "startpos moves d2d4 d7d5 c2c4",
    "startpos moves e2e4 c7c5 g1f3",
    "startpos moves d2d4 g8f6 c2c4",
    "startpos moves e2e4 e7e6 d2d4",
    "startpos moves d2d4 d7d5 c2c4 e7e6",
    "startpos moves e2e4 d7d5",
    "startpos moves e2e4 e7e5 g1f3 b8c6 f1b5 a7a6",
    "startpos moves d2d4 g8f6 c2c4 e7e6 g1f3",
    "startpos moves e2e4 c7c5 g1f3 d7d6",
]


class UCIEngine:
    def __init__(self, path: str, name: str, options: dict = None):
        self.name = name
        self.proc = subprocess.Popen(
            [path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self._send("uci")
        self._wait_for("uciok")
        if options:
            for k, v in options.items():
                self._send(f"setoption name {k} value {v}")
        self._send("isready")
        self._wait_for("readyok")

    def _send(self, cmd: str):
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()

    def _wait_for(self, token: str, timeout: float = 30.0) -> list[str]:
        lines = []
        self.proc.stdout.reconfigure(line_buffering=True)  # type: ignore
        deadline = time.time() + timeout
        while time.time() < deadline:
            line = self.proc.stdout.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            lines.append(line)
            if token in line:
                return lines
        raise TimeoutError(f"Did not receive '{token}' within {timeout}s. Got: {lines}")

    def new_game(self):
        self._send("ucinewgame")
        self._send("isready")
        self._wait_for("readyok")

    def bestmove(self, position: str, movetime_ms: int) -> Optional[str]:
        self._send(f"position {position}")
        self._send(f"go movetime {movetime_ms}")
        try:
            lines = self._wait_for("bestmove", timeout=(movetime_ms / 1000) + 5)
        except TimeoutError:
            return None
        for line in lines:
            if line.startswith("bestmove"):
                parts = line.split()
                mv = parts[1] if len(parts) > 1 else None
                return None if mv in (None, "0000", "(none)") else mv
        return None

    def close(self):
        try:
            self._send("quit")
            self.proc.wait(timeout=3)
        except Exception:
            self.proc.kill()
            self.proc.wait()


def play_game(engine1: UCIEngine, engine2: UCIEngine,
              start_pos: str, movetime_ms: int,
              max_halfmoves: int = 400) -> tuple[Optional[str], str, int]:
    """
    Play one game. engine1=White, engine2=Black.
    Returns (winner, reason, halfmoves).
    winner: "engine1", "engine2", or None (draw)
    """
    position_moves = start_pos
    if "moves" in start_pos:
        pre_moves = start_pos.split("moves")[1].strip().split()
    else:
        pre_moves = []

    white_turn = len(pre_moves) % 2 == 0
    no_progress = 0   # for 50-move rule simulation

    for halfmove in range(max_halfmoves):
        current = engine1 if white_turn else engine2

        mv = current.bestmove(position_moves, movetime_ms)

        if mv is None:
            # Engine returned no move — must be checkmate or stalemate
            if white_turn:
                # White has no move: white is checkmated or stalemated
                # We can't distinguish easily, so call it a win for black
                return ("engine2", "no_move_white", halfmove)
            else:
                return ("engine1", "no_move_black", halfmove)

        # Track no-progress (approximate 50-move rule)
        if 'x' in mv or (len(mv) >= 5):  # capture or promotion (crude heuristic)
            no_progress = 0
        else:
            no_progress += 1

        if "moves" in position_moves:
            position_moves += f" {mv}"
        else:
            position_moves += f" moves {mv}"

        white_turn = not white_turn

        if no_progress >= 100:  # 50 full moves without progress
            return (None, "50_move_rule", halfmove)

    return (None, "max_moves", max_halfmoves)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games",    type=int, default=10,   help="Number of game pairs")
    parser.add_argument("--elo",      type=int, default=2000,  help="Stockfish ELO limit")
    parser.add_argument("--movetime", type=int, default=200,   help="Move time in ms")
    parser.add_argument("--threads",  type=int, default=1,     help="XavChess thread count")
    parser.add_argument("--hash",     type=int, default=96,    help="XavChess TT size in MB")
    parser.add_argument("--verbose",  action="store_true",     help="Show engine moves")
    args = parser.parse_args()

    print(f"=== XavChess vs Stockfish ELO {args.elo} ===")
    print(f"  Game pairs: {args.games}, Movetime: {args.movetime}ms, XavChess threads: {args.threads}, Hash: {args.hash}MB")
    print()

    sf_options = {
        "UCI_LimitStrength": "true",
        "UCI_Elo": str(args.elo),
        "Threads": "1",
        "Hash": "128",
    }
    xc_options = {
        "Threads": str(args.threads),
        "Hash": str(args.hash),
    }

    wins = draws = losses = 0
    openings = (OPENINGS * ((args.games // len(OPENINGS)) + 1))[:args.games]

    for game_pair in range(args.games):
        opening = openings[game_pair % len(OPENINGS)]

        for flip in range(2):
            sf = UCIEngine(STOCKFISH, "Stockfish", sf_options)
            xc = UCIEngine(ENGINE, "XavChess", xc_options)
            sf.new_game()
            xc.new_game()

            if flip == 0:
                winner, reason, moves = play_game(xc, sf, opening, args.movetime)
                if winner == "engine1":   wins += 1;   side = "W"
                elif winner == "engine2": losses += 1; side = "L"
                else:                     draws += 1;  side = "D"
            else:
                winner, reason, moves = play_game(sf, xc, opening, args.movetime)
                if winner == "engine2":   wins += 1;   side = "W"
                elif winner == "engine1": losses += 1; side = "L"
                else:                     draws += 1;  side = "D"

            sf.close()
            xc.close()

            game_num = game_pair * 2 + flip + 1
            total = wins + draws + losses
            score_pct = (wins + draws * 0.5) / total * 100 if total > 0 else 0
            xc_side = "White" if flip == 0 else "Black"
            print(f"  Game {game_num:3d} [{xc_side:5s}] {side}  |  "
                  f"W:{wins} D:{draws} L:{losses}  "
                  f"Score: {score_pct:.1f}%  ({reason}, {moves} moves)")
            sys.stdout.flush()

    total = wins + draws + losses
    score = (wins + draws * 0.5) / total if total > 0 else 0
    elo_diff = -400 * math.log10(1 / score - 1) if 0 < score < 1 else (float('inf') if score >= 1 else float('-inf'))

    print()
    print(f"=== FINAL: W:{wins} D:{draws} L:{losses} / {total} games ===")
    print(f"    Score: {score*100:.1f}%")
    if abs(elo_diff) < 1000:
        print(f"    Est. ELO vs SF{args.elo}: {elo_diff:+.0f} → ~{args.elo + elo_diff:.0f}")
    else:
        print(f"    Est. ELO vs SF{args.elo}: dominant (>{args.elo + 400:.0f} or <{args.elo - 400:.0f})")


if __name__ == "__main__":
    main()
