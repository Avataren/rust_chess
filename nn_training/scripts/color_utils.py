"""
Terminal color helpers for metric output.

All public functions return a plain string when stdout is not a TTY so that
log files and piped output are never polluted with escape codes.

Color semantics used throughout the training loop:
  green  — good / improving / passed
  yellow — neutral / borderline / watch
  red    — bad / degrading / failed
  blue   — informational (no quality judgement)
  white  — neutral label text
"""
from __future__ import annotations

import sys

_ANSI: dict[str, str] = {
    "green":  "\033[92m",
    "yellow": "\033[93m",
    "red":    "\033[91m",
    "blue":   "\033[94m",
    "white":  "\033[97m",
    "bold":   "\033[1m",
    "reset":  "\033[0m",
}


def _tty() -> bool:
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def clr(text: str, color: str) -> str:
    """Wrap *text* in ANSI color if stdout is a TTY, else return plain text."""
    if not _tty():
        return text
    return f"{_ANSI.get(color, '')}{text}{_ANSI['reset']}"


# ── Per-metric colored formatters ─────────────────────────────────────────────

def clr_puzzle(score: float) -> str:
    """Puzzle solve rate (higher is better). Green ≥85%, yellow ≥75%, red <75%."""
    s = f"{score:.1f}%"
    if score >= 85:
        return clr(s, "green")
    if score >= 75:
        return clr(s, "yellow")
    return clr(s, "red")


def clr_val_mae(value: float, best: float) -> str:
    """Val CP-MAE relative to the current best (lower is better).

    Green  — improved by more than 0.2 cp
    Yellow — roughly the same (within ±1 cp)
    Red    — degraded by more than 1 cp
    Blue   — no reference (best < 0)
    """
    s = f"{value:.1f}cp"
    if best < 0:
        return clr(s, "blue")
    delta = value - best
    if delta < -0.2:
        return clr(s, "green")
    if delta <= 1.0:
        return clr(s, "yellow")
    return clr(s, "red")


def clr_gen_mae(value: float, best: float) -> str:
    """Gen-val CP-MAE (lower is better, same scale as val_mae)."""
    return clr_val_mae(value, best)


def clr_gen_mae_trend(delta: float) -> str:
    """Trend arrow for gen-val MAE. Green = dropping, yellow = small rise, red = big rise."""
    if delta <= 0:
        return clr(f"▼{abs(delta):.2f}cp", "green")
    if delta <= 0.5:
        return clr(f"▲{delta:.2f}cp", "yellow")
    return clr(f"▲{delta:.2f}cp", "red")


def clr_winrate(rate: float) -> str:
    """Self-play win rate (higher is better). Green ≥55%, yellow ≥48%, red <48%."""
    s = f"{rate:.1f}%"
    if rate >= 55:
        return clr(s, "green")
    if rate >= 48:
        return clr(s, "yellow")
    return clr(s, "red")


def clr_gate(gate_str: str) -> str:
    """Color a gate status string (PASSED / BORDERLINE / FAILED)."""
    if "PASSED" in gate_str:
        return clr(gate_str, "green")
    if "BORDERLINE" in gate_str:
        return clr(gate_str, "yellow")
    return clr(gate_str, "red")


def clr_promotion(promoted: bool, text: str) -> str:
    return clr(text, "green" if promoted else "red")


# ── Diversity metrics ──────────────────────────────────────────────────────────

def clr_unique_ratio(ratio: float) -> str:
    """Unique FEN ratio. Green ≥0.995, yellow ≥0.98, red <0.98."""
    s = f"{ratio:.3f}"
    if ratio >= 0.995:
        return clr(s, "green")
    if ratio >= 0.98:
        return clr(s, "yellow")
    return clr(s, "red")


def clr_cp_std(std: float) -> str:
    """CP std-dev (diversity of positions). Green ≥180, yellow ≥130, red <130."""
    s = f"{std:.0f}"
    if std >= 180:
        return clr(s, "green")
    if std >= 130:
        return clr(s, "yellow")
    return clr(s, "red")


def clr_decisive(pct: float) -> str:
    """Decisive position % (|cp|>300). Ideal 20–45%. Red if too low or too high."""
    s = f"{pct:.1f}%"
    if 20 <= pct <= 45:
        return clr(s, "green")
    if 10 <= pct <= 55:
        return clr(s, "yellow")
    return clr(s, "red")


def clr_pieces(mean: float, std: float) -> str:
    """Piece count (informational — blue)."""
    return clr(f"{mean:.1f}±{std:.1f}", "blue")
