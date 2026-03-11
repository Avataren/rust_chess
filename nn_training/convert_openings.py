"""Convert lichess chess-openings TSV (PGN) to Rust opening book constants."""
import chess
import csv
import sys
from pathlib import Path

TSV_DIR = Path(__file__).parent / "chess-openings"
TSV_FILES = ["a.tsv", "b.tsv", "c.tsv", "d.tsv", "e.tsv"]

def pgn_to_uci(pgn: str) -> list[str]:
    """Convert a PGN move sequence to a list of UCI move strings."""
    board = chess.Board()
    uci_moves = []
    # Strip move numbers and parse
    tokens = pgn.split()
    for token in tokens:
        # Skip move numbers like "1." "2." etc
        if token[0].isdigit() and '.' in token:
            continue
        try:
            move = board.parse_san(token)
            uci_moves.append(move.uci())
            board.push(move)
        except (chess.InvalidMoveError, chess.IllegalMoveError) as e:
            print(f"  Warning: invalid move '{token}' in '{pgn}': {e}", file=sys.stderr)
            break
    return uci_moves

def main():
    entries = []  # (eco, name, uci_moves)

    for tsv_file in TSV_FILES:
        path = TSV_DIR / tsv_file
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                eco = row["eco"]
                name = row["name"]
                pgn = row["pgn"]
                uci_moves = pgn_to_uci(pgn)
                if uci_moves:
                    entries.append((eco, name, uci_moves))

    print(f"Converted {len(entries)} openings", file=sys.stderr)

    # Generate OPENING_LINES: deduplicated lines for book moves
    # Generate NAMED_OPENINGS: (name, line) for display

    # For OPENING_LINES, we want every unique line (not just the final position).
    # The book stores every intermediate position → next move.
    # We only need the full lines (shorter prefixes are automatically covered
    # by the replay logic in OpeningBook::build).

    # Filter: skip very short lines (1 move) that are just first moves,
    # and skip very long lines (>20 moves) which are rarely reached.
    lines = []
    seen = set()
    for eco, name, uci_moves in entries:
        if len(uci_moves) < 2:
            continue
        # Truncate very long lines
        uci_moves = uci_moves[:20]
        key = " ".join(uci_moves)
        if key not in seen:
            seen.add(key)
            lines.append(uci_moves)

    # Sort by length (shorter first) for nicer output
    lines.sort(key=lambda x: (len(x), x))

    # Write Rust source
    out = []
    out.append("// Auto-generated from lichess-org/chess-openings — do not edit manually.")
    out.append(f"// {len(lines)} opening lines, {len(entries)} named openings.")
    out.append("")

    # OPENING_LINES
    out.append("/// Opening lines expressed as UCI move strings.")
    out.append("/// Every position encountered while replaying a line gets the next move")
    out.append("/// recorded as a book response.")
    out.append("pub(crate) const OPENING_LINES: &[&[&str]] = &[")
    for uci_moves in lines:
        moves_str = ", ".join(f'"{m}"' for m in uci_moves)
        out.append(f"    &[{moves_str}],")
    out.append("];")
    out.append("")

    # NAMED_OPENINGS
    out.append("/// Named openings: (display_name, characteristic_line).")
    out.append("/// The hash of the position AFTER playing the full line is stored so that")
    out.append("/// `probe_name` can identify the opening by board state.")
    out.append("pub(crate) const NAMED_OPENINGS: &[(&str, &[&str])] = &[")
    for eco, name, uci_moves in entries:
        uci_moves = uci_moves[:20]
        moves_str = ", ".join(f'"{m}"' for m in uci_moves)
        # Escape quotes in name
        safe_name = name.replace('"', '\\"')
        out.append(f'    ("{safe_name}", &[{moves_str}]),')
    out.append("];")
    out.append("")

    # Write to file
    out_path = Path(__file__).parent.parent / "chess_evaluation" / "src" / "opening_book_data.rs"
    out_path.write_text("\n".join(out), encoding="utf-8")
    print(f"Wrote {out_path} ({len(lines)} lines, {len(entries)} named)", file=sys.stderr)

if __name__ == "__main__":
    main()
