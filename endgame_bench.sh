#!/bin/bash
# Endgame benchmark: tests if the engine can convert basic endgames.
# Plays both engines from endgame FEN positions and reports results.
#
# Usage: ./endgame_bench.sh [engine] [opponent] [movetime]

ENGINE="${1:-./target/release/chess_uci.exe}"
OPPONENT="${2:-./target/release/chess_uci_legacy.exe}"
MOVETIME="${3:-200}"
SELF_PLAY="./target/release/self_play.exe"
GAMES=6

echo "Endgame Benchmark Suite"
echo "  Engine   : $ENGINE"
echo "  Opponent : $OPPONENT"
echo "  Movetime : ${MOVETIME}ms"
echo "  Games    : $GAMES per endgame"
echo ""

run_endgame() {
    local name="$1"
    local fen="$2"
    local output
    output=$("$SELF_PLAY" "$ENGINE" "$OPPONENT" --games $GAMES --movetime "$MOVETIME" --fen "$fen" 2>&1)
    local score_line
    score_line=$(echo "$output" | grep "score:" | tail -1)
    local score
    score=$(echo "$score_line" | grep -oE '[0-9.]+%' | head -1)
    printf "  %-40s %s\n" "$name" "$score"
}

run_endgame "KQ vs K (white wins)"        "4k3/8/8/8/8/8/8/4K2Q w - - 0 1"
run_endgame "KR vs K (white wins)"        "4k3/8/8/8/8/8/8/R3K3 w - - 0 1"
run_endgame "KQ vs K (black wins)"        "4k2q/8/8/8/8/8/8/4K3 b - - 0 1"
run_endgame "KR vs K (black wins)"        "r3k3/8/8/8/8/8/8/4K3 b - - 0 1"
run_endgame "K+2R vs K (ladder mate)"     "4k3/8/8/8/8/8/8/R3K2R w - - 0 1"
run_endgame "KBB vs K (bishop pair mate)" "4k3/8/8/8/8/8/8/2B1KB2 w - - 0 1"
run_endgame "KP vs K (pawn on 6th)"       "4k3/8/4P3/8/8/8/8/4K3 w - - 0 1"
run_endgame "KP vs K (pawn on 2nd)"       "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1"
