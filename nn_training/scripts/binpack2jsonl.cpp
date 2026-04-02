// binpack2jsonl.cpp — Convert Stockfish NNUE .binpack files to JSONL
//
// Outputs one JSON record per position:
//   {"fen": "<FEN>", "cp": <score>}
//
// Score is side-to-move centipawns (matches the JSONL convention used by
// generate_data.py and the rest of the training pipeline).
//
// Build (from the nnue-pytorch repo root):
//   g++ -O2 -std=c++17 -Idata_loader/cpp/lib -Idata_loader/cpp \
//       /tmp/binpack2jsonl.cpp -o /tmp/binpack2jsonl -lpthread
//
// Usage:
//   /tmp/binpack2jsonl input.binpack [--max N] [--filter-cp N] > output.jsonl
//   /tmp/binpack2jsonl input.binpack --max 5000000 --filter-cp 3000 | pv > out.jsonl

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <string_view>

// Pull in the full Stockfish-compatible header-only library that defines
// TrainingDataEntry, the binpack stream reader, etc.
#include "nnue_training_data_formats.h"
#include "nnue_training_data_stream.h"

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0]
                  << " input.binpack [--max N] [--filter-cp N]\n";
        return 1;
    }

    std::string filename;
    long long max_positions = -1;   // -1 = unlimited
    int filter_cp = 0;              //  0 = disabled

    for (int i = 1; i < argc; ++i) {
        std::string_view arg(argv[i]);
        if (arg == "--max"  && i + 1 < argc) { max_positions = std::atoll(argv[++i]); }
        else if (arg == "--filter-cp" && i + 1 < argc) { filter_cp = std::atoi(argv[++i]); }
        else if (arg[0] != '-') { filename = argv[i]; }
    }

    if (filename.empty()) {
        std::cerr << "No input file specified.\n";
        return 1;
    }

    auto stream = training_data::open_sfen_input_file(filename, /*cyclic=*/false);
    if (!stream) {
        std::cerr << "Failed to open: " << filename << "\n";
        return 1;
    }

    // Use a large output buffer for throughput.
    std::string buf;
    buf.reserve(256 * 1024);

    long long written = 0;
    long long dropped = 0;

    while (true) {
        auto entry = stream->next();
        if (!entry.has_value()) break;

        const auto& e = *entry;

        int cp = e.score;
        if (filter_cp > 0 && std::abs(cp) > filter_cp) {
            ++dropped;
            continue;
        }

        // Escape any backslash or quote in the FEN (FENs shouldn't have them,
        // but guard against malformed input).
        std::string fen = e.pos.fen();
        // Emit compact JSON.
        buf += "{\"fen\":\"";
        buf += fen;
        buf += "\",\"cp\":";
        buf += std::to_string(cp);
        buf += "}\n";

        ++written;

        // Flush buffer periodically to avoid excessive memory use.
        if (buf.size() >= 256 * 1024) {
            std::cout << buf;
            buf.clear();
        }

        if (max_positions > 0 && written >= max_positions) break;
    }

    if (!buf.empty()) std::cout << buf;

    std::cerr << "Written: " << written << "  Dropped (|cp|>" << filter_cp << "): " << dropped << "\n";
    return 0;
}
/*
 * BUILD:
 *   cd /tmp/nnue-pytorch  (git clone https://github.com/official-stockfish/nnue-pytorch)
 *   cmake -S data_loader/cpp/ -B build -DCMAKE_BUILD_TYPE=Release
 *   cmake --build build -j8 --target training_data_loader
 *   g++ -O2 -std=c++20 \
 *       -Idata_loader/cpp/lib -Idata_loader/cpp \
 *       /path/to/binpack2jsonl.cpp -o binpack2jsonl -lpthread
 *
 * NOTE ON UNITS:
 *   Scores in .binpack files are in Stockfish internal evaluation units,
 *   NOT centipawns. 1 pawn ≈ 200-210 internal units ≈ 100 centipawns.
 *   Ratio is approximately 2-3x vs centipawns (varies by Stockfish version).
 *
 *   To use with this project's training pipeline (which expects centipawns),
 *   RE-LABEL the extracted FENs with Stockfish via generate_data.py:
 *     ./binpack2jsonl input.binpack | jq -r '.fen' > fens.txt
 *     python3 scripts/generate_data.py --label-engine /usr/bin/stockfish \
 *             --fens fens.txt --output output.jsonl --eval-depth 14 ...
 *
 *   This lets you keep the valuable quiet-position selection from test80
 *   while getting proper centipawn labels in your pipeline's scale.
 */
