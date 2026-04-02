// binpack2jsonl.cpp — Convert Stockfish NNUE .binpack files to JSONL
//
// Outputs one JSON record per position:
//   {"fen": "<FEN>", "cp": <score>}
//
// By default, score is in Stockfish internal evaluation units (side-to-move).
// Use --cp-divisor 2.96 --white-absolute to produce centipawns in white-absolute
// convention, which is what this project's training pipeline expects in JSONL.
//
// Flags:
//   --cp-divisor FLOAT   Divide score by this value (default 1.0).
//                        Use 2.96 for test80/T60T70 binpacks (SF14/15 era).
//   --white-absolute     Negate score when black is to move, converting from
//                        side-to-move to white-absolute perspective.
//   --max N              Stop after N positions.
//   --filter-cp N        Drop positions where |cp| > N (applied after divisor).
//
// Build (from the nnue-pytorch repo root):
//   g++ -O2 -std=c++17 -Idata_loader/cpp/lib -Idata_loader/cpp \
//       /tmp/binpack2jsonl.cpp -o /tmp/binpack2jsonl -lpthread
//
// Usage — produce white-absolute centipawns, clamp at 3000, take 50M positions:
//   /tmp/binpack2jsonl T60T70.binpack \
//       --cp-divisor 2.96 --white-absolute --filter-cp 3000 --max 50000000 \
//       > data/t60_t70.jsonl

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
                  << " input.binpack [--max N] [--filter-cp N]"
                     " [--cp-divisor F] [--white-absolute]\n";
        return 1;
    }

    std::string filename;
    long long max_positions = -1;   // -1 = unlimited
    int filter_cp = 0;              //  0 = disabled
    float cp_divisor = 1.0f;        //  divide score by this before output
    bool white_absolute = false;    //  if true, negate score when black to move

    for (int i = 1; i < argc; ++i) {
        std::string_view arg(argv[i]);
        if (arg == "--max"  && i + 1 < argc) { max_positions = std::atoll(argv[++i]); }
        else if (arg == "--filter-cp" && i + 1 < argc) { filter_cp = std::atoi(argv[++i]); }
        else if (arg == "--cp-divisor" && i + 1 < argc) { cp_divisor = std::atof(argv[++i]); }
        else if (arg == "--white-absolute") { white_absolute = true; }
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

        std::string fen = e.pos.fen();

        // Determine side to move from FEN (second field: 'w' or 'b').
        bool black_to_move = (fen.find(" b ") != std::string::npos);

        // Convert from SF internal units to centipawns, then apply perspective.
        int cp_stm = e.score;
        int cp_out;
        if (cp_divisor != 1.0f)
            cp_out = static_cast<int>(std::round(cp_stm / cp_divisor));
        else
            cp_out = cp_stm;

        // Convert side-to-move → white-absolute if requested.
        if (white_absolute && black_to_move)
            cp_out = -cp_out;

        if (filter_cp > 0 && std::abs(cp_out) > filter_cp) {
            ++dropped;
            continue;
        }

        // Emit compact JSON.
        buf += "{\"fen\":\"";
        buf += fen;
        buf += "\",\"cp\":";
        buf += std::to_string(cp_out);
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
 *   g++ -O2 -std=c++20 \
 *       -Idata_loader/cpp/lib -Idata_loader/cpp \
 *       /path/to/binpack2jsonl.cpp -o binpack2jsonl -lpthread
 *
 * NOTE ON UNITS:
 *   Scores in .binpack files are Stockfish internal evaluation units, not
 *   centipawns. Empirically, 1 cp ≈ 2.96 internal units for SF14/15-era
 *   datasets (test80, T60T70wIsRightFarseer, data_d9_2021_09_02).
 *   Use --cp-divisor 2.96 --white-absolute to get centipawns in white-absolute
 *   convention directly usable by preprocess_dataset.py.
 *
 *   WDL calibration in features.py absorbs the ~5% rounding error in the
 *   divisor, so Stockfish re-labeling is NOT required.
 */
