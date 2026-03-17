# XavChess

A Rust chess engine with a Bevy GUI, UCI interface, and a dual-perspective NNUE evaluation network.

## Workspace layout

| Crate | Description |
|---|---|
| `chess` | Bevy desktop/WASM GUI |
| `chess_uci` | UCI engine binary |
| `chess_evaluation` | Search, evaluation, bench |
| `chess_board` | Board representation |
| `move_generator` | Legal move generation |
| `chess_foundation` | Shared types |
| `self_play` | Engine vs engine match runner |
| `nn_training` | Python NNUE training pipeline |

---

## Evaluation backends

Three compile-time backends are available as Cargo features on `chess_evaluation`.
Pick **exactly one** per binary.

| Feature | Eval path | Runtime switching | Notes |
|---|---|---|---|
| `classical-eval` | Hand-crafted HCE only | — | No NN code compiled in |
| `nn-full-forward` | Full NN forward pass per call | — | Weights embedded at startup |
| `nn-incremental` | Incremental i16 accumulators in search, full forward elsewhere | — | Fastest; used by the GUI |
| `runtime-switch` | HCE + NN toggled via `set_neural_eval_enabled()` | ✓ | UCI `EvalFile` / `NeuralEval` setoptions work |

The **chess GUI** (`chess` crate) has its own independent feature setting in `chess/Cargo.toml`
and embeds `chess_evaluation/src/eval.npz` at compile time. Change it there to switch the GUI's eval mode.

All other binaries — **chess_uci**, **self_play**, **bench** — read their feature from
a **single place**: the `[workspace.dependencies]` entry in the root `Cargo.toml`.

### Switching the eval mode for chess_uci / self_play / bench

Edit one line in `Cargo.toml`:

```toml
[workspace.dependencies]
chess_evaluation = { path = "chess_evaluation", default-features = false, features = ["nn-incremental"] }
#                                                                                       ^^^^^^^^^^^^^^
#                          classical-eval | nn-full-forward | nn-incremental | runtime-switch
```

Then rebuild the relevant binary — no other files need to change.

### Updating the neural weights

```bash
# Copy a newly trained .npz over the deployed weights file
cp nn_training/artifacts/eval_halfkp_10m.npz chess_evaluation/src/eval.npz

# Rebuild the binaries that embed weights (chess, chess_uci with NN features, bench, self_play)
cargo build --release -p chess -p chess_uci
```

---

## Building

```bash
# Native debug
cargo build -p chess

# Native release
cargo build -p chess --release

# WASM release
./build_web.sh

# UCI engine (uses workspace eval feature)
cargo build -p chess_uci --release

# UCI engine with a specific eval mode (overrides workspace setting)
cargo build -p chess_uci --release --no-default-features --features nn-incremental
```

---

## Benchmarking

```bash
# Default (workspace eval feature, depth 8)
cargo run -p chess_evaluation --bin bench --release

# Specific eval mode
cargo run -p chess_evaluation --bin bench --release --no-default-features --features classical-eval
cargo run -p chess_evaluation --bin bench --release --no-default-features --features nn-full-forward
cargo run -p chess_evaluation --bin bench --release --no-default-features --features nn-incremental

# Depth / thread sweep
cargo run -p chess_evaluation --bin bench --release -- --depth 10 --threads 1,2,4,8

# Full hash + thread grid
cargo run -p chess_evaluation --bin bench --release -- --hash-sweep
```

---

## Self-play between two eval modes

`self_play` takes two external UCI engine executables as arguments.
To pit two eval modes against each other, build `chess_uci` twice under different names:

```bash
# Step 1 — build both engines
cargo build -p chess_uci --release --no-default-features --features classical-eval
cp target/release/chess_uci target/release/uci_classical

cargo build -p chess_uci --release --no-default-features --features nn-full-forward
cp target/release/chess_uci target/release/uci_nn_full

cargo build -p chess_uci --release --no-default-features --features nn-incremental
cp target/release/chess_uci target/release/uci_nn_incr

# Step 2 — run a match
cargo run -p self_play --release -- \
    ./target/release/uci_classical \
    ./target/release/uci_nn_full \
    --games 100 --movetime 100

# Classical vs incremental
cargo run -p self_play --release -- \
    ./target/release/uci_classical \
    ./target/release/uci_nn_incr \
    --games 100 --movetime 100 --no-ponder
```

With `runtime-switch`, you can also configure one engine to use HCE and the other to use NN
without recompiling — pass setoptions via `--engine2-opt`:

```bash
cargo run -p self_play --release -- \
    ./target/release/chess_uci \
    ./target/release/chess_uci \
    --engine2-opt "EvalFile=chess_evaluation/src/eval.npz" \
    --engine2-opt "NeuralEval=true" \
    --games 100 --movetime 100
```

---

## Running tests

```bash
# All evaluation unit tests (uses classical-eval default)
cargo test -p chess_evaluation

# Full workspace
cargo test
```

---

## Neural network training

See [`nn_training/README.md`](nn_training/README.md) for the full training pipeline.
