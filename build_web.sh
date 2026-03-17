#!/usr/bin/env bash
set -euo pipefail

CRATE="chess"
PROFILE="wasm-release"
TARGET="wasm32-unknown-unknown"
OUT_DIR="web/wasm"

BINARY="target/${TARGET}/${PROFILE}/${CRATE}.wasm"

# ── Check / install wasm-bindgen-cli ─────────────────────────────────────────

REQUIRED_WBG=$(cargo tree -p ${CRATE} --target ${TARGET} 2>/dev/null \
    | grep "wasm-bindgen v" \
    | head -1 \
    | sed 's/.*wasm-bindgen v\([0-9.]*\).*/\1/' || true)

if [[ -z "$REQUIRED_WBG" ]]; then
    echo "Could not determine required wasm-bindgen version — continuing anyway"
else
    INSTALLED_WBG=$(wasm-bindgen --version 2>/dev/null | awk '{print $2}' || echo "none")
    if [[ "$INSTALLED_WBG" != "$REQUIRED_WBG" ]]; then
        echo "Installing wasm-bindgen-cli ${REQUIRED_WBG} (have ${INSTALLED_WBG})..."
        cargo install wasm-bindgen-cli --version "${REQUIRED_WBG}" --locked
    else
        echo "wasm-bindgen-cli ${INSTALLED_WBG} — OK"
    fi
fi

# ── Build (nightly + rebuild std with atomics for SharedArrayBuffer) ─────────

echo "Building ${CRATE} for ${TARGET} (profile: ${PROFILE}, threaded)..."
cargo +nightly build -p ${CRATE} --target ${TARGET} --profile ${PROFILE} \
    -Z build-std=panic_abort,std

# ── Generate JS bindings ──────────────────────────────────────────────────────

echo "Running wasm-bindgen..."
wasm-bindgen --out-dir "${OUT_DIR}" --target web "${BINARY}"

# ── Optional: optimise with wasm-opt ─────────────────────────────────────────

WASM_OUT="${OUT_DIR}/${CRATE}_bg.wasm"
if command -v wasm-opt &>/dev/null; then
    echo "Optimising with wasm-opt..."
    wasm-opt -Oz \
        --enable-nontrapping-float-to-int \
        --enable-bulk-memory \
        --enable-sign-ext \
        --enable-threads \
        --enable-mutable-globals \
        --enable-simd \
        "${WASM_OUT}" -o "${WASM_OUT}"
else
    echo "wasm-opt not found — skipping (install binaryen to reduce binary size)"
fi

# ── Summary ──────────────────────────────────────────────────────────────────

SIZE=$(du -sh "${WASM_OUT}" | cut -f1)
echo ""
echo "Done! Output in ${OUT_DIR}/"
echo "  ${WASM_OUT}  (${SIZE})"
echo ""
echo "IMPORTANT: Your server must send these headers for SharedArrayBuffer:"
echo "  Cross-Origin-Opener-Policy: same-origin"
echo "  Cross-Origin-Embedder-Policy: require-corp"
echo ""
echo "Serve with:"
echo "  python3 serve_coop.py"
echo "  then open http://localhost:8080"
