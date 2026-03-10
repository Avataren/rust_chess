#!/usr/bin/env python3
"""Serve web/wasm/ with Cross-Origin Isolation headers (required for SharedArrayBuffer)."""
import http.server, functools, os

os.chdir(os.path.join(os.path.dirname(__file__) or ".", "web", "wasm"))

class COOPHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        super().end_headers()

print("Serving on http://localhost:8080 (with COOP/COEP headers)")
http.server.HTTPServer(("", 8080), COOPHandler).serve_forever()
