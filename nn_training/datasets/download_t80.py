#!/usr/bin/env python3
"""Download test80-2024 binpack files from HuggingFace."""
import sys
from huggingface_hub import hf_hub_download
from pathlib import Path

DATASETS = [
    ("linrock/test80-2024", "test80-2024-04-apr-2tb7p.min-v2.v6.binpack.zst"),  # 12.3 GB
    ("linrock/test80-2024", "test80-2024-03-mar-2tb7p.min-v2.v6.binpack.zst"),  # 8.8 GB
    ("linrock/test80-2024", "test80-2024-05-may-2tb7p.min-v2.v6.binpack.zst"),  # 8.9 GB
]

dest = Path(__file__).parent
for repo, filename in DATASETS:
    out = dest / filename
    if out.exists():
        print(f"Already exists: {filename}")
        continue
    print(f"Downloading {filename} ({repo})...", flush=True)
    hf_hub_download(repo_id=repo, filename=filename, repo_type="dataset",
                    local_dir=str(dest))
    print(f"Done: {filename}", flush=True)
print("All downloads complete.")
