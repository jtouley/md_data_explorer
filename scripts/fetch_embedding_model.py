#!/usr/bin/env python3
"""Build step: fetch the embedding model once, at a pinned revision, with SHA-256 checks.

    make fetch-embedding-model                       # into the package (bundled by `uv build`)
    uv run python scripts/fetch_embedding_model.py --dest build/models/all-MiniLM-L6-v2

Downloads the files listed in ``clinical_analytics.core.embeddings.MODEL_FILES`` from
huggingface.co at ``MODEL_REVISION`` (a commit, so the bytes cannot change), writes each to a
``.part`` file, checks its SHA-256, then renames it into place. Files already present with the
right hash are kept, so the step is safe to run twice. The running app never downloads; it
fails with a clear message when the model is missing.

Model: sentence-transformers/all-MiniLM-L6-v2, Apache-2.0 (see THIRD_PARTY_NOTICES.md).
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from clinical_analytics.core.embeddings import (  # noqa: E402
    MODEL_FILES,
    MODEL_REPO,
    MODEL_REVISION,
    bundled_model_dir,
)

DEFAULT_BASE_URL = "https://huggingface.co"
CHUNK = 1 << 20


class ModelIntegrityError(RuntimeError):
    """A downloaded file does not match its pinned SHA-256."""


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(CHUNK):
            digest.update(block)
    return digest.hexdigest()


def fetch_file(base_url: str, name: str, expected: str, dest: Path) -> str:
    """Download one pinned file into ``dest``; returns 'kept' or 'fetched'."""
    target = dest / name
    if target.is_file() and sha256_of(target) == expected:
        return "kept"
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    url = f"{base_url}/{MODEL_REPO}/resolve/{MODEL_REVISION}/{name}"
    try:
        with urllib.request.urlopen(url, timeout=120) as resp, part.open("wb") as out:  # noqa: S310 - https, pinned
            while block := resp.read(CHUNK):
                out.write(block)
        actual = sha256_of(part)
        if actual != expected:
            raise ModelIntegrityError(f"{name}: SHA-256 {actual} does not match pinned {expected}")
        part.replace(target)
    finally:
        part.unlink(missing_ok=True)
    return "fetched"


def verify(dest: Path) -> list[str]:
    """Names of pinned files that are missing or do not match their hash."""
    return [n for n, h in MODEL_FILES.items() if not (dest / n).is_file() or sha256_of(dest / n) != h]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dest", type=Path, default=None, help="target directory (default: inside the package)")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Hugging Face endpoint or a mirror")
    parser.add_argument("--verify-only", action="store_true", help="check hashes, download nothing")
    args = parser.parse_args(argv)
    dest = (args.dest or bundled_model_dir()).resolve()

    if args.verify_only:
        bad = verify(dest)
        for name in bad:
            print(f"missing or modified: {dest / name}", file=sys.stderr)
        return 1 if bad else 0

    for name, expected in MODEL_FILES.items():
        try:
            status = fetch_file(args.base_url, name, expected, dest)
        except (urllib.error.URLError, OSError, ModelIntegrityError) as e:
            print(f"Could not fetch {name} for {MODEL_REPO}@{MODEL_REVISION[:12]}: {e}", file=sys.stderr)
            return 1
        print(f"{status:7} {name}")
    print(f"Embedding model ready at {dest} ({MODEL_REPO}@{MODEL_REVISION[:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
