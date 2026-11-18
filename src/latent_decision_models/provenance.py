"""File provenance helpers (checksums) used by audits and dataset fetchers."""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    """Return the hex SHA-256 digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_blob_sha1(content: bytes) -> str:
    """Return the git object id of ``content`` as a blob.

    Lets a downloaded file be verified against the SHA listed in a git tree
    (e.g. the GitHub mirror of an OpenNeuro snapshot tag).
    """
    header = f"blob {len(content)}\0".encode()
    return hashlib.sha1(header + content).hexdigest()
