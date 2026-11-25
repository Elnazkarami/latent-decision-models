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


def json_differences(expected, actual, rtol: float = 1e-6, atol: float = 1e-9,
                     ignore_keys=(), path: str = "", allow_new_keys: bool = False) -> list[str]:
    """Every difference between two JSON-like objects, as human-readable paths.

    Numbers are equal when ``|a - b| <= atol + rtol * |b|`` (NaN equals NaN); strings,
    booleans, ``None``, list lengths and dict keys must match exactly. Keys named in
    ``ignore_keys`` are skipped at any depth (e.g. timestamps or absolute paths). With
    ``allow_new_keys``, keys present only in ``actual`` are accepted (a report format that
    gained fields), but every expected value must still match.
    """
    import math

    def num(x):
        return isinstance(x, (int, float)) and not isinstance(x, bool)

    out: list[str] = []
    if isinstance(expected, bool) != isinstance(actual, bool):
        out.append(f"{path or '/'}: expected {expected!r}, got {actual!r}")
    elif num(expected) and num(actual):
        a, b = float(actual), float(expected)
        if not ((math.isnan(a) and math.isnan(b)) or abs(a - b) <= atol + rtol * abs(b)):
            out.append(f"{path or '/'}: expected {expected!r}, got {actual!r}")
    elif isinstance(expected, dict) and isinstance(actual, dict):
        keys = (set(expected) | set(actual)) - set(ignore_keys)
        for k in sorted(keys, key=str):
            if k not in actual:
                out.append(f"{path}/{k}: missing in actual")
            elif k not in expected:
                if not allow_new_keys:
                    out.append(f"{path}/{k}: not expected")
            else:
                out += json_differences(expected[k], actual[k], rtol, atol, ignore_keys,
                                        f"{path}/{k}", allow_new_keys)
    elif isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            out.append(f"{path}: length {len(expected)} expected, got {len(actual)}")
        for i, (e, a) in enumerate(zip(expected, actual, strict=False)):
            out += json_differences(e, a, rtol, atol, ignore_keys, f"{path}[{i}]",
                                    allow_new_keys)
    elif expected != actual:
        out.append(f"{path or '/'}: expected {expected!r}, got {actual!r}")
    return out
