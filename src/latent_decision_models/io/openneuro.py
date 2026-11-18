"""Metadata-only access to pinned OpenNeuro snapshots.

Enforces the project rule "no raw MRI/EEG download without approval" in code:
only small text sidecars / marker files can be fetched, each file is size-capped,
and every download is checked against the snapshot's reported size.
"""

from __future__ import annotations

import fnmatch
import json
import urllib.request
from pathlib import Path

from ..provenance import sha256_file

GRAPHQL_URL = "https://openneuro.org/crn/graphql"
METADATA_SUFFIXES = (".tsv", ".json", ".vhdr", ".vmrk", ".txt", "README", "CHANGES",
                     ".bidsignore")
DEFAULT_MAX_BYTES = 2_000_000


def snapshot_files(dataset_id: str, tag: str, timeout: float = 60) -> list[dict]:
    """Return ``[{filename, size, annexed, urls}]`` for every file in a snapshot."""
    query = (f'{{ snapshot(datasetId:"{dataset_id}", tag:"{tag}") '
             "{ files(recursive:true) { filename size annexed urls } } }")
    req = urllib.request.Request(GRAPHQL_URL, data=json.dumps({"query": query}).encode(),
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.load(resp)
    if payload.get("errors"):
        raise RuntimeError(f"OpenNeuro API error: {payload['errors']}")
    return payload["data"]["snapshot"]["files"]


def is_metadata_path(path: str) -> bool:
    return path.endswith(METADATA_SUFFIXES)


def select_files(files: list[dict], patterns: list[str]) -> list[dict]:
    """Filter snapshot files by glob patterns; refuses non-metadata matches."""
    chosen = [f for f in files if any(fnmatch.fnmatch(f["filename"], p) for p in patterns)]
    refused = [f["filename"] for f in chosen if not is_metadata_path(f["filename"])]
    if refused:
        raise PermissionError(
            f"Refusing non-metadata files (signal data needs explicit approval): {refused[:5]}"
        )
    return chosen


def fetch_files(files: list[dict], dest: str | Path, max_bytes: int = DEFAULT_MAX_BYTES,
                timeout: float = 60) -> list[dict]:
    """Download selected metadata files into ``dest`` preserving dataset paths.

    Returns a manifest with path, reported size, sha256 and source URL.
    """
    dest = Path(dest)
    manifest = []
    for f in files:
        if not is_metadata_path(f["filename"]):
            raise PermissionError(f"Not a metadata file: {f['filename']}")
        if f["size"] is not None and f["size"] > max_bytes:
            raise PermissionError(f"{f['filename']} is {f['size']} B > cap {max_bytes} B")
        url = f["urls"][0]
        out = dest / f["filename"]
        out.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            data = resp.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise PermissionError(f"{f['filename']} exceeded cap while downloading")
        if f["size"] is not None and len(data) != f["size"]:
            raise OSError(f"Size mismatch for {f['filename']}: {len(data)} != {f['size']}")
        out.write_bytes(data)
        manifest.append({"path": f["filename"], "bytes": len(data),
                         "sha256": sha256_file(out), "url": url.split("?")[0]})
    return manifest
