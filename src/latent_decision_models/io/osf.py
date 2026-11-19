"""Read-only access to small files in an OSF project (listing + capped, hash-checked download).

Mirrors the safety rules of :mod:`latent_decision_models.io.openneuro`: only an explicit
allow-list of file suffixes can be fetched, the total download is capped, and every file
is checked against the size and SHA-256 that OSF reports.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://api.osf.io/v2"
DEFAULT_MAX_TOTAL_BYTES = 50_000_000


def _with_view_only(url: str, view_only: str | None) -> str:
    if not view_only:
        return url
    parts = urllib.parse.urlsplit(url)
    query = urllib.parse.parse_qs(parts.query)
    query["view_only"] = [view_only]
    return urllib.parse.urlunsplit(parts._replace(query=urllib.parse.urlencode(query, doseq=True)))


def _get_json(url: str, view_only: str | None, timeout: float = 60, retries: int = 6) -> dict:
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(_with_view_only(url, view_only), timeout=timeout) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code != 429 or attempt == retries - 1:
                raise
            time.sleep(5 * 2 ** attempt)
    raise RuntimeError("unreachable")


def project_metadata(node: str, view_only: str | None = None) -> dict:
    """Title, public flag, license and dates of an OSF node."""
    a = _get_json(f"{API}/nodes/{node}/", view_only)["data"]["attributes"]
    return {k: a.get(k) for k in ("title", "public", "date_created", "date_modified",
                                  "node_license", "category")}


def list_files(node: str, view_only: str | None = None) -> list[dict]:
    """Recursively list ``osfstorage`` files: path, size, sha256, modified, download URL."""
    out: list[dict] = []

    def walk(url: str) -> None:
        while url:
            d = _get_json(url, view_only)
            for item in d["data"]:
                a = item["attributes"]
                if a["kind"] == "folder":
                    walk(item["relationships"]["files"]["links"]["related"]["href"])
                    continue
                out.append({"path": a["materialized_path"], "size": a.get("size"),
                            "sha256": ((a.get("extra") or {}).get("hashes") or {}).get("sha256"),
                            "modified": a.get("date_modified"),
                            "download": item["links"].get("download")})
            url = d["links"].get("next")

    walk(f"{API}/nodes/{node}/files/osfstorage/")
    return out


def check_selection(files: list[dict], suffixes: tuple[str, ...],
                    max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES) -> None:
    """Raise unless every file has an allowed suffix and the total size is under the cap."""
    refused = [f["path"] for f in files if not f["path"].endswith(suffixes)]
    if refused:
        raise PermissionError(f"Refusing files outside the allow-list {suffixes}: {refused[:5]}")
    total = sum(f["size"] or 0 for f in files)
    if total > max_total_bytes:
        raise PermissionError(f"Selection is {total} bytes, above the cap of {max_total_bytes}")


def verify(content: bytes, entry: dict) -> dict:
    """Compare downloaded bytes against OSF's reported size and SHA-256."""
    digest = hashlib.sha256(content).hexdigest()
    return {"path": entry["path"], "bytes": len(content),
            "size_ok": entry["size"] is None or len(content) == entry["size"],
            "sha256": digest,
            "sha256_ok": entry["sha256"] is None or digest == entry["sha256"]}


def fetch_files(files: list[dict], dest: str | Path, suffixes: tuple[str, ...],
                view_only: str | None = None,
                max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES,
                pause_seconds: float = 0.5) -> list[dict]:
    """Download files sequentially (OSF rate-limits parallel requests) and verify each one.

    Files already present with the expected size are re-verified, not re-downloaded.
    Raises ``ValueError`` on any size or hash mismatch.
    """
    check_selection(files, suffixes, max_total_bytes)
    dest = Path(dest)
    manifest = []
    for f in files:
        out = dest / f["path"].lstrip("/")
        if out.exists() and out.stat().st_size == f["size"]:
            content = out.read_bytes()
        else:
            for attempt in range(6):
                try:
                    with urllib.request.urlopen(_with_view_only(f["download"], view_only),
                                                timeout=120) as r:
                        content = r.read()
                    break
                except urllib.error.HTTPError as e:
                    if e.code != 429 or attempt == 5:
                        raise
                    time.sleep(5 * 2 ** attempt)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(content)
            time.sleep(pause_seconds)
        check = verify(content, f)
        if not (check["size_ok"] and check["sha256_ok"]):
            raise ValueError(f"Checksum/size mismatch for {f['path']}")
        manifest.append({**check, "osf_modified": f["modified"]})
    return manifest
