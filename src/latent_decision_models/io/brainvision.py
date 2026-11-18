"""Minimal BrainVision marker (.vmrk) / header (.vhdr) reader.

Only marker metadata is parsed; no signal data are read. For signal IO use
MNE (``pip install 'latent-decision-models[eeg]'``). This reader exists so that
event/marker alignment can be audited from small text files alone.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

_MK = re.compile(r"^Mk(\d+)=([^,]*),([^,]*),(\d+),(\d*),?(\d*)")


def read_vhdr_sampling_rate(path: str | Path) -> float:
    """Return the sampling rate (Hz) from a ``.vhdr`` ``SamplingInterval`` (microseconds)."""
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("SamplingInterval="):
            interval_us = float(line.split("=", 1)[1])
            if interval_us <= 0:
                raise ValueError(f"Non-positive SamplingInterval in {path}")
            return 1e6 / interval_us
    raise ValueError(f"No SamplingInterval found in {path}")


def read_vmrk(path: str | Path, sfreq: float) -> pd.DataFrame:
    """Parse a ``.vmrk`` file into a marker table.

    Columns: ``index`` (Mk number), ``type``, ``description`` (whitespace-normalised,
    e.g. ``"S 11"`` -> ``"S11"``), ``sample`` (0-based), ``time_seconds``.
    BrainVision positions are 1-based; they are converted to 0-based samples.
    """
    if sfreq <= 0:
        raise ValueError("sfreq must be positive")
    rows = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        m = _MK.match(line.strip())
        if not m:
            continue
        sample = int(m.group(4)) - 1
        rows.append(
            {
                "index": int(m.group(1)),
                "type": m.group(2),
                "description": m.group(3).replace(" ", ""),
                "sample": sample,
                "time_seconds": sample / sfreq,
            }
        )
    if not rows:
        raise ValueError(f"No markers parsed from {path}")
    return pd.DataFrame(rows).sort_values("sample", kind="stable").reset_index(drop=True)
