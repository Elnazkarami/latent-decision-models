"""NIfTI-1 header parsing from the first bytes of a file (no image data).

Uses nibabel (optional ``[fmri]`` extra). Works on a partial ``.nii`` or ``.nii.gz``
byte prefix, as returned by :func:`latent_decision_models.io.openneuro.fetch_range`.
"""

from __future__ import annotations

import io
import zlib

import numpy as np

HEADER_BYTES = 348
_SPACE = {0: "unknown", 1: "scanner_anat", 2: "aligned_anat", 3: "talairach", 4: "mni_152",
          5: "template_other"}


def _decompress_prefix(raw: bytes, n: int = HEADER_BYTES + 4) -> bytes:
    out = zlib.decompressobj(16 + zlib.MAX_WBITS).decompress(raw)
    if len(out) < n:
        raise ValueError("gzip prefix too short to hold a NIfTI header")
    return out


def parse_header(raw: bytes) -> dict:
    """Summarize a NIfTI-1 header: shape, voxel size, TR, units, affines and text fields."""
    try:
        import nibabel as nib
    except ImportError as e:  # pragma: no cover - depends on optional extra
        raise ImportError("parse_header needs nibabel: pip install "
                          "'latent-decision-models[fmri]'") from e
    if raw[:2] == b"\x1f\x8b":
        raw = _decompress_prefix(raw)
    if len(raw) < HEADER_BYTES:
        raise ValueError(f"need at least {HEADER_BYTES} bytes, got {len(raw)}")
    hdr = nib.Nifti1Header.from_fileobj(io.BytesIO(raw[:HEADER_BYTES]), check=False)
    dims = [int(x) for x in hdr["dim"][1:hdr["dim"][0] + 1]]
    zooms = [float(x) for x in hdr.get_zooms()]
    space_units, time_units = hdr.get_xyzt_units()
    q_code, s_code = int(hdr["qform_code"]), int(hdr["sform_code"])
    sform = hdr.get_sform() if s_code else None
    qform = hdr.get_qform() if q_code else None
    affine = sform if sform is not None else qform
    origin_voxel = (np.linalg.inv(affine) @ [0, 0, 0, 1])[:3].round(2).tolist() \
        if affine is not None else None
    return {
        "magic": hdr["magic"].item().decode(errors="replace"),
        "shape": dims,
        "n_volumes": dims[3] if len(dims) > 3 else 1,
        "voxel_size_mm": [round(z, 4) for z in zooms[:3]],
        "repetition_time": round(zooms[3], 6) if len(zooms) > 3 else None,
        "space_units": space_units, "time_units": time_units,
        "datatype": str(hdr.get_data_dtype()),
        "scl_slope": float(hdr["scl_slope"]) if np.isfinite(hdr["scl_slope"]) else None,
        "qform_code": _SPACE.get(q_code, str(q_code)),
        "sform_code": _SPACE.get(s_code, str(s_code)),
        "affine": None if affine is None else np.round(affine, 4).tolist(),
        "world_origin_voxel": origin_voxel,
        "descrip": hdr["descrip"].item().decode(errors="replace").strip("\x00 "),
        "aux_file": hdr["aux_file"].item().decode(errors="replace").strip("\x00 "),
        "slice_duration": float(hdr["slice_duration"]),
        "vox_offset": float(hdr["vox_offset"]),
    }
