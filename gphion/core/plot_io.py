"""Lightweight utilities to persist data used for plotting so plots can be
recreated later without re-running analysis.

This module provides a single entry point `save_plot_data` that writes:
- a metadata.json with parameters and lightweight info
- a data.npz with arrays (and dicts of arrays) in a flattened key namespace

Usage:
    from .plot_io import save_plot_data
    save_plot_data(name="jump_distance_hist", payload={
        "bin_centers": bin_centers,
        "hist": jump_dist_hist,
        "bins": bins,
        "params": {"resolution": resolution, "diffusion_dim": diffusion_dim}
    })
"""
from __future__ import annotations

import os
import json
import time
from typing import Any, Dict
import numpy as np


def _flatten_for_npz(prefix: str, obj: Any, out: Dict[str, np.ndarray]):
    """Flatten nested mappings of arrays/sequences into a flat dict suitable for np.savez.
    Non-array scalars are skipped here (they go to metadata.json). Lists are converted to arrays when numeric.
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            _flatten_for_npz(f"{prefix}.{k}" if prefix else str(k), v, out)
    elif isinstance(obj, (list, tuple)):
        # Convert to array if numeric-like
        try:
            arr = np.asarray(obj)
            out[prefix] = arr
        except Exception:
            # not numeric; skip (goes into metadata)
            pass
    elif isinstance(obj, np.ndarray):
        out[prefix] = obj
    else:
        # Non-array payloads (str, numbers) are not stored in npz; metadata handles those
        pass


def _extract_metadata(obj: Any) -> Any:
    """Recursively build a JSON-serializable structure by replacing ndarrays with shapes/dtypes,
    and large lists with their lengths to keep metadata light-weight.
    """
    if isinstance(obj, dict):
        return {k: _extract_metadata(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        # Keep short lists as-is, otherwise store length
        if len(obj) <= 32:
            return [_extract_metadata(v) for v in obj]
        return {"sequence_len": len(obj)}
    if isinstance(obj, np.ndarray):
        return {"ndarray": True, "shape": obj.shape, "dtype": str(obj.dtype)}
    # primitives
    return obj


def save_plot_data(name: str, payload: Dict[str, Any], base_dir: str = "plot_data", timestamp: bool = False) -> str:
    """Save plotting payload to a folder so figures can be reproduced.

    Args:
        name: Short name of the plot or dataset (used in directory name).
        payload: Mapping with arrays (numpy) and parameters.
        base_dir: Root directory where plot-data folders will be created.
        timestamp: Whether to include a timestamp in the folder name.

    Returns:
        The directory path to the saved payload.
    """
    safe_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name).strip("._") or "plot"
    ts = time.strftime("%Y%m%d-%H%M%S") if timestamp else ""
    folder = os.path.join(base_dir, f"{safe_name}{('-' + ts) if ts else ''}")
    os.makedirs(folder, exist_ok=True)

    # Write arrays into an NPZ file
    flat: Dict[str, np.ndarray] = {}
    _flatten_for_npz("", payload, flat)
    if flat:
        np.savez(os.path.join(folder, "data.npz"), **flat)

    # Write metadata JSON
    meta = {
        "name": name,
        "saved_at": ts or time.strftime("%Y%m%d-%H%M%S"),
        "keys": sorted(list(flat.keys())),
        "metadata": _extract_metadata(payload),
    }
    with open(os.path.join(folder, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return folder
