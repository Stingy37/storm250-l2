
from __future__ import annotations

import ctypes
import gc
import logging
import os
from math import asin, cos, radians, sin, sqrt
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)



############################################################################# MISC. ###########################################################################



# Haversine distance (km) between two lon/lat points
def haversine(lon1, lat1, lon2, lat2) -> float:
    R = 6371.0  # Earth radius in km
    dlon = radians(lon2 - lon1)
    dlat = radians(lat2 - lat1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 2 * R * asin(sqrt(a))


def section_seperator(lines_to_print: int, *, log: Optional[logging.Logger] = None) -> None:
    """
    In library code, printing is usually undesirable, so:
      - if `log` is provided, we emit blank lines via logger
      - otherwise this is a no-op (keeps call sites harmless)
    """
    if lines_to_print <= 0:
        return
    if log is not None:
        for _ in range(lines_to_print):
            log.info("")
    # else: intentionally no print()


# Helper: ensure directory exists
def _ensure_dir(path: Union[str, os.PathLike]) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def _fmt_bytes(n: Union[int, float]) -> str:
    """Human-friendly byte formatter used in provenance/release tooling."""
    try:
        n = float(n)
    except Exception:
        return str(n)
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    i = 0
    while n >= 1024.0 and i < len(units) - 1:
        n /= 1024.0
        i += 1
    if i == 0:
        return f"{int(n)} {units[i]}"
    return f"{n:.2f} {units[i]}"


def trimmed_cluster_center(vals, k=2.5):
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan")

    med = np.median(vals)
    mad = np.median(np.abs(vals - med)) or 1.0

    core = vals[np.abs(vals - med) <= k * mad]
    return float(np.mean(core))


def mad_scale(vals: np.ndarray) -> float:
    """Median absolute deviation scale with a lower bound to avoid zero division."""
    vals = np.asarray(vals, dtype=float)
    v = vals[np.isfinite(vals)]
    if v.size == 0:
        return 1.0
    
    # taking the median -> now, a single scalar value
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med)))
    return max(mad, 1.0)  # avoid near-zero scale


# Helper: weighted dimensions
def _compute_weighted_dimensions(
    widths_m: np.ndarray,
    heights_m: np.ndarray,
    debug: bool = False,
) -> tuple[float, float, np.ndarray]:
    """
    Compute weighted average of dimensions (width, height in meters).

    1. Find a “typical center” (most common cluster)
    2. Penalize points farther from that cluster, normalized using "typical spread" of dimension
    3. Average using those penalties as weights

    Returns (avg_width_m, avg_height_m, weights_array).
    """
    widths = np.asarray(widths_m, dtype=float)
    heights = np.asarray(heights_m, dtype=float)
    assert widths.shape == heights.shape

    # find which width and height values are most common
    #   - width is a list of widths
    #   - height is a list of heights 
    #   - single scalar value
    center_w = trimmed_cluster_center(widths)
    center_h = trimmed_cluster_center(heights)
    if debug:
        logger.info("[_compute_weighted_dimensions] center_w=%.1f, center_h=%.1f", center_w, center_h)

    # calculate the median absolute spread of width / height list 
    #   - median(|Wi - median(w)|)
    #                                               |- where average is median, so resistant
    #   - SINGLE SCALAR VALUE, so represents the "average" spread of widths / heights
    scale_w = mad_scale(widths)    # single value for typical deviation of widths
    scale_h = mad_scale(heights)   # single value for typical deviation of widths

    # absolute deviation for each dimension in the list
    #   - based on the most common dimension calculated from trimmed_cluster_center
    #   - still a np.array
    # dev_w = [|w1 - m_w|, |w2 - m_w|, ...]
    dev_w = np.abs(widths - center_w)
    dev_h = np.abs(heights - center_h)

    # now finally, calculate a score based on how large the spread of a certain dimension is
    #   - we normalize with scale_w/h, which represents the typical spread 
    #       - ex. if dev_w = 1 and scale_w = .1, then 1 IN THIS CONTEXT represents a large spread 
    #                                                        \- assign a low score
    #
    #   - score = 1 / (1 + normalized_spread) bounds scores from (0, 1)
    #        \- if normalized_spread = infinity, then score = 0
    #        \- if normalized_spread = 0,        then score = 1
    # 
    #   - still a np.array -\ (lots of element-wise operations)
    #       - w_score.shape == widths.shape
    #       - h_score.shape == heights.shape
    # 
    #                       /-- normalized_spread --\
    w_score = 1.0 / (1.0 + (dev_w / (scale_w + 1e-12)))
    h_score = 1.0 / (1.0 + (dev_h / (scale_h + 1e-12)))

    # now combine w_score and h_score into ONE score for a pair of width x height dimensions
    #       \- a single width x height pair must have consistency in both dimensions to be weighted heavily 
    # 
    # combined.shape = widths.shape = heights.shape (still element-wise operations)
    combined = w_score * h_score

    # fallback to equal weights if something weird happens
    if float(np.sum(combined)) <= 0:
        combined = np.ones_like(combined)

    # normalize so that weights (scores) sum to one
    weights = combined / float(np.sum(combined))

    # find weighted average (SINGLE SCALAR) using the weights/scores calculated above
    #                                                       \- penalizing inconsistent dimensions
    avg_w = float(np.sum(weights * widths))
    avg_h = float(np.sum(weights * heights))

    if debug:
        logger.info("[_compute_weighted_dimensions] avg_w=%.1f m, avg_h=%.1f m", avg_w, avg_h)
    return avg_w, avg_h, weights


######################################################################## RAM CLEANUP ########################################################################


def _safe_close(obj: Any) -> None:
    # Best-effort close for things that may hold file handles or buffers
    for name in ("close", "shutdown", "stop", "terminate", "flush"):
        try:
            getattr(obj, name)()
            return
        except Exception:
            pass


def aggressive_memory_cleanup(locals_dict: Dict[str, Any]) -> None:
    """
    Best-effort cleanup for long-running pipeline scripts.

    It will:
      - close matplotlib figures if matplotlib is available
      - attempt to close common resources in locals()
      - drop a few known-large variables by name
      - clear known module-level caches if present
      - clear TF / torch GPU caches if those libs exist
      - run GC
      - attempt malloc_trim on Linux/glibc (safe if unavailable)
    """
    # 1) Close plotting state (matplotlib holds a LOT)
    try:
        import matplotlib.pyplot as plt  # local import: avoid hard dependency at import-time

        plt.close("all")
    except Exception:
        pass

    # 2) Shut down any executors/threads/queues 
    for _, v in list(locals_dict.items()):
        # queues, threads, executors, file systems, netCDF datasets, s3fs/gcsfs, etc.
        try:
            _safe_close(v)
        except Exception:
            pass

    # 3) Drop large objects by name (add / remove as necessary)
    big_names = [
        "lsr_df",
        "spc_df",
        "grs_df",
        "grouped",
        "linked_radar_df",
        "bboxed_df",
        "files",
        "full_df"
    ]
    for name in big_names:
        if name in locals_dict:
            locals_dict[name] = None

    # 4) Clear module-level caches if you have them
    for cache_name in [
        "_GRID_PLAN_CACHE",  # example from earlier code
        "_s3fs",
        "_s3client",
    ]:
        if cache_name in globals():
            try:
                obj = globals()[cache_name]
                _safe_close(obj)
            except Exception:
                pass
            globals()[cache_name] = None

    # 5) Free TF/PyTorch contexts if they exist (safe no-ops if not installed)
    try:
        import tensorflow as tf  # type: ignore

        tf.keras.backend.clear_session()
    except Exception:
        pass

    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
    except Exception:
        pass

    # 6) Force Python GC
    gc.collect()
    gc.collect()  # call twice sometimes helps drop finalizers/cycles

    # 7) Ask glibc to return free arenas to the OS (Linux only)
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass
