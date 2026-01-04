"""
storm250/io.py

Low-level I/O and cache helpers.

This module is intentionally "dumb":
- It reads/writes local files.
- It does NOT know about cfg/root_dir. Callers must pass resolved absolute paths.
"""
from __future__ import annotations

import gzip
import json
import logging
import os
import pickle
from pathlib import Path
from typing import Any, Mapping, Optional, Union

import numpy as np
import numpy.ma as _ma

from .utils import _ensure_dir

logger = logging.getLogger(__name__)

PathLike = Union[str, os.PathLike]


############################################ HELPERS ############################################


def _as_path(p: PathLike) -> Path:
    """Normalize user path input (str/PathLike) to pathlib.Path."""
    return p if isinstance(p, Path) else Path(os.fspath(p))


def _save_gz_pickle(obj: Any, path: PathLike, debug: bool = False) -> None:
    try:
        p = _as_path(path)
        _ensure_dir(str(p.parent))
        with gzip.open(p, "wb") as f:
            pickle.dump(obj, f, protocol=4)
    except Exception:
        if debug:
            logger.exception("[find_radar_scans] skeleton write failed %s", path)


def _load_gz_pickle(path: PathLike, debug: bool = False) -> Optional[Any]:
    try:
        p = _as_path(path)
        with gzip.open(p, "rb") as f:
            return pickle.load(f)
    except Exception:
        if debug:
            logger.exception("[find_radar_scans] skeleton read failed %s", path)
        return None


def _save_field_pack(
    base_path: PathLike,
    field_dict: Mapping[str, Any],
    downcast: bool = True,
    debug: bool = False,
) -> None:
    """
    Save a single field from a Py-ART radar object into two small files:
      - {base_path}.npz  : compressed data + mask
      - {base_path}.json : tiny field-specific metadata (everything except 'data')
    """
    try:
        bp = _as_path(base_path)
        _ensure_dir(str(bp.parent))

        arr = field_dict["data"]
        data = np.asarray(arr, dtype=np.float32 if downcast else np.asarray(arr).dtype)
        mask = np.asarray(_ma.getmaskarray(arr), dtype=np.uint8)

        npz_path = Path(str(bp) + ".npz")
        json_path = Path(str(bp) + ".json")

        np.savez_compressed(npz_path, data=data, mask=mask)

        meta = {k: v for k, v in field_dict.items() if k != "data"}
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(meta, f)
    except Exception:
        if debug:
            logger.exception("[find_radar_scans] field save failed %s", base_path)


def _load_field_pack(base_path: PathLike, debug: bool = False) -> Optional[dict]:
    """
    Load a single field pack and return a pyart-compatible field dict, where a field radar product is from pyart radar object. 
    """
    try:
        bp = _as_path(base_path)
        npz_path = Path(str(bp) + ".npz")
        json_path = Path(str(bp) + ".json")

        z = np.load(npz_path, allow_pickle=False)
        with open(json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        data = _ma.MaskedArray(z["data"], mask=z["mask"].astype(bool))
        meta["data"] = data
        return meta
    except Exception:
        if debug:
            logger.exception("[find_radar_scans] field load failed %s", base_path)
        return None


############################################ LOADING RADAR ############################################


_SKELETON_SUFFIX = ".skeleton.pkl.gz"

def _radar_base_from_skeleton_path(skeleton_path: PathLike) -> str:
    """
    Helper to load Py-ART radar objects from cache.

    Convert:
        /.../KHGX20220322_120125_V06.skeleton.pkl.gz
    into:
        /.../KHGX20220322_120125_V06
    which is the base used by field packs:
        base + ".reflectivity".npz/.json
        base + ".velocity".npz/.json
        ...
    """
    p = str(_as_path(skeleton_path))
    if not p.endswith(_SKELETON_SUFFIX):
        raise ValueError(f"Not a skeleton cache path: {p}")
    return p[: -len(_SKELETON_SUFFIX)]


def _load_radar_from_skeleton_and_field_packs(
    skeleton_path: PathLike,
    field_keys: list[str],
    *,
    debug: bool = False,
) -> Optional[Any]:
    """
    Rebuild a Py-ART Radar object from:
      - skeleton gz-pickle (geometry-only Radar with empty fields)
      - field pack(s) saved at base + f".{key}".npz/.json
    """
    radar = _load_gz_pickle(skeleton_path, debug=debug)
    if radar is None:
        return None

    base = _radar_base_from_skeleton_path(skeleton_path)

    # populate requested fields (skip missing packs)
    for k in field_keys:
        pack = _load_field_pack(base + f".{k}", debug=debug)
        if pack is not None:
            radar.fields[k] = pack

    return radar
