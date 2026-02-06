"""
Centralized configuration loader + validator + path resolution.

- All "knobs" live in YAML (configs/v0.1.yaml)
- Code can keep using relative paths like "Datasets/..." (from old colab setup)
- Validation can create required directories (so EC2/EBS setup is painless)
"""

from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

import logging
import yaml  # PyYAML

logger = logging.getLogger(__name__)
PathLike = Union[str, os.PathLike]



################################################################# DEFAULT CONFIGURATION #################################################################
#         \- determines WHAT knobs / parameters are changeable (sort of like defining parameters of a function... except function -> dataset pipeline)      
#                                                                               |- specific arguments supplied later via yaml, overrides default values


DEFAULT_CONFIG: Dict[str, Any] = {
    # pointer to schema yaml 
    "schema": {
        "context": "storm250_context_v1.0.0"
    },

    # self describing metadata about the specific dataset run
    "dataset": {
        "name": "storm250_level2",
        "version": "1.0.0"
    },

    # EBS mount mirror root (replaces /content/drive/MyDrive/... on Colab)
    "root_dir": "/data/Storm250",

    # If True, validate_config will create -p root_dir + all directories in cfg["paths"].
    "ensure_dirs": True,

    # Project paths: keep these as Drive-style relative paths for minimal refactor.
    # resolve_path() will map them under root_dir.
    "paths": {
        # GR-S tracks 
        "grs_raw_dir": "Datasets/cell_tracks/raw_grs",
        "grs_processed_dir": "Datasets/cell_tracks/processed_grs",
        "grs_linked_dir": "Datasets/cell_tracks/linked_grs",

        # NEXRAD caching / local raw (optional) 
        "nexrad_lv2_cache_dir": "Datasets/nexrad_datasets/level_two_raw",
        "nexrad_lv3_cache_dir": "Datasets/nexrad_datasets/level_three_raw",

        # Surface obs (optional for now) 
        "lsr_raw_dir": "Datasets/surface_obs_datasets/lsr_raw",
        "lsr_reports_dir": "Datasets/surface_obs_datasets/lsr_reports",
        "spc_reports_dir": "Datasets/surface_obs_datasets/spc_reports",
        "linked_obs_cache_dir": "Datasets/surface_obs_datasets/linked_obs_cache",

        # Scratch, just to mirror what we already have on colab / drive 
        "temp_dir": "Datasets/temp",

        # Final dataset outputs (Storm250-L2) 
        "dataset_out_dir": "Datasets/training_datasets/level_two",
        "starter_out_dir": "Datasets/training_datasets/starter",

        # Packaging staging + final releases 
        "zenodo_bundle_dir": "Datasets/training_datasets/zenodo_bundle",
        "release_zenodo_dir": "Release/zenodo",

        # Logs 
        "logs_dir": "Logs",
        "plots_dir": "Logs/plots",
        "env_dir": "Logs/environment",
    },

    # Core build knobs 
    "build": {
        "max_range_km": 250.0,
        "time_tolerance_s": 29,
        "min_track_samples": 60,
        "max_gap_hours": 6,
        "reflectivity_dbz_thresh": 20.0,
        "buffer_km": 5.0,

        # Cropping/bbox parameters
        "min_blob_size": 6000,
        "grid_resolution_m": 250.0,
        "include_nearby_km": 3.0,
        "debug_plot_limit": 0,

        # Data loading parameters
        "lsr_force_refresh": False,
        "grs_base_url": "https://data-osdf.rda.ucar.edu/ncar/rda/d841006/tracks",
        "grs_timeout_s": 10,

        # Product filtering (NEXRAD products to process)
        "nexrad_products": ["reflectivity"],

        # Resume / skipping behavior (important on long EC2 runs)
        "skip_existing": True,
        "resume": True,
        "dry_run": False,

        # Multiprocessing / workers (tune per instance)
        "use_multiprocessing": True,
        "n_workers": 8,
    },

    # S3 / AWS defaults 
    "aws": {
        "nexrad_bucket": "noaa-nexrad-level2",
        "s3_anon": True,
        "aws_region": None,
        "retries": 5,
    },

    # Logging defaults 
    "logging": {
        "level": "INFO",
    },

    # Execution configuration (runtime parameters)
    "execution": {
        "year": 2017,
        "rewrite_existing": False,
        "multiprocessing_method": "fork",
    },

    # Radar metadata reference
    "radar_info_yaml": "radar_info.yaml",
}


################################################################# ULTILITIES #################################################################


# Overwriting behavior (overwrites default configs with "arguments" supplied by .yaml file)
def _deep_update(base: Dict[str, Any], updates: Mapping[str, Any]) -> Dict[str, Any]:
    """Recursively merge updates into base (dicts only)."""
    for k, v in updates.items():
        if isinstance(v, Mapping) and isinstance(base.get(k), Mapping):
            base[k] = _deep_update(dict(base[k]), v)
        else:
            base[k] = v
    return base

def _expand_path(p: PathLike) -> Path:
    """Expand env vars + ~ and return Path."""
    s = os.fspath(p)
    s = os.path.expandvars(s)
    s = os.path.expanduser(s)
    return Path(s)

# Easy backwards compatibility for colab file-paths (maps to corresponding EBS location)
def resolve_path(cfg: Mapping[str, Any], p: PathLike) -> Path:
    """
    Resolve a relative path under cfg["root_dir"], and returns the full path for usage later. 

    Examples:
      resolve_path(cfg, "Datasets/temp") -> /data/Storm250/Datasets/temp
      resolve_path(cfg, "/data/Storm250/Datasets/temp") -> same absolute path

    Notes:
    - We treat ANY non-absolute path as relative to root_dir.
    - This is what allows us to keep Drive-style paths in the YAML and code 
        - Essentially, we allow for same file structure across different environments -> very flexible. 

    Returned path points to full file location.
    """
    root = _expand_path(cfg.get("root_dir", DEFAULT_CONFIG["root_dir"]))
    path = _expand_path(p)

    # Strip leading "./" to make relative paths cleaner. We convert ANY non-absolute path to one under root_dir
    #   ex. "Datasets/temp" -> "<root_dir>/Datasets/temp"
    #       - this ensures file-structure portability across different environments (EC2, local, etc.)
    if not path.is_absolute():
        rel = Path(str(path)).as_posix()
        if rel.startswith("./"):
            rel = rel[2:]
        return root / rel
    return path


def get_path(cfg: Mapping[str, Any], key: str) -> Path:
    """
    Resolve a named path in cfg["paths"] — that is, if that path has been defined in configs, should 
    """
    paths = cfg.get("paths", {})
    if key not in paths:
        raise KeyError(f"Config paths missing key '{key}'. Available keys: {sorted(paths.keys())}")
    return resolve_path(cfg, paths[key])


####################################################################################### BUILD CONFIG #######################################################################################


def validate_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate + normalize config in-place and return it. 
    Makes sure the supplied config can be used, meaning that all the necessary configs must be provided in the supplied yaml. 
    """

    # Root dir
    if "root_dir" not in cfg:
        cfg["root_dir"] = DEFAULT_CONFIG["root_dir"]

    root = resolve_path(cfg, ".")  # root itself
    cfg["root_dir"] = str(_expand_path(cfg["root_dir"]))

    # Ensure required top-level sections exist in the configs
    for section in ("paths", "build", "aws", "logging", "execution"):
        if section not in cfg or not isinstance(cfg[section], dict):
            cfg[section] = deepcopy(DEFAULT_CONFIG[section])

    # Validate critical numeric knobs 
    b = cfg["build"]
    for k in ("max_range_km", "reflectivity_dbz_thresh", "buffer_km", "grid_resolution_m", "include_nearby_km"):
        if k in b:
            b[k] = float(b[k])
    for k in ("time_tolerance_s", "min_track_samples", "max_gap_hours", "n_workers", "min_blob_size", "debug_plot_limit", "grs_timeout_s"):
        if k in b:
            b[k] = int(b[k])

    if b["max_range_km"] <= 0:
        raise ValueError("build.max_range_km must be > 0")
    if b["time_tolerance_s"] < 0:
        raise ValueError("build.time_tolerance_s must be >= 0")
    if b["n_workers"] <= 0:
        raise ValueError("build.n_workers must be >= 1")

    # Validate execution section
    e = cfg["execution"]
    if "year" in e:
        e["year"] = int(e["year"])
        if e["year"] < 1900 or e["year"] > 2100:
            raise ValueError("execution.year must be a valid year (1900-2100)")
    if "rewrite_existing" in e:
        e["rewrite_existing"] = bool(e["rewrite_existing"])
    if "multiprocessing_method" in e:
        valid_methods = ["fork", "spawn", "forkserver"]
        if e["multiprocessing_method"] not in valid_methods:
            raise ValueError(f"execution.multiprocessing_method must be one of {valid_methods}")

    # Ensure dirs 
    ensure_dirs = bool(cfg.get("ensure_dirs", True))
    # Create directory if it doesn't exist and ensure_dirs is true in configs
    if ensure_dirs:
        root.mkdir(parents=True, exist_ok=True)

        # Create all configured dirs under cfg["paths"]
        for name, rel in cfg["paths"].items():
            p = resolve_path(cfg, rel)
            p.mkdir(parents=True, exist_ok=True)

    return cfg


def default_config() -> Dict[str, Any]:
    """
    Return a fresh default config dict.
    """
    return deepcopy(DEFAULT_CONFIG)


# main config stuff... loads, overwrites, validates, returns
def load_config(path: PathLike) -> Dict[str, Any]:
    """
    Load YAML config from `path`, merge on top of DEFAULT_CONFIG, validate and return.

    Pattern:
      cfg = load_config("configs/v0.1.yaml")
      out_dir = get_path(cfg, "dataset_out_dir")
      tmp = resolve_path(cfg, "Datasets/temp")
    """
    path = _expand_path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        user_cfg = yaml.safe_load(f) or {}

    if not isinstance(user_cfg, dict):
        raise ValueError(f"Config YAML must parse to a dict; got {type(user_cfg)}")

    # update the default config with the supplied config
    cfg = default_config()
    cfg = _deep_update(cfg, user_cfg)

    cfg = validate_config(cfg)
    return cfg
