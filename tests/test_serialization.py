from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
import numpy.ma as ma
import pandas as pd
import pytest
import yaml
import h5py
from storm250.serialization import save_df_for_training


############################################################ HELPERS ############################################################


def _find_repo_root(start: Path) -> Path:
    """
    Walk upward until we find a folder that looks like the repo root (has 'schema/').

    Bit of patchwork by AI, but it "works" (unless we change filestructure, which we won't.)
    """
    cur = start.resolve()
    for _ in range(8):
        if (cur / "schema").exists():
            return cur
        cur = cur.parent
    # fallback: current working dir
    cwd = Path.cwd().resolve()
    if (cwd / "schema").exists():
        return cwd
    raise FileNotFoundError(
        "Could not locate repo root (expected a 'schema/' directory). "
        "Run pytest from the repo root, or ensure 'schema/' exists."
    )


def _schema_paths() -> tuple[Path, Path]:
    """
    Bit of patchwork by AI, but it "works" (unless we change filestructure, which we won't.)
    """
    root = _find_repo_root(Path(__file__).parent)
    ctx = root / "schema" / "storm250_context_v1.0.0.yaml"
    prod = root / "schema" / "storm250_product_v1.0.0.yaml"
    if not ctx.exists():
        raise FileNotFoundError(f"Missing context schema YAML: {ctx}")
    if not prod.exists():
        raise FileNotFoundError(f"Missing product schema YAML: {prod}")
    return ctx, prod


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _as_str(x: Any) -> str:
    # h5py sometimes returns bytes
    if isinstance(x, (bytes, np.bytes_)):
        return x.decode("utf-8", errors="replace")
    return str(x)


########################################################### FAKE RADAR ###########################################################
#                                         (used to mock Py-ART radar and avoid network tests)


@dataclass
class FakeRadar:
    """
    Minimal stand-in for the parts of a Py-ART Radar object that save_df_for_training touches.
    """
    fields: dict
    azimuth: dict
    range: dict
    fixed_angle: dict
    elevation: dict
    sweep_start_ray_index: dict
    sweep_end_ray_index: dict
    latitude: dict
    longitude: dict
    altitude: dict
    metadata: dict
    nsweeps: int

    # optional attributes referenced defensively in your code
    nrays: int | None = None


def make_fake_radar_multisweep(
    *,
    field_key: str = "reflectivity",
    units: str = "dBZ",
    rays_per_sweep: list[int] = [5, 3],
    ngates: int = 7,
    site_lat: float = 35.33,
    site_lon: float = -97.28,
    site_alt: float = 370.0,
) -> FakeRadar:
    """
    Helper: makes a FakeRadar object mimicking Py-ART radar object that contains multiple fields + sweeps. 

    Simulates the full cropped radar object returned as part of build_bboxes_for_linked_df.
    """
    ns = len(rays_per_sweep)
    starts = np.cumsum([0] + rays_per_sweep[:-1]).astype(int)
    ends = (starts + np.array(rays_per_sweep, dtype=int)).astype(int)
    nrays_total = int(sum(rays_per_sweep))

    # Create masked data to verify NaN fill behavior
    raw = np.linspace(0, 50, nrays_total * ngates, dtype=np.float32).reshape(nrays_total, ngates)
    mask = np.zeros_like(raw, dtype=bool)
    mask[0, 0] = True
    mask[-1, -1] = True
    data = ma.MaskedArray(raw, mask=mask)

    # azimuth in degrees per ray (monotonic-ish; unwrap-safe)
    az = np.linspace(0, 359, nrays_total, dtype=np.float32)

    # range in meters per gate
    rng = np.linspace(250.0, 250.0 * ngates, ngates, dtype=np.float32)

    # fixed angles per sweep + per-ray elevation
    fixed = np.array([0.5, 1.5], dtype=np.float32)[:ns]
    elev = np.repeat(fixed, rays_per_sweep).astype(np.float32)

    fields = {
        field_key: {
            "data": data,
            "units": units,
        }
    }

    return FakeRadar(
        fields=fields,
        azimuth={"data": az},
        range={"data": rng},
        fixed_angle={"data": fixed},
        elevation={"data": elev},
        sweep_start_ray_index={"data": starts},
        sweep_end_ray_index={"data": ends},
        latitude={"data": np.array([site_lat], dtype=np.float32)},
        longitude={"data": np.array([site_lon], dtype=np.float32)},
        altitude={"data": np.array([site_alt], dtype=np.float32)},
        metadata={},  # non-pseudo
        nsweeps=ns,
        nrays=nrays_total,
    )


def make_fake_radar_pseudo(
    *,
    field_key: str = "reflectivity",
    units: str = "dBZ",
    nrays: int = 6,
    ngates: int = 7,
    site_lat: float = 35.33,
    site_lon: float = -97.28,
    site_alt: float = 370.0,
) -> FakeRadar:
    """
    Helper: makes a FakeRadar object mimicking Py-ART radar object that contains one host sweep with metadata corresponding to pseudo-composite. 

    Simulates the pseudo-composite radar object returned as part of build_bboxes_for_linked_df.
    """
    # pseudo: single host sweep, metadata flag present
    raw = np.linspace(0, 50, nrays * ngates, dtype=np.float32).reshape(nrays, ngates)
    mask = np.zeros_like(raw, dtype=bool)
    mask[0, 1] = True
    data = ma.MaskedArray(raw, mask=mask)

    az = np.linspace(0, 359, nrays, dtype=np.float32)
    rng = np.linspace(250.0, 250.0 * ngates, ngates, dtype=np.float32)
    fixed = np.array([0.5], dtype=np.float32)
    elev = np.full((nrays,), 0.5, dtype=np.float32)

    fields = {field_key: {"data": data, "units": units}}

    return FakeRadar(
        fields=fields,
        azimuth={"data": az},
        range={"data": rng},
        fixed_angle={"data": fixed},
        elevation={"data": elev},
        sweep_start_ray_index={"data": np.array([0], dtype=int)},
        sweep_end_ray_index={"data": np.array([nrays], dtype=int)},
        latitude={"data": np.array([site_lat], dtype=np.float32)},
        longitude={"data": np.array([site_lon], dtype=np.float32)},
        altitude={"data": np.array([site_alt], dtype=np.float32)},
        metadata={"pseudo_host_sweep": 0},
        nsweeps=1,
        nrays=nrays,
    )


############################################################ TESTS ############################################################


# Schema tests 
def test_schema_yamls_exist_and_have_required_keys():
    ctx_path, prod_path = _schema_paths()
    ctx = _load_yaml(ctx_path)
    prod = _load_yaml(prod_path)

    for name, d in [("context", ctx), ("product", prod)]:
        assert "schema_name" in d, f"{name} schema missing 'schema_name'"
        assert "schema_version" in d, f"{name} schema missing 'schema_version'"
        assert isinstance(d["schema_name"], str) and d["schema_name"].strip()
        assert isinstance(d["schema_version"], str) and d["schema_version"].strip()

    # Product schema must define the datasets your writer creates
    assert "datasets" in prod and isinstance(prod["datasets"], dict), "product schema missing 'datasets' dict"

    required_dsets = [
        "data",
        "azimuth_deg",
        "range_m",
        "elevation_deg",
        "azimuth_host_sweep_index",
        "bbox_min_lat",
        "bbox_max_lat",
        "bbox_min_lon",
        "bbox_max_lon",
    ]
    for ds in required_dsets:
        assert ds in prod["datasets"], f"product schema missing dataset definition for '{ds}'"


# Fake radar + contract tests 
def test_save_fake_multisweep_writes_schema_compliant_hdf(tmp_path: Path):
    ctx_path, prod_path = _schema_paths()
    ctx_schema = _load_yaml(ctx_path)
    prod_schema = _load_yaml(prod_path)

    # Two timesteps, second timestep missing scan -> should produce all-NaN frame at t=1
    scan0 = make_fake_radar_multisweep(field_key="reflectivity")
    df = pd.DataFrame(
        {
            "time": ["2020-05-01T00:00:00Z", "2020-05-01T00:05:00Z"],
            "radar_site": ["KTLX", "KTLX"],
            "storm_id": [123, 123],
            "min_lat": [34.9, 34.9],
            "max_lat": [35.7, 35.7],
            "min_lon": [-98.2, -98.2],
            "max_lon": [-96.8, -96.8],
            "reflectivity_scan": [scan0, None],
            "reflectivity_matched_volume_s3_key": ["fake/key0", ""],
        }
    )

    out = save_df_for_training(
        df,
        base_dir=str(tmp_path),
        debug=False,
        drop_scans_after_save=False,
        dataset_version="1.0.0-test",
        context_schema_yaml_path=str(ctx_path),
        product_schema_yaml_path=str(prod_path),
    )
    assert out, "Expected at least one saved product"

    prod_files = [Path(x["path"]) for x in out if x.get("path")]
    assert len(prod_files) == 1
    prod_file = prod_files[0]
    assert prod_file.exists()

    storm_dir = prod_file.parent

    # context + schema sidecar should exist and be referenced by relpaths in product file attrs
    context_h5s = list(storm_dir.glob("*_context_*.h5"))
    assert len(context_h5s) == 1, f"Expected 1 context h5, got {len(context_h5s)}"
    context_h5 = context_h5s[0]
    sidecars = list(storm_dir.glob("*_context_*.schema.json"))
    assert len(sidecars) == 1, f"Expected 1 context schema sidecar, got {len(sidecars)}"
    sidecar = sidecars[0]

    # verify sidecar JSON aligned with context 
    sj = json.loads(sidecar.read_text(encoding="utf-8"))
    assert sj.get("schema_name") == ctx_schema.get("schema_name")
    assert sj.get("schema_version") == ctx_schema.get("schema_version")
    assert sj.get("applies_to") == context_h5.name
    assert "columns" in sj and isinstance(sj["columns"], dict)
    # must include key that writer adds
    assert "time_unix_ms" in sj["columns"]

    # verify product HDF structure + attrs + schema contract 
    with h5py.File(prod_file, "r") as f:
        # file-level required attrs
        assert _as_str(f.attrs["dataset_version"]) == "1.0.0-test"
        assert _as_str(f.attrs["schema_name"]) == _as_str(prod_schema["schema_name"])
        assert _as_str(f.attrs["schema_version"]) == _as_str(prod_schema["schema_version"])

        assert _as_str(f.attrs["radar_site"]) == "KTLX"
        assert _as_str(f.attrs["storm_id"]) == "123"
        assert _as_str(f.attrs["product_prefix"]) == "reflectivity"

        # relpaths point to real files
        ctx_rel = Path(_as_str(f.attrs["context_relpath"]))
        sc_rel = Path(_as_str(f.attrs["context_schema_relpath"]))
        assert (storm_dir / ctx_rel).exists(), "context_relpath does not exist"
        assert (storm_dir / sc_rel).exists(), "context_schema_relpath does not exist"

        # datasets exist (contract check)
        for ds_name in prod_schema.get("datasets", {}).keys():
            assert ds_name in f, f"Missing dataset '{ds_name}' in product HDF"

        # data shape contract basics
        data = f["data"]
        assert data.ndim == 4
        T, H, W, C = data.shape
        assert T == 2
        assert H > 0 and W > 0 and C > 0

        # /data attrs exist
        assert "units" in data.attrs
        assert "missing_value" in data.attrs
        assert "description" in data.attrs

        # schema attrs copied onto geometry/bbox datasets (presence check for keys declared in YAML)
        for ds_name, spec in (prod_schema.get("datasets") or {}).items():
            attrs = (spec or {}).get("attrs") or {}
            if not attrs:
                continue
            h5attrs = f[ds_name].attrs
            for k in attrs.keys():
                assert k in h5attrs, f"Dataset '{ds_name}' missing attr '{k}' (expected from schema)"

        # verify t=1 frame all NaN (since scan was None)
        frame1 = data[1, :, :, :]
        assert np.isnan(frame1[...]).all(), "Expected all-NaN frame at t=1 when scan is None"

        # verify masked values became NaN at t=0 somewhere
        frame0 = data[0, :, :, :]
        assert np.isnan(frame0[...]).any(), "Expected some NaNs from masked array fill at t=0"


def test_save_fake_pseudo_produces_single_channel_and_flags(tmp_path: Path):
    """
    Test that serialization 
    """
    ctx_path, prod_path = _schema_paths()

    scan0 = make_fake_radar_pseudo(field_key="reflectivity")
    df = pd.DataFrame(
        {
            "time": ["2020-05-01T00:00:00Z"],
            "radar_site": ["KTLX"],
            "storm_id": [999],
            "min_lat": [34.9],
            "max_lat": [35.7],
            "min_lon": [-98.2],
            "max_lon": [-96.8],
            "reflectivity_scan": [scan0],
            "reflectivity_matched_volume_s3_key": ["fake/key0"],
        }
    )

    out = save_df_for_training(
        df,
        base_dir=str(tmp_path),
        debug=False,
        drop_scans_after_save=False,
        dataset_version="1.0.0-test",
        context_schema_yaml_path=str(ctx_path),
        product_schema_yaml_path=str(prod_path),
    )
    assert out
    prod_file = Path(out[0]["path"])
    assert prod_file.exists()

    with h5py.File(prod_file, "r") as f:
        data = f["data"]
        assert data.shape[0] == 1  # T
        assert data.shape[-1] == 1  # C = 1 for pseudo
        # pseudo flags
        assert bool(f.attrs["channels_are_sweeps"]) is False
        assert _as_str(f.attrs["composite_method"]) == "MAX_ALL_SWEEPS"
