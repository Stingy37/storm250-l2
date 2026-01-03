from __future__ import annotations

import copy
import os
import re
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import storm250.geo as geo
pyart = pytest.importorskip("pyart")


########################################################## HELPERS ########################################################


def _list_one_lv2_object_and_time(
    *,
    bucket: str,
    site: str,
    candidate_days: list[date],
) -> tuple[str, pd.Timestamp]:
    """
    Easy way to implement real network tests.

    Returns needed info: (s3_key, volume_time_utc)
      - s3_key is bucket-relative: "YYYY/MM/DD/SITE/FNAME"
      - volume_time_utc is pd.Timestamp(tz="UTC")
    """
    s3fs = pytest.importorskip("s3fs")

    fs = s3fs.S3FileSystem(anon=True)

    site = site.upper().strip()
    name_re = re.compile(rf"{re.escape(site)}(\d{{8}})_(\d{{6}})")

    for day in candidate_days:
        prefix = f"{day:%Y}/{day:%m}/{day:%d}/{site}/"
        uri = f"s3://{bucket}/{prefix}"

        try:
            objs = fs.ls(uri, detail=False)
        except Exception:
            objs = []

        # Filter to actual files and parse a timestamp from the filename
        for p in objs:
            if p.endswith("/"):
                continue
            fname = os.path.basename(p)
            mo = name_re.search(fname)
            if not mo:
                continue
            ymd = mo.group(1)
            hms = mo.group(2)
            ts = pd.to_datetime(f"{ymd}{hms}", format="%Y%m%d%H%M%S", utc=True, errors="coerce")
            if pd.isna(ts):
                continue

            key = f"{day:%Y}/{day:%m}/{day:%d}/{site}/{fname}"
            return key, ts

    pytest.skip(f"No Level-II objects found for site={site} in candidate_days={candidate_days}")


def _pick_reflectivity_field(radar) -> str:
    """Pick a reflectivity-like field name from a Py-ART radar object."""
    keys = list(getattr(radar, "fields", {}).keys())
    if not keys:
        pytest.skip("Radar has no fields")

    if "reflectivity" in radar.fields:
        return "reflectivity"

    for k in keys:
        if "reflect" in k.lower():
            return k

    pytest.skip(f"No reflectivity-like field found. Available fields: {keys}")


def _set_pseudo_host_sweep(radar, sweep: int) -> None:
    """
    Set a sweep to be "pseudo_host_sweep" in radar metadata to align with _compute_metric_grid_and_labels.
    """
    md = getattr(radar, "metadata", None)
    if md is None:
        radar.metadata = {}
    elif not isinstance(md, dict):
        radar.metadata = dict(md)
    radar.metadata["pseudo_host_sweep"] = int(sweep)


def _radar_latlon(radar) -> tuple[float, float]:
    """Return (lat, lon_norm) where lon_norm is in [-180, 180)."""
    rlat = float(radar.latitude["data"][0]) if np.ndim(radar.latitude["data"]) else float(radar.latitude["data"])
    rlon = float(radar.longitude["data"][0]) if np.ndim(radar.longitude["data"]) else float(radar.longitude["data"])
    rlon = ((rlon + 180.0) % 360.0) - 180.0
    return rlat, rlon


def _drop_gate_geo_attrs(radar) -> None:
    # If Py-ART has already cached gate_longitude/latitude, delete them so recompute reflects any az edits.
    for k in ("gate_longitude", "gate_latitude", "gate_altitude"):
        try:
            delattr(radar, k)
        except Exception:
            pass


######################################################## UNIT TESTS (NO NETWORK) ########################################################


def test_ray_canonical_roll_simple():
    az = np.array([180.0, 181.0, 182.0, 0.0, 1.0, 2.0])
    roll = geo._ray_canonical_roll(az)
    assert roll == 3  # the '0.0' ray is at index 3


def test_normalize_lons_to_center_cross_dateline():
    # Two longitudes that are "close" across the dateline; normalize them around a center near -180.
    lons = np.array([179.9, -179.9], dtype=float)
    out = geo._normalize_lons_to_center(lons, center_lon=-179.8)
    assert np.all(np.isfinite(out))
    assert float(np.max(out) - np.min(out)) < 1.0


######################################################## NETWORK TESTS ########################################################


# Network fixture: pull one real Level-II file from public S3 and load with Py-ART 
# (same logic as test_nexrad.py, just DRY'ed into a fixture)
@pytest.fixture()
def lv2_radar():
    """
    Fixture for network tests, returns a Py-ART radar object. 
    """
    bucket = "unidata-nexrad-level2"
    site = "KTLX"

    # Try a few dates; skip if none are found (keeps test robust)
    key, vol_time = _list_one_lv2_object_and_time(
        bucket=bucket,
        site=site,
        candidate_days=[
            date(2017, 5, 1),
            date(2017, 6, 1),
            date(2018, 5, 1),
        ],
    )

    # Read one real volume (network)
    s3_uri = f"s3://{bucket}/{key}"
    radar = pyart.io.read_nexrad_archive(s3_uri)
    return radar

# Network tests
@pytest.mark.network
def test_aeqd_transform_orientation_not_mirrored(lv2_radar):
    """
    Sanity check the AEQD meters grid orientation:
      +x should be east (lon increases after normalization),
      +y should be north (lat increases).
    """
    radar = lv2_radar
    _set_pseudo_host_sweep(radar, 0)

    class_field = _pick_reflectivity_field(radar)

    # Clear your plan cache if it exists (avoid cross-test state)
    if hasattr(geo, "_GRID_PLAN_CACHE"):
        try:
            geo._GRID_PLAN_CACHE.clear()
        except Exception:
            pass

    plan = geo._get_or_build_grid_plan(radar, class_field, grid_res_m=1000.0, pad_m=5000.0, debug=False)
    t_xy2geog = plan["t_xy2geog"]

    rlat, rlon = _radar_latlon(radar)

    xs = np.array([+10_000.0, -10_000.0, 0.0, 0.0], dtype=float)
    ys = np.array([0.0, 0.0, +10_000.0, -10_000.0], dtype=float)
    lons, lats = t_xy2geog.transform(xs, ys)

    lons = geo._normalize_lons_to_center(np.asarray(lons, dtype=float), rlon)
    lats = np.asarray(lats, dtype=float)

    assert lons[0] > rlon
    assert lons[1] < rlon
    assert lats[2] > rlat
    assert lats[3] < rlat


@pytest.mark.network
def test_grid_plan_nearest_gate_is_reasonably_close(lv2_radar):
    """
    The grid_plan stores (grid_ray, grid_gate) so every metric grid node picks the nearest valid gate.
    This test checks that the chosen gate is not absurdly far from its grid node (no mirroring/wrapping bugs).
    """
    radar = lv2_radar
    _set_pseudo_host_sweep(radar, 0)
    class_field = _pick_reflectivity_field(radar)

    if hasattr(geo, "_GRID_PLAN_CACHE"):
        try:
            geo._GRID_PLAN_CACHE.clear()
        except Exception:
            pass

    plan = geo._get_or_build_grid_plan(radar, class_field, grid_res_m=1000.0, pad_m=5000.0, debug=False)
    gx = np.asarray(plan["grid_x"], dtype=float)
    gy = np.asarray(plan["grid_y"], dtype=float)
    nx = int(plan["nx"])
    ny = int(plan["ny"])
    t_g2x = plan["t_geog2xy"]

    sweep = int(getattr(radar, "metadata", {}).get("pseudo_host_sweep", 0))
    start = int(radar.sweep_start_ray_index["data"][sweep])
    stop = int(radar.sweep_end_ray_index["data"][sweep])

    az = np.asarray(radar.azimuth["data"][start:stop], dtype=float)
    roll = geo._ray_canonical_roll(az)

    _drop_gate_geo_attrs(radar)
    radar.init_gate_longitude_latitude()
    lats_full = np.asarray(radar.gate_latitude["data"][start:stop, :], dtype=float)
    lons_full = np.asarray(radar.gate_longitude["data"][start:stop, :], dtype=float)

    lats = np.roll(lats_full, -roll, axis=0)
    lons = np.roll(lons_full, -roll, axis=0)

    _, rlon = _radar_latlon(radar)

    total = ny * nx
    n_sample = min(2000, total)
    rng = np.random.default_rng(0)
    sample = rng.choice(total, size=n_sample, replace=False)

    x_idx = sample % nx
    y_idx = sample // nx
    x_node = gx[x_idx]
    y_node = gy[y_idx]

    ray = np.asarray(plan["grid_ray"], dtype=np.int64)[sample]
    gate = np.asarray(plan["grid_gate"], dtype=np.int64)[sample]

    gate_lat = lats[ray, gate]
    gate_lon = geo._normalize_lons_to_center(lons[ray, gate], rlon)

    x_gate, y_gate = t_g2x.transform(gate_lon, gate_lat)
    x_gate = np.asarray(x_gate, dtype=float)
    y_gate = np.asarray(y_gate, dtype=float)

    dist = np.hypot(x_gate - x_node, y_gate - y_node)
    dist = dist[np.isfinite(dist)]
    assert dist.size > 0

    assert float(np.median(dist)) < 2500.0


@pytest.mark.network
def test_metric_grid_invariant_to_ray_start_rotation(lv2_radar):
    """
    If the sweep starts at a different "first ray" but the (az, field) pairing is preserved,
    canonical roll + grid mapping must produce the same metric grid mask.
    """
    radar1 = lv2_radar
    _set_pseudo_host_sweep(radar1, 0)
    class_field = _pick_reflectivity_field(radar1)

    if hasattr(geo, "_GRID_PLAN_CACHE"):
        try:
            geo._GRID_PLAN_CACHE.clear()
        except Exception:
            pass

    info1 = geo._compute_metric_grid_and_labels(
        radar1,
        class_field=class_field,
        threshold=20,
        pad_m=5000.0,
        grid_res_m=1000.0,
        debug=False,
    )
    mask1 = np.asarray(info1["grid_mask"], dtype=bool)

    radar2 = copy.deepcopy(radar1)
    _set_pseudo_host_sweep(radar2, 0)

    sweep = int(getattr(radar2, "metadata", {}).get("pseudo_host_sweep", 0))
    start = int(radar2.sweep_start_ray_index["data"][sweep])
    stop = int(radar2.sweep_end_ray_index["data"][sweep])
    n_rays = stop - start
    if n_rays < 10:
        pytest.skip("Not enough rays in sweep to run a meaningful rotation test")

    shift = 17 % n_rays
    if shift == 0:
        shift = 1

    radar2.azimuth["data"][start:stop] = np.roll(radar2.azimuth["data"][start:stop], shift)
    radar2.elevation["data"][start:stop] = np.roll(radar2.elevation["data"][start:stop], shift)
    radar2.time["data"][start:stop] = np.roll(radar2.time["data"][start:stop], shift)

    fld = radar2.fields[class_field]["data"]
    fld[start:stop, :] = np.roll(fld[start:stop, :], shift, axis=0)

    _drop_gate_geo_attrs(radar2)

    info2 = geo._compute_metric_grid_and_labels(
        radar2,
        class_field=class_field,
        threshold=20,
        pad_m=5000.0,
        grid_res_m=1000.0,
        debug=False,
    )
    mask2 = np.asarray(info2["grid_mask"], dtype=bool)

    assert mask1.shape == mask2.shape
    assert np.array_equal(mask1, mask2), (
        "Metric grid mask changed after a pure ray-start rotation (canonical roll failed)"
    )
