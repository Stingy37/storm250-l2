# tests/test_cropping.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

# IMPORTANT: cropping.py imports pyplot at module import time, so set backend first
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import numpy.ma as ma
import pandas as pd
import pytest

from storm250.cropping import (
    _apply_mask_to_fields_for_sweep_slice,
    _bbox_center,
    _bbox_with_buffer_deg,
    _crop_one_sweep_to_bbox_inplace,
    _crop_pseudocomp_scan_to_bbox,
    _crop_radar_volume_to_bbox,
    _find_blob_for_point,
    build_bboxes_for_linked_df,
)
from storm250.composite import _make_reflectivity_pseudocomposite


############################################ PURE UNIT TESTS (NO NETWORKING) ############################################


def test_bbox_center_handles_dateline_crossing():
    """
    Geometry test. 
    """
    # Crossing the dateline: width computed via +360 wrap
    minlat, maxlat = 10.0, 20.0
    minlon, maxlon = 170.0, -170.0  # crosses dateline

    clat, clon = _bbox_center(minlat, maxlat, minlon, maxlon)

    assert clat == pytest.approx(15.0)
    # 170 -> -170 spans 20 deg; center should be 180 => normalized to -180
    assert clon == pytest.approx(-180.0)


def test_bbox_with_buffer_deg_expands_extents():
    """
    Make sure that the buffer results in a bigger area compared to the bbox without the buffer.
    """
    minlat, maxlat, minlon, maxlon = 30.0, 31.0, -100.0, -99.0
    lat_min2, lat_max2, lon_min2, lon_max2 = _bbox_with_buffer_deg(minlat, maxlat, minlon, maxlon, buffer_km=10.0)

    assert lat_min2 < minlat
    assert lat_max2 > maxlat
    assert lon_min2 < minlon
    assert lon_max2 > maxlon


@dataclass
class _DummyRadar:
    """
    Mirrors parts of Py-ART radar object that are crucial to our tests.
    """
    fields: dict[str, dict[str, Any]]
    sweep_start_ray_index: dict[str, np.ndarray]
    sweep_end_ray_index: dict[str, np.ndarray]


def test_apply_mask_to_fields_for_sweep_slice_nan_fill_and_mask_merge():
    """
    Test that a given masked array is applied correctly to sweeps.
    """
    # Full radar: 6 rays, 4 gates, sweep slice is rays [2:5] => 3x4
    nrays, ngates = 6, 4
    s, e = 2, 5
    mask_outside = np.zeros((e - s, ngates), dtype=bool)
    mask_outside[0, 1] = True
    mask_outside[2, 3] = True

    # Field A: plain float array -> with prefer_nan_fill=True, so should write NaNs in slice only
    a = np.zeros((nrays, ngates), dtype=np.float64)

    # Field B: MaskedArray -> should OR masks in slice only
    b_data = np.ones((nrays, ngates), dtype=np.float32)
    b_mask = np.zeros((nrays, ngates), dtype=bool)
    b_mask[1, 1] = True  # existing mask outside slice should persist
    b = ma.MaskedArray(b_data, mask=b_mask)

    radar = _DummyRadar(
        fields={
            "a": {"data": a},
            "b": {"data": b},
        },
        sweep_start_ray_index={"data": np.array([s], dtype=int)},
        sweep_end_ray_index={"data": np.array([e], dtype=int)},
    )

    _apply_mask_to_fields_for_sweep_slice(
        radar,
        s=s,
        e=e,
        mask_outside=mask_outside,
        prefer_nan_fill=True,
        shared_bcast={},
    )

    out_a = radar.fields["a"]["data"]
    assert isinstance(out_a, np.ndarray)
    assert out_a.dtype == np.float32  # normalized
    assert np.isfinite(out_a[:s, :]).all()
    assert np.isfinite(out_a[e:, :]).all()
    assert np.isnan(out_a[s + 0, 1])
    assert np.isnan(out_a[s + 2, 3])

    out_b = radar.fields["b"]["data"]
    assert isinstance(out_b, ma.MaskedArray)
    # existing mask kept
    assert bool(out_b.mask[1, 1]) is True
    # new masks added in slice
    assert bool(out_b.mask[s + 0, 1]) is True
    assert bool(out_b.mask[s + 2, 3]) is True
    # outside slice should remain unmasked (except existing mask)
    assert bool(out_b.mask[0, 0]) is False
    assert bool(out_b.mask[5, 0]) is False


def test_crop_one_sweep_noop_when_inside_mask_all_true(monkeypatch):
    import storm250.cropping as cropping

    nrays, ngates = 3, 4
    data = ma.MaskedArray(np.ones((nrays, ngates), dtype=np.float32), mask=np.zeros((nrays, ngates), dtype=bool))

    radar = _DummyRadar(
        fields={"reflectivity": {"data": data}},
        sweep_start_ray_index={"data": np.array([0], dtype=int)},
        sweep_end_ray_index={"data": np.array([nrays], dtype=int)},
    )

    # Patch inside mask to all-True => no-op
    monkeypatch.setattr(
        cropping,
        "_fast_metric_inside_mask",
        lambda *args, **kwargs: np.ones((nrays, ngates), dtype=bool),
    )

    before_mask = radar.fields["reflectivity"]["data"].mask.copy()
    changed = _crop_one_sweep_to_bbox_inplace(
        radar,
        sweep=0,
        lat_min=0,
        lat_max=1,
        lon_min=0,
        lon_max=1,
        buffer_km=0.0,
        debug=False,
        prefer_nan_fill=True,
    )

    assert changed is False
    after_mask = radar.fields["reflectivity"]["data"].mask
    assert np.array_equal(before_mask, after_mask)


def test_crop_one_sweep_masks_outside(monkeypatch):
    import storm250.cropping as cropping

    nrays, ngates = 3, 4
    data = ma.MaskedArray(np.arange(nrays * ngates, dtype=np.float32).reshape(nrays, ngates), mask=False)

    radar = _DummyRadar(
        fields={"reflectivity": {"data": data}},
        sweep_start_ray_index={"data": np.array([0], dtype=int)},
        sweep_end_ray_index={"data": np.array([nrays], dtype=int)},
    )

    # Inside mask has some False => should mask those outside
    inside = np.ones((nrays, ngates), dtype=bool)
    inside[0, 0] = False
    inside[2, 3] = False

    monkeypatch.setattr(cropping, "_fast_metric_inside_mask", lambda *args, **kwargs: inside.copy())

    changed = _crop_one_sweep_to_bbox_inplace(
        radar,
        sweep=0,
        lat_min=0,
        lat_max=1,
        lon_min=0,
        lon_max=1,
        buffer_km=0.0,
        debug=False,
        prefer_nan_fill=False,  # force masking behavior even if plain ndarray shows up
    )

    assert changed is True
    out = radar.fields["reflectivity"]["data"]
    assert isinstance(out, ma.MaskedArray)
    assert bool(out.mask[0, 0]) is True
    assert bool(out.mask[2, 3]) is True
    assert bool(out.mask[0, 1]) is False


############################### NEXRAD TESTS WITH REAL NEXRAD DATA ###############################


_BUCKET = "unidata-nexrad-level2"

# storm volumes that are known to contain storms that meet cropping / bbox thresholds 
_KDLH_ROWS = [
    #                  s3_key               storm_lat | storm_lon
    ("2017/06/14/KDLH/KDLH20170614_044958_V06", 47.86, -91.55),
    ("2017/06/14/KDLH/KDLH20170614_045440_V06", 47.89, -91.53),
]


def _masked_fraction(arr: Any) -> float:
    """
    Helper: return fraction of elements that are outside the cropped area (masked or NaN).
    """
    if isinstance(arr, ma.MaskedArray):
        m = ma.getmaskarray(arr)
        return float(m.mean()) if m.size else 0.0
    arr = np.asarray(arr)
    if np.issubdtype(arr.dtype, np.floating):
        return float(np.isnan(arr).mean()) if arr.size else 0.0
    return 0.0


@pytest.mark.network
def test_find_blob_for_point_on_known_storm_volume():
    """
    Test if a blob can be found for a single radar volume, given storm lat / lons. 
    """
    pyart = pytest.importorskip("pyart")
    pytest.importorskip("s3fs")
    pytest.importorskip("scipy")

    key, clat, clon = _KDLH_ROWS[0]
    radar = pyart.io.read_nexrad_archive(f"s3://{_BUCKET}/{key}")

    if "reflectivity" not in radar.fields:
        pytest.skip("reflectivity not present in this volume")

    pseudo = _make_reflectivity_pseudocomposite(
        radar,
        field_name="reflectivity",
        out_field="reflectivity",
        chunk_size=2048,
        debug=False,
    )

    # Use forgiving params so the test is robust
    minlat, maxlat, minlon, maxlon, centers, info = _find_blob_for_point(
        pseudo,
        clat,
        clon,
        class_field="reflectivity",
        threshold=15,
        min_size=10,
        pad_m=10_000.0,
        grid_res_m=2000.0,
        include_nearby_km=10.0,
        debug=False,
    )

    assert minlat is not None and maxlat is not None and minlon is not None and maxlon is not None
    assert isinstance(centers, list) and len(centers) >= 1
    assert isinstance(info, dict)
    assert minlat < maxlat


@pytest.mark.network
def test_crop_pseudocomp_scan_to_bbox_masks_outside():
    pyart = pytest.importorskip("pyart")
    pytest.importorskip("s3fs")
    pytest.importorskip("scipy")

    key, clat, clon = _KDLH_ROWS[0]
    radar = pyart.io.read_nexrad_archive(f"s3://{_BUCKET}/{key}")

    if "reflectivity" not in radar.fields:
        pytest.skip("reflectivity not present in this volume")

    pseudo = _make_reflectivity_pseudocomposite(
        radar,
        field_name="reflectivity",
        out_field="reflectivity",
        chunk_size=2048,
        debug=False,
    )

    # Derive bbox from the pseudo-comp itself (no reliance on old stored bbox params)
    minlat, maxlat, minlon, maxlon, centers, info = _find_blob_for_point(
        pseudo,
        clat,
        clon,
        class_field="reflectivity",
        threshold=15,
        min_size=10,
        pad_m=10_000.0,
        grid_res_m=2000.0,
        include_nearby_km=10.0,
        debug=False,
    )
    if minlat is None:
        pytest.skip("Could not find blob bbox on this volume with chosen parameters")

    before = pseudo.fields["reflectivity"]["data"]
    before_outside = _masked_fraction(before)

    cropped = _crop_pseudocomp_scan_to_bbox(
        pseudo,
        minlat,
        maxlat,
        minlon,
        maxlon,
        buffer_km=0.0,           # encourage some masking
        debug=False,
        inplace=True,
        drop_gate_coords=True,
        prefer_nan_fill=True,
    )

    assert cropped is pseudo  # in-place
    after = cropped.fields["reflectivity"]["data"]
    after_outside = _masked_fraction(after)

    # We expect some outside masking, and also some inside data remains.
    assert after_outside > before_outside
    assert after_outside < 0.999

    # Gate coords should be dropped if requested
    assert "gate_longitude" not in cropped.__dict__
    assert "gate_latitude" not in cropped.__dict__


@pytest.mark.network
def test_crop_radar_volume_to_bbox_masks_some_sweeps():
    pyart = pytest.importorskip("pyart")
    pytest.importorskip("s3fs")
    pytest.importorskip("scipy")

    key, clat, clon = _KDLH_ROWS[0]
    radar = pyart.io.read_nexrad_archive(f"s3://{_BUCKET}/{key}")

    if "reflectivity" not in radar.fields:
        pytest.skip("reflectivity not present in this volume")

    # Use pseudo-comp only to estimate bbox; then crop full volume
    pseudo = _make_reflectivity_pseudocomposite(radar, field_name="reflectivity", out_field="reflectivity", debug=False)

    minlat, maxlat, minlon, maxlon, centers, info = _find_blob_for_point(
        pseudo,
        clat,
        clon,
        class_field="reflectivity",
        threshold=15,
        min_size=10,
        pad_m=10_000.0,
        grid_res_m=2000.0,
        include_nearby_km=10.0,
        debug=False,
    )
    if minlat is None:
        pytest.skip("Could not find blob bbox on this volume with chosen parameters")

    # Measure "outside-ness" on reflectivity before cropping (whole field)
    before = radar.fields["reflectivity"]["data"]
    before_outside = _masked_fraction(before)

    cropped = _crop_radar_volume_to_bbox(
        radar,
        minlat,
        maxlat,
        minlon,
        maxlon,
        buffer_km=0.0,
        debug=False,
        inplace=True,
        drop_gate_coords=True,
        prefer_nan_fill=True,
        sweeps=[0, 1, 2],  # keep the test bounded
    )

    assert cropped is radar
    after = cropped.fields["reflectivity"]["data"]
    after_outside = _masked_fraction(after)

    assert after_outside >= before_outside
    assert after_outside < 0.999

    assert "gate_longitude" not in cropped.__dict__
    assert "gate_latitude" not in cropped.__dict__


@pytest.mark.network
def test_build_bboxes_for_linked_df_integration_single_row():
    pyart = pytest.importorskip("pyart")
    pytest.importorskip("s3fs")
    pytest.importorskip("scipy")

    key, clat, clon = _KDLH_ROWS[0]
    radar = pyart.io.read_nexrad_archive(f"s3://{_BUCKET}/{key}")

    if "reflectivity" not in radar.fields:
        pytest.skip("reflectivity not present in this volume")

    # Minimal “linked_df” row for your new product-key based pipeline:
    linked_df = pd.DataFrame(
        [
            {
                "time": pd.Timestamp("2017-06-14T04:50:00Z"),
                "latitude": float(clat),
                "longitude": float(clon),
                "storm_id": 2898,
                "radar_site": "KDLH",
                "reflectivity_matched_member_name": key,
                # product key (ends with _scan) -> reference_key becomes "dhr"
                "dhr_scan": radar,
            }
        ]
    )

    out_df = build_bboxes_for_linked_df(
        linked_df,
        class_field="reflectivity",
        threshold=15,
        min_size=10,
        pad_km=10.0,
        grid_res_m=20_000.0,  # makes the later search step much cheaper (larger step_km)
        buffer_km=0.0,
        include_nearby_km=10.0,
        debug=False,
    )

    assert isinstance(out_df, pd.DataFrame)
    assert len(out_df) >= 1

    row0 = out_df.iloc[0].to_dict()
    for k in ("min_lat", "max_lat", "min_lon", "max_lon"):
        assert k in row0
        assert np.isfinite(float(row0[k]))

    assert float(row0["min_lat"]) < float(row0["max_lat"])

    # Cropped radar should be present back in dhr_scan
    cropped_radar = row0.get("dhr_scan", None)
    assert cropped_radar is not None
    assert "reflectivity" in cropped_radar.fields

    # Pseudo-composite product should also exist (may be None if pseudo failed, but for Level II it should work)
    pseudo = row0.get("reflectivity_composite_scan", None)
    assert pseudo is not None
    assert int(pseudo.nsweeps) == 1
    assert "reflectivity" in pseudo.fields

    # Both should show evidence of cropping (some outside-ness but not all)
    frac_full = _masked_fraction(cropped_radar.fields["reflectivity"]["data"])
    frac_pseudo = _masked_fraction(pseudo.fields["reflectivity"]["data"])
    assert frac_full < 0.999
    assert frac_pseudo < 0.999
