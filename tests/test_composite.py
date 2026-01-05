from __future__ import annotations

import os
import re
from datetime import date
from pathlib import Path

import numpy as np
import numpy.ma as ma
import pandas as pd
import pytest

from storm250.composite import _make_reflectivity_pseudocomposite


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


@pytest.mark.network
def test_make_reflectivity_pseudocomposite_builds_single_sweep(tmp_path: Path):
    pyart = pytest.importorskip("pyart")

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
    s3_url = f"s3://{bucket}/{key}"
    radar = pyart.io.read_nexrad_archive(s3_url)

    # Field name expectation (keep test honest but robust)
    field_name = "reflectivity"
    if field_name not in radar.fields:
        pytest.skip(f"Radar fields did not include '{field_name}'. Available: {sorted(radar.fields.keys())[:30]}...")

    # Run
    pseudo = _make_reflectivity_pseudocomposite(
        radar,
        field_name=field_name,
        out_field=field_name,
        chunk_size=2048,
        debug=False,
        plot_dir=None,
        plot_stub=None,
    )

    # ---- Basic structural invariants ----
    assert pseudo is not None
    assert int(pseudo.nsweeps) == 1

    assert isinstance(getattr(pseudo, "fields", None), dict)
    assert field_name in pseudo.fields
    assert len(pseudo.fields) == 1  # should only carry the composite field

    # Metadata tag used downstream
    md = dict(getattr(pseudo, "metadata", {}) or {})
    assert md.get("pseudo_host_sweep", None) == 0

    # Sweep indices should describe the single sweep
    s0 = int(pseudo.sweep_start_ray_index["data"][0])
    e0 = int(pseudo.sweep_end_ray_index["data"][0])
    assert s0 == 0
    assert e0 > 0

    # Field array shape should match (nrays, ngates)
    out_ma = pseudo.fields[field_name]["data"]
    assert isinstance(out_ma, ma.MaskedArray)
    assert out_ma.dtype == np.float32

    nrays = e0 - s0
    ngates = int(np.asarray(pseudo.range["data"]).size)
    assert out_ma.shape == (nrays, ngates)

    # Gate coordinates should exist and match the field’s geometry
    pseudo.init_gate_longitude_latitude()
    assert hasattr(pseudo, "gate_latitude")
    assert hasattr(pseudo, "gate_longitude")
    glat = pseudo.gate_latitude["data"]
    glon = pseudo.gate_longitude["data"]
    assert glat.shape[0] == nrays and glat.shape[1] == ngates
    assert glon.shape[0] == nrays and glon.shape[1] == ngates

    # Sanity: should contain some valid (unmasked) data
    valid = ~ma.getmaskarray(out_ma) & np.isfinite(out_ma.filled(np.nan))
    assert int(valid.sum()) > 0

    # ------- “Composite-ness” sanity check (weak but meaningful) -------
    # Since the host sweep itself contributes, the pseudo-comp max should be >= the host sweep max.
    # So, we check for that. 
    sw_start = radar.sweep_start_ray_index["data"].astype(int)
    sw_end = radar.sweep_end_ray_index["data"].astype(int)

    rays_per_sweep = (sw_end - sw_start).astype(int)
    host_sweep = int(np.argmax(rays_per_sweep))

    hs = int(sw_start[host_sweep])
    he = int(sw_end[host_sweep])
    host_field = radar.fields[field_name]["data"][hs:he, :]
    if not isinstance(host_field, ma.MaskedArray):
        host_field = ma.MaskedArray(host_field, mask=np.zeros_like(host_field, dtype=bool))

    # perform the comparison
    host_max = float(np.nanmax(host_field.filled(np.nan)))
    pseudo_max = float(np.nanmax(out_ma.filled(np.nan)))
    assert pseudo_max >= host_max - 1e-3
