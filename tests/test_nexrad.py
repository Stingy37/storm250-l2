# tests/test_nexrad.py
from __future__ import annotations

import os
import re
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from storm250.nexrad import (
    _make_radar_skeleton,
    find_radar_scans,
)
from storm250.io import (
    _load_field_pack,
    _load_gz_pickle,
)


############################################## HELPERS ##############################################


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


############################################### tests ##############################################


@pytest.mark.network
def test_make_radar_skeleton_is_pyart_radar_and_strips_fields(tmp_path: Path):
    """
    Test that the created radar skeleton is still a Py-ART radar object
    with no fields and only geometry metadata. 
    """
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
    s3_uri = f"s3://{bucket}/{key}"
    radar = pyart.io.read_nexrad_archive(s3_uri)

    # Create the skeleton
    sk = _make_radar_skeleton(radar)

    ################## test skeleton properties ##################
    
    # Should still be a Py-ART Radar object
    assert sk is not None
    assert sk.__class__.__name__ == radar.__class__.__name__

    # Skeleton should have NO fields
    assert isinstance(sk.fields, dict)
    assert len(sk.fields) == 0

    # And should preserve core geometry pieces
    assert "data" in sk.time and len(sk.time["data"]) == len(radar.time["data"])
    assert "data" in sk.range and len(sk.range["data"]) == len(radar.range["data"])
    assert "data" in sk.azimuth and len(sk.azimuth["data"]) == len(radar.azimuth["data"])
    assert "data" in sk.elevation and len(sk.elevation["data"]) == len(radar.elevation["data"])


@pytest.mark.network
def test_find_radar_scans_writes_cache_and_reconstructs_from_cache(tmp_path: Path):
    """
    Test that cache for skeleton + field packs is written. 
    """
    pytest.importorskip("pyart")
    pytest.importorskip("s3fs")

    bucket = "unidata-nexrad-level2"
    site = "KTLX"

    key, vol_time = _list_one_lv2_object_and_time(
        bucket=bucket,
        site=site,
        candidate_days=[
            date(2017, 5, 1),
            date(2017, 6, 1),
            date(2018, 5, 1),
        ],
    )

    # Minimal storm_df: one row at the exact volume timestamp
    storm_df = pd.DataFrame(
        {
            "storm_id": [1],
            "radar_site": [site],
            "time": [vol_time],  # already UTC timestamp
        }
    )

    cache_dir = tmp_path / "level2_cache"

    # build the cache through first-time call of find_radar_scans
    df = find_radar_scans(
        storm_df=storm_df,
        site_column="radar_site",
        time_column="time",
        level2_base=bucket,
        cache_dir=str(cache_dir),
        product_filter=["reflectivity"],
        time_tolerance_seconds=1,
        keep_in_memory=True,
        debug=True,
    )
    assert len(df) == 1

    # Product-key is "reflectivity," so check that field-specific columns exist
    assert "reflectivity_scan" in df.columns
    assert "reflectivity_cache_volume_path" in df.columns
    assert "reflectivity_matched_volume_s3_key" in df.columns

    #               /- keep_in_memory = true, so radar_obj should be loaded in memory
    radar_obj = df.loc[0, "reflectivity_scan"]
    assert radar_obj is not None 

    # check that the skeleton cache is saved correctly
    cache_path = Path(df.loc[0, "reflectivity_cache_volume_path"])
    assert cache_path.name.endswith(".skeleton.pkl.gz")
    assert cache_path.exists()

    # check that the field caches (npz + json for each field) are saved correctly
    # Field pack paths are base_dir + ".reflectivity".npz/json
    base_dir = str(cache_path).replace(".skeleton.pkl.gz", "")
    fld_base = Path(base_dir + ".reflectivity")
    assert fld_base.with_suffix(fld_base.suffix + ".npz").exists()
    assert fld_base.with_suffix(fld_base.suffix + ".json").exists()

    # Rebuild radar manually from cache 
    sk = _load_gz_pickle(cache_path, debug=True)
    assert sk is not None
    assert hasattr(sk, "fields")
    assert len(sk.fields) == 0  # skeleton should be empty

    pack = _load_field_pack(fld_base, debug=True)
    assert pack is not None
    assert "data" in pack

    # now add the field caches to the skeleton Py-ART radar object
    sk.fields["reflectivity"] = pack
    assert "reflectivity" in sk.fields

    # Ensure downcast happened (float32 data in pack)
    data = sk.fields["reflectivity"]["data"]
    # masked array OR ndarray
    arr = data.data if hasattr(data, "data") else data
    assert str(arr.dtype) == "float32"


@pytest.mark.network
def test_find_radar_scans_second_call_uses_existing_cache_no_rewrite(tmp_path: Path):
    """
    Test that cache is reused on second call. 
    """
    pytest.importorskip("pyart")
    pytest.importorskip("s3fs")

    bucket = "unidata-nexrad-level2"
    site = "KTLX"

    key, vol_time = _list_one_lv2_object_and_time(
        bucket=bucket,
        site=site,
        candidate_days=[
            date(2017, 5, 1),
            date(2017, 6, 1),
            date(2018, 5, 1),
        ],
    )

    storm_df = pd.DataFrame({"storm_id": [1], "radar_site": [site], "time": [vol_time]})
    cache_dir = tmp_path / "level2_cache"

    df1 = find_radar_scans(
        storm_df=storm_df,
        level2_base=bucket,
        cache_dir=str(cache_dir),
        product_filter=["reflectivity"],
        time_tolerance_seconds=1,
        keep_in_memory=True,
        debug=True,
    )
    assert len(df1) == 1

    cache_path = Path(df1.loc[0, "reflectivity_cache_volume_path"])
    base_dir = str(cache_path).replace(".skeleton.pkl.gz", "")
    fld_base = Path(base_dir + ".reflectivity")

    sk_path = cache_path
    npz_path = fld_base.with_suffix(fld_base.suffix + ".npz")
    json_path = fld_base.with_suffix(fld_base.suffix + ".json")

    assert sk_path.exists() and npz_path.exists() and json_path.exists()

    mt_sk_1 = sk_path.stat().st_mtime
    mt_npz_1 = npz_path.stat().st_mtime
    mt_json_1 = json_path.stat().st_mtime

    # Second call should be a cache hit (still needs S3 listing, but should not re-read/rewrite volume)
    df2 = find_radar_scans(
        storm_df=storm_df,
        level2_base=bucket,
        cache_dir=str(cache_dir),
        product_filter=["reflectivity"],
        time_tolerance_seconds=1,
        keep_in_memory=True,
        debug=True,
    )
    assert len(df2) == 1
    assert Path(df2.loc[0, "reflectivity_cache_volume_path"]) == cache_path

    # Should not rewrite cached artifacts
    assert sk_path.stat().st_mtime == mt_sk_1
    assert npz_path.stat().st_mtime == mt_npz_1
    assert json_path.stat().st_mtime == mt_json_1
