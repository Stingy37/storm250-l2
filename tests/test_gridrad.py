from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import pytest
import yaml

from storm250.gridrad import _prepare_site_arrays, _active_mask_for_chunk, load_grs_tracks


####################################################### Fixtures / helpers #######################################################


#                  /- scope="function" -> every test that asks for radar_info_subset gets a new value from radar_info_subset()... i.e. "called" for every test
@pytest.fixture(scope="function") # we aren't mutating radar_info_subset though, so technically its okay if we have scope="module" or scope="session"
def radar_info_subset() -> Dict[str, Any]: #                                                                     |- shared across file     |- shared across pytest session
    # subset of full radar_info for easier testing 
    return {
        "KABR": {
            "position": {"lat": 45.455833, "lon": -98.413333},
            "active_ranges": [{"start": None, "end": None}],
        },
        "KABX": {
            "position": {"lat": 35.149722, "lon": -106.82388},
            "active_ranges": [{"start": None, "end": None}],
        },
        "KAKQ": {
            "position": {"lat": 36.98405, "lon": -77.007361},
            "active_ranges": [{"start": None, "end": None}],
        },
        "KAMA": {
            "position": {"lat": 35.233333, "lon": -101.70927},
            "active_ranges": [{"start": None, "end": None}],
        },
        "KAMX": {
            "position": {"lat": 25.611083, "lon": -80.412667},
            "active_ranges": [{"start": None, "end": None}],
        },
        "KAPX": {
            "position": {"lat": 44.90635, "lon": -84.719533},
            "active_ranges": [{"start": None, "end": None}],
        },
        "KARX": {
            "position": {"lat": 43.822778, "lon": -91.191111},
            "active_ranges": [{"start": None, "end": None}],
        },
        "KATX": {
            "position": {"lat": 48.194611, "lon": -122.49569},
            "active_ranges": [{"start": None, "end": None}],
        },
        "KBBX": {
            "position": {"lat": 39.495639, "lon": -121.63161},
            "active_ranges": [{"start": None, "end": None}],
        },
        "KBGM": {
            "position": {"lat": 42.199694, "lon": -75.984722},
            "active_ranges": [{"start": None, "end": None}],
        },
        "KBHX": {
            "position": {"lat": 40.498583, "lon": -124.29216},
            "active_ranges": [{"start": None, "end": None}],
        },
        "KBIS": {
            "position": {"lat": 46.770833, "lon": -100.76055},
            "active_ranges": [{"start": None, "end": None}],
        },
        "KBLX": {
            "position": {"lat": 45.853778, "lon": -108.6068},
            "active_ranges": [{"start": None, "end": None}],
        },
        "KBMX": {
            "position": {"lat": 33.172417, "lon": -86.770167},
            "active_ranges": [{"start": None, "end": None}],
        },
        "KBOX": {
            "position": {"lat": 41.955778, "lon": -71.136861},
            "active_ranges": [{"start": None, "end": None}],
        },
        "KBRO": {
            "position": {"lat": 25.916, "lon": -97.418967},
            "active_ranges": [{"start": None, "end": None}],
        },
        "KBUF": {
            "position": {"lat": 42.948789, "lon": -78.736781},
            "active_ranges": [{"start": None, "end": None}],
        },
        "KBYX": {
            "position": {"lat": 24.5975, "lon": -81.703167},
            "active_ranges": [{"start": None, "end": None}],
        },
    }


def _date_list(start: datetime, end: datetime) -> List[datetime]:
    d = start
    out: List[datetime] = []
    while d <= end:
        out.append(d)
        d += timedelta(days=1)
    return out


def _load_full_radar_info() -> Dict[str, Any] | None:
    # load full configs/radar_info.yaml for gr-s tracks test
    repo_root = Path(__file__).resolve().parents[1]
    candidates = [
        repo_root / "configs" / "radar_info.yaml",
    ]
    for p in candidates:
        if p.exists():
            if p.suffix in (".yaml", ".yml"):
                # deserializes the .yaml file into a python dictionary 
                return yaml.safe_load(p.read_text(encoding="utf-8"))
    return None


####################################################### unit tests helpers #######################################################


#                                                 /- because of the fixture decorator, pytest under-the-hood returns value of radar_info_subset()
def test_prepare_site_arrays_open_intervals(radar_info_subset): #       |- equivalent to calling radar_info_subset = radar_info_subset()
    names, lats, lons, ranges = _prepare_site_arrays(radar_info_subset)

    assert len(names) == len(lats) == len(lons) == len(ranges)
    assert len(names) > 0
    assert lats.dtype == np.float32
    assert lons.dtype == np.float32

    int_min = np.iinfo(np.int64).min
    int_max = np.iinfo(np.int64).max

    # all subset entries have open intervals -> should map to [int_min, int_max]
    for rs in ranges:
        assert rs.shape[1] == 2
        assert rs.dtype == np.int64
        assert np.all(rs[:, 0] == int_min)
        assert np.all(rs[:, 1] == int_max)

#                                                         /- fixture -> we know this test depends on radar_info_subset... which returns a new version FOR this test
def test_prepare_site_arrays_skips_missing_positions(radar_info_subset):
    bad = dict(radar_info_subset)
    bad["KBAD"] = {"position": {"lat": None, "lon": None}, "active_ranges": [{"start": None, "end": None}]}

    names, lats, lons, ranges = _prepare_site_arrays(bad)

    assert "KBAD" not in names
    assert len(names) == len(lats) == len(lons) == len(ranges)


def test_active_mask_for_chunk_respects_ranges():
    # Two sites with different active windows
    radar_info = {
        "A": {"position": {"lat": 0.0, "lon": 0.0}, "active_ranges": [{"start": "2017-01-01", "end": "2017-01-10"}]},
        "B": {"position": {"lat": 0.0, "lon": 1.0}, "active_ranges": [{"start": "2017-01-05", "end": "2017-01-06"}]},
    }
    names, lats, lons, ranges = _prepare_site_arrays(radar_info)

    # times: Jan 04, Jan 05, Jan 07 (UTC)
    t = pd.to_datetime(["2017-01-04", "2017-01-05", "2017-01-07"], utc=True).view("int64").view("int64")
    mask = _active_mask_for_chunk(t, ranges)

    # shape [L,S]
    assert mask.shape == (3, 2)

    # Site A active all 3 (within Jan1-Jan10)
    assert mask[:, 0].tolist() == [True, True, True]
    # Site B active only on Jan5
    assert mask[:, 1].tolist() == [False, True, False]


################################## full integration tests for load_grs_tracks (real downloads) #######################################################


@pytest.mark.network
def test_load_grs_tracks_two_months_smoke(tmp_path, radar_info_subset):
    """
    test that load_grs_tracks work
    """
    # prefer full radar_info, if radar_info.yaml is present (it should be)
    radar_info = _load_full_radar_info() or radar_info_subset

    # temporary paths for testing 
    save_dir = tmp_path / "raw_grs"
    proc_dir = tmp_path / "processed_grs"

    # only do ~ 2 months
    dates = _date_list(datetime(2017, 5, 1), datetime(2017, 6, 30))

    df = load_grs_tracks(
        year=2017,
        radar_info=radar_info,
        dates=dates,
        save_dir=save_dir,
        processed_cache_dir=proc_dir,
        max_distance_km=250.0,
        min_rows=1,
        max_workers=8,
        chunk_size=100_000,
        verify_ssl=False,
        debug=True,
    )

    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0

    # core columns should exist 
    for col in ["time", "latitude", "longitude", "storm_id", "radar_site", "distance_to_site"]:
        assert col in df.columns

    # processed cache should exist (parquet preferred, csv fallback)
    any_cache = list(proc_dir.glob("2017*.parquet")) + list(proc_dir.glob("2017*.csv"))
    assert len(any_cache) >= 1


@pytest.mark.network
def test_load_grs_tracks_processed_cache_short_circuits(tmp_path, radar_info_subset):
    """
    test that cache behavior works
    """
    radar_info = _load_full_radar_info() or radar_info_subset

    save_dir = tmp_path / "raw_grs"
    proc_dir = tmp_path / "processed_grs"

    # Shorter slice here to keep runtime down while still testing cache behavior
    dates = _date_list(datetime(2017, 5, 1), datetime(2017, 5, 7))

    df1 = load_grs_tracks(
        year=2017,
        radar_info=radar_info,
        dates=dates,
        save_dir=save_dir,
        processed_cache_dir=proc_dir,
        max_distance_km=250,
        min_rows=1,
        max_workers=8,
        verify_ssl=False,
        debug=False,
    )

    # Ensure some processed cache was written
    caches = list(proc_dir.glob("2017*.parquet")) + list(proc_dir.glob("2017*.csv"))
    assert caches, "Expected processed cache output to exist"

    # Nuke raw cache to prove we don't need it on second run (relying instead on cache)
    if save_dir.exists():
        for p in save_dir.glob("*"):
            p.unlink()

    # Second call: give an invalid base_url 
    # If the function doesn't use cache, this should fail.
    df2 = load_grs_tracks(
        year=2017,
        radar_info=radar_info,
        dates=dates,
        save_dir=save_dir,
        processed_cache_dir=proc_dir,
        base_url="https://example.invalid/does-not-exist",
        max_distance_km=250,
        min_rows=1,
        max_workers=8,
        verify_ssl=False,
        debug=True,
    )

    # should be equal, since both dataframes should be the same 
    assert len(df2) == len(df1)
    assert set(df2.columns) == set(df1.columns)

    # Compare deterministically after sorting (avoids any row-order differences)
    key_cols = [c for c in ["storm_id", "time", "latitude", "longitude"] if c in df1.columns]
    s1 = df1.sort_values(key_cols).reset_index(drop=True)
    s2 = df2.sort_values(key_cols).reset_index(drop=True)

    pd.testing.assert_frame_equal(s1, s2, check_dtype=False)
