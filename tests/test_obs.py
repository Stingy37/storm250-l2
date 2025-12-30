from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

import storm250.obs as obs


################################################################## TESTS #######################################################


#       /- network decorator, so we can choose whether to run (or not run)
@pytest.mark.network
def test_load_raw_lsr_smoke_and_cache_short_circuits(tmp_path, monkeypatch):
    """
    1) Fetch a short range from IEM (network).
    2) Verify cache file exists.
    3) Disable network by monkeypatching requests.get to raise, then ensure we load from cache.
    """
    cache_dir = tmp_path / "lsr_cache"

    start = date(2017, 5, 1)
    end = date(2017, 5, 3)

    df1 = obs.load_raw_lsr(
        start=start,
        end=end,
        cache_dir=cache_dir,
        force_refresh=False,
        debug=True,
    )

    # basic shape/columns checks (df may be empty depending on events)
    assert list(df1.columns) == ["time", "lat", "lon", "gust", "type"]

    # cache should exist with YYYYMMDD_YYYYMMDD.csv naming
    expected_cache = cache_dir / f"{start:%Y%m%d}_{end:%Y%m%d}.csv"
    assert expected_cache.exists(), "Expected LSR cache file to be written"

    # Now prove we short-circuit cache by killing the network call.
    def _no_network(*args, **kwargs):
        raise RuntimeError("Network call attempted during cache short-circuit test")

    # requests shouldn't work anymore
    monkeypatch.setattr(obs.requests, "get", _no_network)

    df2 = obs.load_raw_lsr(
        start=start,
        end=end,
        cache_dir=cache_dir,
        force_refresh=False,
        debug=True,
    )

    # make sure we load from cache 
    pd.testing.assert_frame_equal(df1.reset_index(drop=True), df2.reset_index(drop=True), check_dtype=False)


def test_filter_lsr_filters_by_time_and_bbox_and_adds_distance():
    """
    Pure unit test: small synthetic df, ensure time-window + bbox filtering works,
    and required output columns exist.
    """
    scan_time = datetime(2017, 5, 1, 12, 0, 0)
    center_lat, center_lon = 30.0, -95.0

    lsr_df = pd.DataFrame(
        [
            # inside bbox, inside time window
            {"time": scan_time, "lat": 30.01, "lon": -95.01, "gust": 55.0, "type": "tstm wind"},
            # inside time window, but outside bbox
            {"time": scan_time + timedelta(minutes=2), "lat": 40.0, "lon": -120.0, "gust": 60.0, "type": "tstm wind"},
            # inside bbox, but outside time window
            {"time": scan_time + timedelta(hours=2), "lat": 30.02, "lon": -95.02, "gust": 65.0, "type": "tstm wind"},
        ]
    )

    out = obs.filter_lsr(
        lsr_df=lsr_df,
        bounding_lat=(29.9, 30.1),
        bounding_lon=(-95.1, -94.9),
        center_lat=center_lat,
        center_lon=center_lon,
        scan_time=scan_time,
        time_window=timedelta(minutes=5),
        debug=True,
    )

    assert list(out.columns) == ["source", "time", "station_lat", "station_lon", "gust", "obs_distance"]
    assert len(out) == 1
    assert out["source"].iloc[0] == "lsr_iastate"
    assert float(out["obs_distance"].iloc[0]) >= 0.0


def test_load_raw_spc_reads_test_data_file():
    """
    Uses tests/data/2017_wind.csv 
    """
    #           |- Path(__file__).parent gives the parent directory OF test_obs, so tests
    #           |                                          \- so the full data_dir here is "tests/data"
    data_dir = Path(__file__).parent / "data"
    assert (data_dir / "2017_wind.csv").exists(), "Expected tests/data/2017_wind.csv to exist"

    start = datetime(2017, 5, 1)
    end = datetime(2017, 5, 7)

    df1 = obs.load_raw_spc(start=start, end=end, spc_dir=data_dir, debug=False)

    assert list(df1.columns) == ["time", "lat", "lon", "gust", "type"]
    assert (df1["type"] == "spc_wind").all()
    # may be empty depending on file contents, but typically should not be
    assert df1["time"].dtype.kind in ("M", "m")  # datetime64-ish after parsing

    # "cache-like" behavior: re-reading should be deterministic
    df2 = obs.load_raw_spc(start=start, end=end, spc_dir=data_dir, debug=False)
    pd.testing.assert_frame_equal(df1.reset_index(drop=True), df2.reset_index(drop=True), check_dtype=False)

# another pure unit test
def test_filter_spc_filters_by_time_and_bbox_and_adds_distance():
    scan_time = datetime(2017, 5, 1, 12, 0, 0)
    center_lat, center_lon = 30.0, -95.0

    spc_df = pd.DataFrame(
        [
            {"time": scan_time, "lat": 30.01, "lon": -95.01, "gust": 70.0, "type": "spc_wind"},
            {"time": scan_time + timedelta(minutes=2), "lat": 40.0, "lon": -120.0, "gust": 80.0, "type": "spc_wind"},
            {"time": scan_time + timedelta(hours=2), "lat": 30.02, "lon": -95.02, "gust": 90.0, "type": "spc_wind"},
        ]
    )

    out = obs.filter_spc(
        spc_df=spc_df,
        bounding_lat=(29.9, 30.1),
        bounding_lon=(-95.1, -94.9),
        center_lat=center_lat,
        center_lon=center_lon,
        scan_time=scan_time,
        time_window=timedelta(minutes=5),
        debug=True,
    )

    assert list(out.columns) == ["source", "time", "station_lat", "station_lon", "gust", "obs_distance"]
    assert len(out) == 1
    assert out["source"].iloc[0] == "spc"
    assert float(out["obs_distance"].iloc[0]) >= 0.0
