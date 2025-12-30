from __future__ import annotations

import logging
import tempfile
from datetime import date, datetime, time, timedelta
from io import StringIO
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import requests

from .utils import haversine

logger = logging.getLogger(__name__)


######################################################################## FILTER LSR DATA ###############################################################################


def load_raw_lsr(
    start: Union[date, datetime],
    end: Union[date, datetime],
    debug: bool = False,
    cache_dir: Union[str, Path] = "Datasets/surface_obs_datasets/lsr_reports",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Pull raw Local Storm Reports (LSRs) from the IEM archive CSV service
    for each calendar-day between `start` and `end` (inclusive).

    Caching:
      - a cache CSV is saved as {cache_dir}/YYYYMMDD_YYYYMMDD.csv
      - if that file exists and force_refresh is False, it will be loaded and returned
      - corrupted cache files are removed and the fetch is retried

    Returns a DataFrame with columns:
      - time (UTC datetime of the exact report)
      - lat, lon (location of report)
      - gust (the reported magnitude, as float)
      - type (the original TYPETEXT)
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Coerce start/end to datetimes (treat date inputs as full-day ranges)
    if isinstance(start, date) and not isinstance(start, datetime):
        start_dt = datetime.combine(start, time.min)
    else:
        start_dt = start

    if isinstance(end, date) and not isinstance(end, datetime):
        end_dt = datetime.combine(end, time(23, 59, 59))
    else:
        end_dt = end

    cache_fname = cache_dir / f"{start_dt:%Y%m%d}_{end_dt:%Y%m%d}.csv"

    # If present and not forcing refresh, try to load cache
    if cache_fname.exists() and not force_refresh:
        if debug:
            logger.info("Loading LSRs from cache: %s", cache_fname)
        try:
            cached = pd.read_csv(cache_fname, parse_dates=["time"], engine="python")
            cols = ["time", "lat", "lon", "gust", "type"]
            missing = [c for c in cols if c not in cached.columns]
            if missing:
                raise ValueError(f"Cached file missing columns: {missing}")
            return cached[cols]
        except Exception:
            if debug:
                logger.exception("Failed to read cache (%s). Removing and refetching.", cache_fname)
            try:
                cache_fname.unlink()
            except Exception:
                pass

    # Fetch from IEM if cache not used
    base_url = "https://mesonet.agron.iastate.edu/cgi-bin/request/gis/lsr.py"

    # Format start/end as ISO8601 with trailing Z (include +1 sec to make end inclusive)
    sts = start_dt.strftime("%Y-%m-%dT%H:%MZ")
    ets = (end_dt + timedelta(seconds=1)).strftime("%Y-%m-%dT%H:%MZ")

    params = {"wfo": "ALL", "sts": sts, "ets": ets, "fmt": "csv"}

    if debug:
        logger.info("Fetching LSRs from %s to %s with params=%s", sts, ets, params)

    resp = requests.get(base_url, params=params, timeout=30.0)
    resp.raise_for_status()

    if debug:
        logger.info("Fetch HTTP %s | Response snippet: %s", resp.status_code, resp.text[:200].replace("\n", " "))

    df = pd.read_csv(
        StringIO(resp.text),
        parse_dates=["VALID"],
        engine="python",
        on_bad_lines="skip",
    )

    if debug:
        logger.info("%d total reports before filtering", len(df))

    if "TYPETEXT" not in df.columns:
        raise KeyError(f"Expected column 'TYPETEXT' missing. Columns={list(df.columns)}")

    keep_codes = [
        "tstm wind",
        "tstm wnd gst",
        "non-tstm wnd gst",
        "marine tstm wind",
    ]

    if debug:
        vc = df["TYPETEXT"].astype(str).str.lower().value_counts().head(20).to_dict()
        logger.info("Top TYPETEXT counts (lowercased, top 20): %s", vc)

    df = df[df["TYPETEXT"].astype(str).str.lower().isin(keep_codes)].reset_index(drop=True)

    if debug:
        logger.info("%d total reports remaining after TYPETEXT filtering", len(df))

    df = df.rename(
        columns={
            "VALID": "time",
            "LAT": "lat",
            "LON": "lon",
            "MAG": "gust",
            "REMARK": "remarks",
            "TYPETEXT": "type",
        }
    )

    # coerce gust to numeric then drop NaNs in critical fields
    df["gust"] = pd.to_numeric(df["gust"], errors="coerce")
    df = df.dropna(subset=["time", "lat", "lon", "gust"]).reset_index(drop=True)

    if debug:
        logger.info("%d total reports after dropping NaNs in critical fields", len(df))

    out = df[["time", "lat", "lon", "gust", "type"]].copy()

    # Save to cache atomically
    try:
        with tempfile.NamedTemporaryFile(delete=False, dir=str(cache_dir), suffix=".csv") as tmp:
            tmp_name = Path(tmp.name)
        out.to_csv(tmp_name, index=False)
        tmp_name.replace(cache_fname)
        if debug:
            logger.info("Saved LSR cache to %s", cache_fname)
    except Exception:
        if debug:
            logger.exception("Warning: failed to write LSR cache to %s", cache_fname)
    return out


def filter_lsr(
    lsr_df,
    bounding_lat,
    bounding_lon,
    center_lat,
    center_lon,
    scan_time,
    time_window=timedelta(minutes=5),
    debug=True,
):
    """
    - Filters pre-loaded lsr_df to time_window & bbox
    - bounding_lat = (min_lat, max_lat)
    - bounding_lon = (min_lon, max_lon)
    Returns DataFrame of {source, time, station_lat, station_lon, gust, obs_distance}
    """
    min_lat, max_lat = bounding_lat
    min_lon, max_lon = bounding_lon

    if debug:
        logger.info(
            "[filter_lsr] bounding_lat=(%s,%s), bounding_lon=(%s,%s) | scan_time=%s window=±%s",
            min_lat,
            max_lat,
            min_lon,
            max_lon,
            scan_time,
            time_window,
        )

    # normalize scan_time to python datetime
    if hasattr(scan_time, "strftime") and not isinstance(scan_time, pd.Timestamp):
        try:
            scan_time_dt = datetime(
                scan_time.year,
                scan_time.month,
                scan_time.day,
                scan_time.hour,
                scan_time.minute,
                scan_time.second,
            )
            if debug:
                logger.info("[filter_lsr] converted scan_time to datetime: %s", scan_time_dt)
        except Exception:
            scan_time_dt = pd.to_datetime(str(scan_time))
            if debug:
                logger.info("[filter_lsr] fallback converted scan_time via to_datetime: %s", scan_time_dt)
    else:
        scan_time_dt = pd.Timestamp(scan_time)
        if debug:
            logger.info("[filter_lsr] scan_time is Timestamp: %s", scan_time_dt)

    df = lsr_df.copy()
    df["time"] = pd.to_datetime(df["time"])

    start_ts = pd.Timestamp(scan_time_dt) - time_window
    end_ts = pd.Timestamp(scan_time_dt) + time_window

    df = df[df["time"].between(start_ts, end_ts)]
    if debug:
        logger.info("[filter_lsr] after time filter: %d records", len(df))

    df = df[
        (df["lat"] >= min_lat)
        & (df["lat"] <= max_lat)
        & (df["lon"] >= min_lon)
        & (df["lon"] <= max_lon)
    ]
    if debug:
        logger.info("[filter_lsr] records after spatial filter: %d", len(df))

    df["obs_distance"] = df.apply(
        lambda r: haversine(center_lon, center_lat, r["lon"], r["lat"]),
        axis=1,
    )

    df = df.rename(columns={"lat": "station_lat", "lon": "station_lon"})
    df["source"] = "lsr_iastate"

    result = df[["source", "time", "station_lat", "station_lon", "gust", "obs_distance"]]

    if debug:
        logger.info("[filter_lsr] final returned records: %d", len(result))

    return result


##################################################################################################################################################################


def load_raw_spc(
    start: datetime,
    end: datetime,
    spc_dir: Union[str, Path] = "Datasets/surface_obs_datasets/spc_reports",
    debug: bool = False,
) -> pd.DataFrame:
    """
    Load SPC per-year 'wind' CSV(s) from a folder and return a DataFrame with:
      - time (python/pandas datetime)
      - lat, lon (averaged from slat/elat and slon/elon)
      - gust (float; from `mag`)
      - type (string; 'spc_wind')
    """
    folder = Path(spc_dir).expanduser()
    if not folder.is_dir():
        raise FileNotFoundError(f"Could not find SPC folder: {folder}")

    if debug:
        logger.info("[load_raw_spc] using folder: %s", folder)

    years = list(range(int(start.year), int(end.year) + 1))
    if debug:
        logger.info("[load_raw_spc] loading years: %s", years)

    dfs = []
    for y in years:
        fname = f"{y}_wind.csv"
        fpath = folder / fname
        if fpath.is_file():
            if debug:
                logger.info("[load_raw_spc] reading %s", fpath)
            try:
                dfi = pd.read_csv(fpath, low_memory=False)
                dfi["__source_file"] = fname
                dfs.append(dfi)
            except Exception:
                if debug:
                    logger.exception("[load_raw_spc] failed to read %s", fpath)
        else:
            if debug:
                logger.info("[load_raw_spc] file not found: %s (skipping)", fpath)

    if not dfs:
        raise FileNotFoundError(f"No SPC wind CSVs found in {folder} for years {years}.")

    df = pd.concat(dfs, ignore_index=True, sort=False)
    if debug:
        logger.info("[load_raw_spc] concatenated rows: %d", len(df))

    df["time"] = pd.to_datetime(
        df["date"].astype(str).str.strip() + " " + df["time"].astype(str).str.strip(),
        errors="coerce",
    )

    if debug:
        n_bad_time = int(df["time"].isna().sum())
        logger.info("[load_raw_spc] parsed time; %d rows failed to parse time", n_bad_time)

    for c in ["slat", "elat", "slon", "elon", "mag"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            df[c] = np.nan

    df["lat"] = df[["slat", "elat"]].mean(axis=1)
    df["lon"] = df[["slon", "elon"]].mean(axis=1)
    df["gust"] = df["mag"]

    if debug:
        logger.info(
            "[load_raw_spc] after coordinate/gust computation: rows=%d lat/nulls=%d lon/nulls=%d gust/nulls=%d",
            len(df),
            int(df["lat"].isna().sum()),
            int(df["lon"].isna().sum()),
            int(df["gust"].isna().sum()),
        )

    before = len(df)
    df = df.dropna(subset=["time", "lat", "lon", "gust"]).reset_index(drop=True)
    if debug:
        logger.info("[load_raw_spc] dropped %d rows missing time/lat/lon/gust -> %d remaining", before - len(df), len(df))

    df_out = pd.DataFrame(
        {
            "time": df["time"],
            "lat": df["lat"],
            "lon": df["lon"],
            "gust": df["gust"].astype(float),
            "type": "spc_wind",
        }
    )

    return df_out[["time", "lat", "lon", "gust", "type"]]


def filter_spc(
    spc_df,
    bounding_lat,
    bounding_lon,
    center_lat,
    center_lon,
    scan_time,
    time_window=timedelta(minutes=5),
    debug=True,
):
    """
    Filters pre-loaded spc_df to time_window & bbox and returns:
      ['source','time','station_lat','station_lon','gust','obs_distance']
    """
    min_lat, max_lat = bounding_lat
    min_lon, max_lon = bounding_lon

    if debug:
        logger.info(
            "[filter_spc] bounding_lat=(%s,%s), bounding_lon=(%s,%s) | scan_time=%s window=±%s",
            min_lat,
            max_lat,
            min_lon,
            max_lon,
            scan_time,
            time_window,
        )

    if hasattr(scan_time, "strftime") and not isinstance(scan_time, pd.Timestamp):
        try:
            scan_time_dt = datetime(
                scan_time.year,
                scan_time.month,
                scan_time.day,
                scan_time.hour,
                scan_time.minute,
                scan_time.second,
            )
            if debug:
                logger.info("[filter_spc] converted scan_time to datetime: %s", scan_time_dt)
        except Exception:
            scan_time_dt = pd.to_datetime(str(scan_time))
            if debug:
                logger.info("[filter_spc] fallback converted scan_time via to_datetime: %s", scan_time_dt)
    else:
        scan_time_dt = pd.Timestamp(scan_time)
        if debug:
            logger.info("[filter_spc] scan_time is Timestamp: %s", scan_time_dt)

    df = spc_df.copy()
    df["time"] = pd.to_datetime(df["time"])

    start_ts = pd.Timestamp(scan_time_dt) - time_window
    end_ts = pd.Timestamp(scan_time_dt) + time_window

    df = df[df["time"].between(start_ts, end_ts)]
    if debug:
        logger.info("[filter_spc] after time filter: %d records", len(df))

    df = df[
        (df["lat"] >= min_lat)
        & (df["lat"] <= max_lat)
        & (df["lon"] >= min_lon)
        & (df["lon"] <= max_lon)
    ]
    if debug:
        logger.info("[filter_spc] records after spatial filter: %d", len(df))

    df["obs_distance"] = df.apply(
        lambda r: haversine(center_lon, center_lat, r["lon"], r["lat"]),
        axis=1,
    )

    df = df.rename(columns={"lat": "station_lat", "lon": "station_lon"})
    df["source"] = "spc"

    result = df[["source", "time", "station_lat", "station_lon", "gust", "obs_distance"]].reset_index(drop=True)

    if debug:
        logger.info("[filter_spc] final returned records: %d", len(result))

    return result
