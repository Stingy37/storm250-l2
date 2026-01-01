from __future__ import annotations

import logging
import os
import re
from datetime import date
from io import BytesIO
from typing import Dict, List, Optional

from storm250.io import _save_gz_pickle, _load_field_pack, _load_gz_pickle, _ensure_dir, _save_field_pack

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)



def _product_key(p: str) -> str:
    # keep your sanitizer semantics
    return re.sub(r"[^0-9A-Za-z]+", "_", p).strip("_").lower() or "prod"


def _product_name_map(allowed_fields):
    # mapping from "file-safe key" -> "pyart field name"
    return {_product_key(f): f for f in allowed_fields}


#                        /- where r is a existing full PyART radar object
def _make_radar_skeleton(r):
    """
    Creates a Py-ART radar object but without fields or field metadata. Essentially only contains geometry.
    """
    import pyart
    import numpy as np

    # minimal copies; IMPORTANT PART -> no fields
    time = {"data": r.time["data"].copy()}
    rng_dict = {"data": r.range["data"].copy()}  # source attr is 'range'
    fields = {}  # where fields take up most of the memory... not lightweight metadata
    metadata = dict(getattr(r, "metadata", {}) or {})

    latitude = dict(r.latitude)
    longitude = dict(r.longitude)
    altitude = dict(r.altitude)

    sweep_number = r.sweep_number.copy()
    sweep_start_ray_index = r.sweep_start_ray_index.copy()
    sweep_end_ray_index = r.sweep_end_ray_index.copy()
    azimuth = {"data": r.azimuth["data"].copy()}
    elevation = {"data": r.elevation["data"].copy()}

    scan_type = getattr(r, "scan_type", "ppi")

    # optional dict-like attrs (only include if present)
    opt = {}
    for name in (
        "fixed_angle",
        "target_scan_rate",
        "rays_are_indexed",
        "ray_angle_res",
        "scan_rate",
        "antenna_transition",
        "altitude_agl",
    ):
        val = getattr(r, name, None)
        if val is not None:
            opt[name] = dict(val)

    # sweep_mode is required by many builds; ensure it exists
    sm = getattr(r, "sweep_mode", None)
    if sm is not None:
        sweep_mode = dict(sm)
    else:
        # create a reasonable default
        nsweeps = int(sweep_number["data"].size)
        default_mode = "azimuth_surveillance" if scan_type == "ppi" else "rhi"
        sweep_mode = {"data": np.array([default_mode] * nsweeps)}

    # keep bulky blocks out
    instrument_parameters = None
    radar_calibration = None

    # Build the new PyART radar object via keywords; be compatible with both `_range` and `range`
    try:
        return pyart.core.Radar(
            time=time,
            _range=rng_dict,
            fields=fields,
            metadata=metadata,
            scan_type=scan_type,
            latitude=latitude,
            longitude=longitude,
            altitude=altitude,
            sweep_number=sweep_number,
            sweep_mode=sweep_mode,
            fixed_angle=opt.get("fixed_angle"),
            sweep_start_ray_index=sweep_start_ray_index,
            sweep_end_ray_index=sweep_end_ray_index,
            azimuth=azimuth,
            elevation=elevation,
            altitude_agl=opt.get("altitude_agl"),
            target_scan_rate=opt.get("target_scan_rate"),
            rays_are_indexed=opt.get("rays_are_indexed"),
            ray_angle_res=opt.get("ray_angle_res"),
            scan_rate=opt.get("scan_rate"),
            antenna_transition=opt.get("antenna_transition"),
            instrument_parameters=instrument_parameters,
            radar_calibration=radar_calibration,
        )
    except TypeError as e:
        # Some older/newer Py-ART builds use 'range' instead of '_range'
        if "unexpected keyword argument '_range'" in str(e):
            return pyart.core.Radar(
                time=time,
                range=rng_dict,
                fields=fields,
                metadata=metadata,
                scan_type=scan_type,
                latitude=latitude,
                longitude=longitude,
                altitude=altitude,
                sweep_number=sweep_number,
                sweep_mode=sweep_mode,
                fixed_angle=opt.get("fixed_angle"),
                sweep_start_ray_index=sweep_start_ray_index,
                sweep_end_ray_index=sweep_end_ray_index,
                azimuth=azimuth,
                elevation=elevation,
                altitude_agl=opt.get("altitude_agl"),
                target_scan_rate=opt.get("target_scan_rate"),
                rays_are_indexed=opt.get("rays_are_indexed"),
                ray_angle_res=opt.get("ray_angle_res"),
                scan_rate=opt.get("scan_rate"),
                antenna_transition=opt.get("antenna_transition"),
                instrument_parameters=instrument_parameters,
                radar_calibration=radar_calibration,
            )
        raise


def _slim_to_fields(r, allowed_fields):
    """
    In-place: drop all fields not in allowed_fields. Downcast to float32 to shrink memory.
    """
    keep = set(allowed_fields)
    for k in list(r.fields.keys()):
        if k not in keep:
            del r.fields[k]
        else:
            arr = r.fields[k]["data"]
            if arr.dtype != np.float32:
                r.fields[k]["data"] = np.asarray(arr, dtype=np.float32)


def _ray_canonical_roll(az):
    """Return roll offset so that the first ray is the one with the smallest azimuth (0..360)."""
    azm = np.mod(np.asarray(az, dtype=float), 360.0)
    if azm.size == 0 or not np.isfinite(azm).any():
        return 0
    return int(np.nanargmin(azm))


def find_radar_scans(
    storm_df: pd.DataFrame,
    site_column: str = "radar_site",
    time_column: str = "time",
    level2_base: str = "unidata-nexrad-level2",  # AWS Level II bucket (name or s3://bucket)
    cache_dir: str = "Datasets/nexrad_datasets/level_two_raw",  # keep local Drive/FS cache
    product_filter: List = None,  # e.g. ["reflectivity","velocity","zdr","rhohv"]
    time_tolerance_seconds: int = 29,  # +/- tolerance for matching volume start times
    keep_in_memory: bool = True,  # keep Py-ART Radar object(s) in the returned DF
    debug: bool = False,
) -> pd.DataFrame:
    """
    Link GR-S storm rows to NEXRAD Level II volume files on AWS S3 (Unidata bucket).
    For each storm timestamp (UTC), find the nearest Level II volume for the storm's radar
    within +/- `time_tolerance_seconds`. Optionally restrict to specific Level II fields
    using `product_filter` (e.g., ["reflectivity","velocity","zdr"]). Then, build cache system
    for each Level II volume to (optionally) prevent in-memory storage and allow easy reconstruction in
    future runs. Each PYART radar object is split into lightweight geometry (one gz pkl file), then
    npz array + field metadata for each field (reflectivity, velocity, etc.)

    Returns a subset of `storm_df` with extra columns:
      - If product_filter is provided: for each normalized field key `k`:
          * `{k}_scan`                : Py-ART Radar object (or None if keep_in_memory=False)
          * `{k}_cache_volume_path`   : local pickle path where the Radar object is cached
                                         \- for ALL field rows resulting from a single radar volume,
                                         this path should point to the same gz pkl file
          * `{k}_matched_volume_s3_key` : matched S3 object key (filename)
                                        \- for ALL field rows resulting from a single radar volume,
                                         this path should point back to the same S3 object
          - Note that each field gets its own columns
      - Else (no product_filter): legacy single triple:
          * 'radar_scan', 'cache_member_name', 'matched_member_name'

    Notes:
      - Uses AWS S3 bucket `unidata-nexrad-level2` by default; accepts either bare bucket
        name or full "s3://bucket" in `level2_base`.
      - No tar handling (complete migration to Level II single-file volumes).
      - Preserves overall structure, caching, and debug verbosity from Level III version.
    """

    ############################################################ HELPERS ############################################################

    # Utilities / Normalizers
    def _strip_s3_prefix(b):
        return b[5:] if isinstance(b, str) and b.startswith("s3://") else b

    def _s3_uri(bucket: str, key: str) -> str:
        return f"s3://{_strip_s3_prefix(bucket).rstrip('/')}/{key.lstrip('/')}"

    # Canonicalize product names to Py-ART Level II field keys
    # (Accept common aliases; fall back to lowercased alnum)
    _LV2_ALIASES = {
        "reflectivity": {"reflectivity", "refl", "ref", "dz", "z"},
        "velocity": {"velocity", "vel", "vr"},
        "spectrum_width": {"spectrum_width", "sw", "width"},
        "differential_reflectivity": {"differential_reflectivity", "zdr"},
        "differential_phase": {"differential_phase", "phidp", "phi", "kdp?"},
        "cross_correlation_ratio": {"cross_correlation_ratio", "rho", "rhohv"},
        "normalized_coherent_power": {"normalized_coherent_power", "ncp"},
        "clutter_filter_power_removed": {"clutter_filter_power_removed", "cfpr", "cpwr"},
    }

    def _normalize_lv2_product_filter(pf: Optional[List[str]]) -> List[str]:
        if not pf:
            return []
        norm = []
        for raw in pf:
            if raw is None:
                continue
            s = str(raw).strip().lower()
            hit = None
            for key, aliases in _LV2_ALIASES.items():
                if s == key or s in aliases:
                    hit = key
                    break
            if not hit:
                # fallback: clean up to a "likely" field name; Py-ART may not have it but keep column schema
                hit = re.sub(r"[^0-9a-z]+", "_", s).strip("_")
            if hit not in norm:
                norm.append(hit)
        return norm

    _s3fs = None
    _s3client = None

    try:
        import s3fs  # best path: lets Py-ART read s3:// URIs directly

        _s3fs = s3fs.S3FileSystem(anon=True)
        if debug:
            logger.info("[find_radar_scans] using s3fs (anonymous)")
    except Exception as e:
        if debug:
            logger.info("[find_radar_scans] s3fs not available: %s - falling back to boto3", e)
        try:
            import boto3
            from botocore import UNSIGNED
            from botocore.client import Config

            _s3client = boto3.client("s3", config=Config(signature_version=UNSIGNED), region_name="us-east-1")
            if debug:
                logger.info("[find_radar_scans] using boto3 (unsigned) for S3 access")
        except Exception as e2:
            if debug:
                logger.info("[find_radar_scans] ERROR: no S3 access libraries available: %s", e2)
            # We'll continue; listing will fail and return empty.

    def _s3_ls(bucket: str, prefix: str) -> List[str]:
        """
        Return a list of 's3://bucket/key' paths under the prefix (no directories).
        """
        full = _s3_uri(bucket, prefix)
        out = []
        if _s3fs is not None:
            try:
                for p in _s3fs.ls(full, detail=False):
                    # s3fs returns 's3://bucket/key' for objects
                    # Filter out "directories" if any
                    if p.endswith("/"):
                        continue
                    out.append(p)
            except Exception as e:
                if debug:
                    logger.info("[find_radar_scans] s3fs.ls(%s) failed: %s", full, e)
        elif _s3client is not None:
            try:
                Bucket = _strip_s3_prefix(bucket)
                Prefix = prefix.lstrip("/")
                paginator = _s3client.get_paginator("list_objects_v2")
                for page in paginator.paginate(Bucket=Bucket, Prefix=Prefix):
                    for it in page.get("Contents", []):
                        out.append(_s3_uri(Bucket, it["Key"]))
            except Exception as e:
                if debug:
                    logger.info("[find_radar_scans] boto3 list_objects_v2(%s,%s) failed: %s", bucket, prefix, e)
        return out

    def _s3_read_pyart(bucket: str, key: str):
        """
        Main IO operation: read a Level II volume into a Py-ART Radar object.
        Prefer s3fs path; fallback to boto3 BytesIO.

        Returns the Py-ART radar object
        """
        import pyart

        s3path = _s3_uri(bucket, key)
        try:
            if _s3fs is not None:
                # Py-ART 2.0+ reads s3:// URIs with s3fs installed
                return pyart.io.read_nexrad_archive(s3path)
            elif _s3client is not None:
                resp = _s3client.get_object(Bucket=_strip_s3_prefix(bucket), Key=key.lstrip("/"))
                bio = BytesIO(resp["Body"].read())
                return pyart.io.read_nexrad_archive(bio)
        except Exception as e:
            if debug:
                logger.info("[find_radar_scans] pyart read failed for %s: %s", s3path, e)
        return None

    ########################################################## NORMALIZE INPUT DF ############################################################

    # Quick-empty checks
    if storm_df is None or len(storm_df) == 0:
        if debug:
            logger.info("[find_radar_scans] empty storm_df, returning empty DataFrame")
        return pd.DataFrame(columns=list(storm_df.columns) + ["radar_scan", "cache_member_name"])

    # Normalize time column
    storm_df = storm_df.copy()
    storm_df[time_column] = pd.to_datetime(storm_df[time_column], utc=True, errors="coerce")
    storm_df = storm_df.dropna(subset=[time_column])
    if len(storm_df) == 0:
        if debug:
            logger.info("[find_radar_scans] all times coerce->NaT, returning empty")
        return pd.DataFrame(columns=list(storm_df.columns) + ["radar_scan", "cache_member_name"])

    # Normalize requested Level II fields
    allowed_fields = _normalize_lv2_product_filter(product_filter)  # [] = accept-all (legacy single column)
    allowed_keys = [_product_key(p) for p in allowed_fields] if allowed_fields else []

    # Expect single radar site per group
    sites = storm_df[site_column].dropna().unique().tolist()
    if len(sites) == 0:
        raise ValueError("No radar site present in storm_df")
    if len(sites) > 1 and debug:
        logger.info(
            "[find_radar_scans] Warning: multiple radar sites found for group: %s. Using first: %s", sites, sites[0]
        )
    site = sites[0].upper().strip()

    # Compute date range for S3 prefix listing
    times = storm_df[time_column].sort_values()
    min_dt = times.min()
    max_dt = times.max()
    start_date = min_dt.date()
    end_date = max_dt.date()

    # create caches
    bucket = level2_base or "unidata-nexrad-level2"
    bucket = _strip_s3_prefix(bucket)
    pkl_cache_dir = os.path.join(cache_dir, site)
    _ensure_dir(pkl_cache_dir)

    if debug:
        logger.info("[find_radar_scans] site=%s, date range %s -> %s", site, start_date, end_date)
        logger.info("[find_radar_scans] S3 bucket=%s (AWS Level II)", bucket)
        logger.info("[find_radar_scans] pkl_cache_dir=%s", pkl_cache_dir)

    # Build set of target timestamps (rounded to seconds)
    target_times = list(times.dt.round("s"))
    date_to_times: Dict[date, List[pd.Timestamp]] = {}
    for ts in target_times:
        date_to_times.setdefault(ts.date(), []).append(ts)

    # Results
    linked_rows: List[dict] = []
    tol = pd.Timedelta(seconds=time_tolerance_seconds)

    ################################################### SEARCH FOR S3 LEVEL II VOLUME ############################################################

    # Iterate by date; list S3 once per day
    for day, times_for_day in date_to_times.items():
        y = f"{day:%Y}"
        m = f"{day:%m}"
        d = f"{day:%d}"
        prefix = f"{y}/{m}/{d}/{site}/"

        # List all Level II objects for this site/day
        objects = _s3_ls(bucket, prefix)
        if not objects:
            if debug:
                logger.info(
                    "[find_radar_scans] no Level II objects for %s on %s under s3://%s/%s", site, day, bucket, prefix
                )
            continue
        if debug:
            logger.info("[find_radar_scans] %d Level II object(s) for %s on %s", len(objects), site, day)

        # Map: file_time -> object key (most days have multiple)
        file_time_to_key: Dict[pd.Timestamp, str] = {}

        # Example name: KHGX20220322_120125_V06  (no extension)
        # We allow optional extension just in case.
        name_re = re.compile(rf"{re.escape(site)}(\d{{8}})_(\d{{6}})(?:_[A-Za-z0-9]+)?(?:\.[A-Za-z0-9]+)?$")

        for s3path in objects:
            fname = os.path.basename(s3path)
            mo = name_re.search(fname)
            if not mo:
                continue
            ymd = mo.group(1)
            hms = mo.group(2)
            timestr = f"{ymd}{hms}"
            ftime = pd.to_datetime(timestr, format="%Y%m%d%H%M%S", utc=True, errors="coerce")
            if pd.isna(ftime):
                continue
            # Store key (bucket-relative)
            key = f"{y}/{m}/{d}/{site}/{fname}"
            file_time_to_key[ftime] = key

        if debug:
            logger.info("[find_radar_scans] parsed %d volume times for %s on %s", len(file_time_to_key), site, day)

        if not file_time_to_key:
            continue

        # For each storm time, find nearest volume within tolerance
        matched_count_for_day = 0
        volume_times_sorted = sorted(file_time_to_key.keys())

        for storm_ts in times_for_day:
            # Binary search for closest time
            # (small N per day; linear scan also fine; keep readable)
            best_time = None
            best_dt_delta = None
            for vts in volume_times_sorted:
                delta = abs(storm_ts - vts)
                if delta <= tol:
                    if best_time is None or delta < best_dt_delta:
                        best_time = vts
                        best_dt_delta = delta
                elif vts > storm_ts and (best_time is not None) and (vts - storm_ts) > best_dt_delta + tol:
                    # simple early break once we're beyond a plausible better match
                    break

            if best_time is None:
                if debug:
                    logger.info("[find_radar_scans] no L2 match within +/-%ss for %s", time_tolerance_seconds, storm_ts)
                continue

            key = file_time_to_key[best_time]
            fname = os.path.basename(key)

            ############################################# BUILD / SAVE RADAR OBJECT ####################################################
            #                   load the radar object and immediately slim it (remove unnecessary fields) + split into
            #                                                corresponding caches

            # field-level caching paths
            base_rel = os.path.join(day.strftime("%Y%m%d"), fname)
            base_dir = os.path.join(pkl_cache_dir, base_rel)
            skeleton_path = base_dir + ".skeleton.pkl.gz"

            # Unified path to geometry cache, exposed in df for all fields resulting from this volume
            df_cache_path = skeleton_path if allowed_fields else (base_dir + ".radar.pkl.gz")

            # radar_obj -> represents the actual volume we will get from s3 OR from existing cache
            # IMPORTANT: radar_obj is a Py-ART radar object (always)
            #
            #              ___________________________________
            #              | radar_obj (Py-ART radar object) |
            #              ----------------|------------------
            #                              |
            #          |-------------------|--------------------------------------|
            #    ______|__________________________________________           _____|____
            #    | geometry metadata (one for the entire volume) |           | fields |
            #    -------------------------------------------------           -----|----
            #                                                      _______________|__________________
            #                                                      | field metadata (one per field) |
            #                                                      ----------------------------------
            radar_obj = None

            if allowed_fields:
                keymap = _product_name_map(
                    allowed_fields
                )  # {'reflectivity': 'reflectivity', ...} via sanitized key mapping if needed
                field_paths = {k: base_dir + f".{k}" for k in keymap.keys()}  # per-field base (we add .npz/.json)

                # Check if we already have a full set in cache
                have_skeleton = os.path.exists(skeleton_path)
                have_all_fields = all(
                    os.path.exists(p + ".npz") and os.path.exists(p + ".json") for p in field_paths.values()
                )

                # if we have geometry skeleton + field caches (npz + json) then just rebuild the radar_obj from that
                if have_skeleton and have_all_fields:
                    # Rebuild slim radar from cache
                    sk = _load_gz_pickle(skeleton_path, debug=debug)  # where sk -> geometry metadata
                    if sk is not None:  #                                   |
                        radar_obj = sk  #                             but crucially, sk stores the geometry metadata AS a Py-ART radar object with empty fields
                        #                                   \- hence, we can set radar_obj = sk and fill in the fields by loading from the field caches
                        # now load in the fields
                        for k, fldname in keymap.items():
                            pack = _load_field_pack(field_paths[k], debug=debug)
                            if pack is not None:
                                radar_obj.fields[fldname] = pack
                        if debug:
                            logger.info(
                                "[find_radar_scans] cache HIT (skeleton + %s) at %s", list(keymap.keys()), base_dir
                            )

            # If not fully cached, fetch from S3, slim, and write field-level caches
            if radar_obj is None:
                radar_obj = _s3_read_pyart(bucket, key)
                if radar_obj is None:
                    if debug:
                        logger.info("[find_radar_scans] FAILED to read %s from s3://%s", key, bucket)
                    continue

                # separate radar_obj into caches
                if allowed_fields:
                    # Keep only requested fields & convert field arrays to float32 (lowers memory usage before we even start building caches)
                    _slim_to_fields(radar_obj, allowed_fields)

                    # Write skeleton Py-ART radar object, and save it as a gz pickle (compressed pkl)
                    _save_gz_pickle(_make_radar_skeleton(radar_obj), skeleton_path, debug=debug)
                    if debug:
                        logger.info("[find_radar_scans] skeleton WRITE: %s", skeleton_path)

                    # now build the cache for fields (results in npz and json being saved, per field)
                    for fldname in allowed_fields:
                        if fldname in radar_obj.fields:
                            k = _product_key(fldname)  # file-safe key
                            _save_field_pack(base_dir + f".{k}", radar_obj.fields[fldname], downcast=True, debug=debug)
                    if debug:
                        logger.info("[find_radar_scans] fields WRITE at base: %s", base_dir)
                else:
                    # Legacy (no product_filter): still shrink + compress a single file to avoid 400MB pickles
                    # PATH: single gz-pickle instead of raw pickle
                    single_path = df_cache_path
                    _save_gz_pickle(radar_obj, single_path, debug=debug)
                    if debug:
                        logger.info("[find_radar_scans] cache WRITE (single gz): %s", single_path)

            ################################################## BUILD RETURNED DATAFRAME ##################################################

            # Find matching storm_df rows with that exact normalized timestamp
            matching_row_mask = storm_df[time_column] == storm_ts
            matched_rows = storm_df[matching_row_mask]
            if matched_rows.empty:
                if debug:
                    logger.info(
                        "[find_radar_scans] no exact storm_df rows matched post-normalization for time %s", storm_ts
                    )
                continue

            # Attach columns (multi-field or legacy single)
            if allowed_fields:
                # build product-specific triples; reuse the *same* Radar object per field
                prod_infos = {}
                for fld in allowed_fields:
                    k = _product_key(fld)
                    prod_infos[k] = {
                        "scan": radar_obj if keep_in_memory else None,
                        "pkl": df_cache_path,
                        "name": key,  # S3 key used as "matched member name"
                    }

                for _, row in matched_rows.iterrows():
                    out = row.to_dict()
                    for k, info in prod_infos.items():
                        out[f"{k}_scan"] = info["scan"]
                        out[f"{k}_cache_volume_path"] = info["pkl"]
                        out[f"{k}_matched_volume_s3_key"] = info["name"]
                    linked_rows.append(out)
                    matched_count_for_day += 1
                    if debug:
                        logger.info("[find_radar_scans] linked %s -> %s fields=%s", storm_ts, fname, allowed_fields)
            else:
                # legacy single triple, SHOULDN'T happen
                for _, row in matched_rows.iterrows():
                    out = row.to_dict()
                    out["radar_scan"] = radar_obj if keep_in_memory else None
                    out["cache_member_name"] = df_cache_path
                    out["matched_member_name"] = key
                    linked_rows.append(out)
                    matched_count_for_day += 1
                    if debug:
                        logger.info("[find_radar_scans] linked %s -> %s", storm_ts, fname)

        if debug:
            logger.info("[find_radar_scans] finished %s %s - matched %d row(s)", site, day, matched_count_for_day)

    # Build final linked DataFrame
    if len(linked_rows) == 0:
        if debug:
            logger.info("[find_radar_scans] found no matches; returning empty DataFrame")
        # Keep consistent schema
        base_cols = list(storm_df.columns)
        if allowed_fields:
            extra = []
            for fld in allowed_fields:
                k = _product_key(fld)
                extra += [f"{k}_scan", f"{k}_cache_volume_path", f"{k}_matched_volume_s3_key"]
            return pd.DataFrame(columns=base_cols + extra)
        else:
            # legacy, SHOULDN'T get here
            return pd.DataFrame(columns=base_cols + ["radar_scan", "cache_member_name", "matched_member_name"])

    linked_df = pd.DataFrame(linked_rows)

    # Column ordering like before
    if allowed_fields:
        prod_keys = [_product_key(p) for p in allowed_fields]
        extra_cols = []
        for k in prod_keys:
            extra_cols += [f"{k}_scan", f"{k}_cache_volume_path", f"{k}_matched_volume_s3_key"]
        cols = list(storm_df.columns) + extra_cols
    else:
        cols = list(storm_df.columns) + ["radar_scan", "cache_member_name", "matched_member_name"]

    linked_df = linked_df.reindex(columns=cols)

    if debug:
        logger.info("[find_radar_scans] returning linked_df with %d rows for site %s", len(linked_df), site)
        logger.info("linked_radar_df shape: %s", getattr(linked_df, "shape", None))
        logger.info("linked_radar_df head(50):\n%s", linked_df.head(50).to_string(index=False))

    return linked_df
