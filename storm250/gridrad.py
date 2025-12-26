import os
import io
import re
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed


############################################################################ HELPERS ############################################################################


def _prepare_site_arrays(radar_info: dict):
    """
    From dict-of-dicts radar_info -> (names, lat_array, lon_array, list_of_range_arrays)
    Each range array is shape (R, 2) of int64 nanoseconds [start_ns, end_ns].
    None -> open interval (min/max int64).
    """
    names, lats, lons, ranges = [], [], [], []
    int_min, int_max = np.iinfo(np.int64).min, np.iinfo(np.int64).max

    for site, meta in radar_info.items():
        pos = (meta or {}).get('position', {})
        lat = pos.get('lat', None)
        lon = pos.get('lon', None)
        if lat is None or lon is None:
            continue  # skip incomplete entries

        # normalize active_ranges
        ar = (meta or {}).get('active_ranges') or [{'start': None, 'end': None}]
        rs = []
        for span in ar:
            s = span.get('start', None)
            e = span.get('end', None)
            # cast to UTC ns; None -> open
            s_ns = int_min if s is None else pd.to_datetime(s, utc=True).value
            e_ns = int_max if e is None else pd.to_datetime(e, utc=True).value
            rs.append((s_ns, e_ns))

        names.append(site)
        lats.append(lat)
        lons.append(lon)
        ranges.append(np.asarray(rs, dtype=np.int64))  # (R, 2)

    return (
        names,
        np.asarray(lats, dtype=np.float32),
        np.asarray(lons, dtype=np.float32),
        ranges,  # list of (R,2) int64 arrays per site
    )


def _active_mask_for_chunk(t_chunk_ns: np.ndarray, site_ranges: list) -> np.ndarray:
    """
    Build a boolean mask [L, S] where True means site S is active at row-time L.
    site_ranges[j] is an (R,2) ns array of [start,end] for site j (open intervals handled already).
    """
    L = t_chunk_ns.shape[0]
    S = len(site_ranges)
    mask = np.zeros((L, S), dtype=bool)

    # vectorize per-site (rows x ranges), then OR over ranges
    for j, rs in enumerate(site_ranges):
        if rs.size == 0:
            continue
        starts = rs[:, 0][None, :]  # [1, R]
        ends   = rs[:, 1][None, :]  # [1, R]
        tt     = t_chunk_ns[:, None]  # [L, 1]
        m = (tt >= starts) & (tt <= ends)  # [L, R]
        mask[:, j] = m.any(axis=1)

    return mask


######################################################################## LOAD GRS TRACKS ##########################################################################


def load_grs_tracks(year: int,
                    radar_info: dict,
                    base_url: str = "https://data-osdf.rda.ucar.edu/ncar/rda/d841006/tracks",
                    min_rows: int = 5,
                    max_distance_km: float = 250.0,
                    debug: bool = False,
                    timeout: float = 20.0,
                    save_dir: str = "Datasets/cell_tracks/raw_grs",
                    processed_cache_dir: str = "Datasets/cell_tracks/processed_grs",
                    max_workers: int = 10,
                    chunk_size: int = 100_000,
                    max_gap_hours: float = 6.0   # ← NEW: continuity gap threshold
                    ) -> pd.DataFrame:
    """
    Download & cache daily GR-S CSVs, parse them in parallel (fast C-engine if possible),
    compute nearest radar site (vectorized chunked haversine), split multi-site storms
    (vectorized), drop small groups, snake_case columns, and save processed-year cache.

    All original debug printing and logic preserved.
    """
    # ensure cache dirs
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(processed_cache_dir, exist_ok=True)

    processed_parquet = os.path.join(processed_cache_dir, f"{year}.parquet")
    processed_csv = os.path.join(processed_cache_dir, f"{year}.csv")

    # Quick processed-cache short-circuit
    if os.path.exists(processed_parquet):
        if debug:
            print(f"[load_grs_tracks] loading processed cache {processed_parquet}")
        try:
            return pd.read_parquet(processed_parquet)
        except Exception:
            if debug:
                print("[load_grs_tracks] parquet read failed — will try CSV or re-process")
            if os.path.exists(processed_csv):
                return pd.read_csv(processed_csv)

    # helper to download a single day (safe atomic write)
    def _download_day(url, local_path):
        tmp = local_path + ".tmp"
        try:
            r = requests.get(url, allow_redirects=True, timeout=timeout)
        except Exception as e:
            if debug:
                print(f"[download] request error {url}: {e}")
            return False, url
        if r.status_code != 200:
            return False, url
        content = r.content
        if not content or len(content) < 50:
            return False, url
        try:
            with open(tmp, "wb") as f:
                f.write(content)
            os.replace(tmp, local_path)
            return True, url
        except Exception as e:
            if debug:
                print(f"[download] write error for {local_path}: {e}")
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except Exception:
                pass
            return False, url

    # Build list of all YYYYMMDD dates for the year
    start_date = datetime(year, 1, 1)
    end_date = datetime(year, 12, 31)
    dates = []
    d = start_date
    while d <= end_date:
        dates.append(d)
        d += timedelta(days=1)

    # Check local cache and collect missing days
    missing = []
    for day in dates:
        ymd = day.strftime("%Y%m%d")
        local_path = os.path.join(save_dir, f"{ymd}.csv")
        if not (os.path.exists(local_path) and os.path.getsize(local_path) > 50):
            missing.append((day, local_path))
    if debug:
        print(f"[load_grs_tracks] {len(dates)} days in year, {len(missing)} missing cached files")

    # Download missing days in parallel (I/O bound)
    if missing:
        if debug:
            print(f"[load_grs_tracks] launching downloader with max_workers={max_workers} ...")
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {}
            for day, local_path in missing:
                ymd = day.strftime("%Y%m%d")
                url = f"{base_url}/{year}/{ymd}.csv"
                futures[ex.submit(_download_day, url, local_path)] = (day, local_path)
            for fut in as_completed(futures):
                ok, url = fut.result()
                if debug:
                    if ok:
                        print(f"[load_grs_tracks] downloaded: {url}")
                    else:
                        print(f"[load_grs_tracks] missed / 404 or error: {url}")



    # Parse all cached daily files concurrently (fast local I/O, C engine first)
    def _parse_local_file_path(local_path):
        """
        Parse a local CSV file path. Try fast C-engine with common delimiters first,
        fall back to python engine auto-detect if necessary. Return DataFrame or None.
        """
        try:
            # 1) try comma with C engine
            try:
                df = pd.read_csv(local_path, sep=",", engine="c", low_memory=False)
                if df.shape[1] > 1:
                    return df
            except Exception:
                pass

            # 2) try tab with C engine
            try:
                df = pd.read_csv(local_path, sep="\t", engine="c", low_memory=False)
                if df.shape[1] > 1:
                    return df
            except Exception:
                pass

            # 3) fallback: python engine auto-detect (slower)
            try:
                df = pd.read_csv(local_path, sep=None, engine="python")
                return df
            except Exception:
                return None
        except Exception:
            return None

    # collect list of local paths that exist
    local_paths = []
    for day in dates:
        ymd = day.strftime("%Y%m%d")
        local_path = os.path.join(save_dir, f"{ymd}.csv")
        if os.path.exists(local_path) and os.path.getsize(local_path) > 50:
            local_paths.append(local_path)
        else:
            if debug:
                print(f"[load_grs_tracks] no file for {ymd} @ path {local_path} skipping")

    if debug:
        print(f"[load_grs_tracks] parsing {len(local_paths)} cached files with max_workers={max_workers}")

    frames = []
    # Use a ThreadPoolExecutor; pandas C engine releases the GIL so threads are effective here
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_parse_local_file_path, p): p for p in local_paths}
        for fut in as_completed(futures):
            p = futures[fut]
            try:
                df = fut.result()
            except Exception as e:
                df = None
                if debug:
                    print(f"[load_grs_tracks] parse worker failed for {p}: {e}")
            if df is not None:
                # same debug-style print line you used
                if debug:
                    print(f"[load_grs_tracks] parsed {p} -> rows={len(df)}, cols={len(df.columns)}")
                frames.append(df)
            else:
                if debug:
                    print(f"[load_grs_tracks] parsed {p} -> returned None or failed")

    if not frames:
        if debug:
            print(f"[load_grs_tracks] no track files parsed for year {year}")
        return pd.DataFrame()

    # CONCATENATE
    grs = pd.concat(frames, ignore_index=True, sort=False)
    if debug:
        print(f"[load_grs_tracks] concatenated {len(frames)} files -> {len(grs)} rows")
        # show columns for debugging
        if debug:
            print(f"[load_grs_tracks] concatenated columns: {list(grs.columns)}")

    # IMPORTANT: canonicalize core column names after concat
    # Strip whitespace from column names and collapse repeated spaces
    new_cols = []
    for c in grs.columns:
        # make sure it's a string
        s = str(c).strip()
        new_cols.append(s)
    grs.columns = new_cols

    # heuristically find the lon/lat/time/storm columns (case-insensitive)
    cols_lower = {c.lower(): c for c in grs.columns}

    def _find_col(*keywords):
        for lowname, orig in cols_lower.items():
            if all(k.lower() in lowname for k in keywords):
                return orig
        return None

    # find storm column (we only need it in order to create _orig_storm_number if missing)
    storm_col_found = _find_col("storm", "number") or _find_col("storm") or next((c for c in grs.columns if "storm" in c.lower()), None)
    time_col_found = _find_col("time") or next((c for c in grs.columns if "time" in c.lower()), None)
    lon_col_found = _find_col("long") or next((c for c in grs.columns if "lon" in c.lower()), None)
    lat_col_found = _find_col("lat") or next((c for c in grs.columns if "lat" in c.lower()), None)

    if debug:
        print(f"[load_grs_tracks] detected post-concat columns -> storm: {storm_col_found}, time: {time_col_found}, lon: {lon_col_found}, lat: {lat_col_found}")

    if not (lon_col_found and lat_col_found and time_col_found):
        # If we can't find core columns, throw a helpful error with available columns
        raise KeyError(f"Couldn't find longitude/latitude/time columns after concat. Available columns: {list(grs.columns)}")

    # rename canonical columns so downstream code can rely on them
    rename_map = {}
    rename_map[lon_col_found] = "Longitude"
    rename_map[lat_col_found] = "Latitude"
    rename_map[time_col_found] = "Time"
    grs = grs.rename(columns=rename_map)

    # Ensure _orig_storm_number exists (may be named differently in files)
    if "_orig_storm_number" not in grs.columns:
        if storm_col_found:
            grs["_orig_storm_number"] = pd.to_numeric(grs[storm_col_found], errors="coerce").astype('Int64')
        else:
            # attempt any column with 'storm' substring
            possible = next((c for c in grs.columns if "storm" in c.lower()), None)
            if possible:
                grs["_orig_storm_number"] = pd.to_numeric(grs[possible], errors="coerce").astype('Int64')

    # Now the original dropna call is safe
    grs = grs.dropna(subset=["Longitude", "Latitude", "_orig_storm_number", "Time"]).reset_index(drop=True)
    if debug:
        print(f"[load_grs_tracks] rows after dropping NaNs in core fields: {len(grs)}")



    # FAST nearest-site via chunked vectorized haversine (unchanged math; now time-aware)
    # Prepare site arrays from dict-of-dicts + active date ranges
    site_names, site_lats, site_lons, site_ranges = _prepare_site_arrays(radar_info)

    if not site_names:
        raise ValueError("radar_info is empty or lacks valid positions.")

    # prepare site arrays (radians, float32 to reduce mem)
    site_lats_rad = np.deg2rad(site_lats).astype(np.float32)
    site_lons_rad = np.deg2rad(site_lons).astype(np.float32)

    N = len(grs)
    nearest_sites_idx = np.full(N, -1, dtype=np.int32)  # -1 = no active site
    nearest_dists = np.full(N, np.nan, dtype=np.float32)

    lon_arr = grs["Longitude"].to_numpy(dtype=np.float32)
    lat_arr = grs["Latitude"].to_numpy(dtype=np.float32)

    # Time as int64 ns for range checks
    t_arr = pd.to_datetime(grs["Time"], utc=True, errors="coerce")
    t_ns_arr = t_arr.view('int64').to_numpy()

    R_earth_km = 6371.0  # km

    def _compute_chunk(i0: int, i1: int, active_mask: np.ndarray):
        """
        Compute distances [L,S] for rows i0:i1, then mask out inactive sites,
        pick argmin per row, and return (idx, mindist).
        """
        la = lat_arr[i0:i1].astype(np.float32)
        lo = lon_arr[i0:i1].astype(np.float32)
        la_rad = np.deg2rad(la).reshape(-1, 1)
        lo_rad = np.deg2rad(lo).reshape(-1, 1)
        s_lat = site_lats_rad.reshape(1, -1)
        s_lon = site_lons_rad.reshape(1, -1)

        dlat = la_rad - s_lat
        dlon = lo_rad - s_lon
        sin_dlat2 = np.sin(dlat * 0.5) ** 2
        sin_dlon2 = np.sin(dlon * 0.5) ** 2
        a = sin_dlat2 + np.cos(la_rad) * np.cos(s_lat) * sin_dlon2
        a = np.clip(a, 0.0, 1.0)
        c = 2.0 * np.arcsin(np.sqrt(a))
        d = (R_earth_km * c).astype(np.float32)   # [L,S]

        # mask out sites not active at the row's time
        # inactive -> set to +inf so they can't be selected
        # active_mask is [L,S], True=active
        d[~active_mask] = np.inf

        # pick nearest among active
        idx = np.argmin(d, axis=1)
        mind = d[np.arange(d.shape[0]), idx]

        # rows with no active sites -> mind = inf, set idx=-1, dist=NaN
        no_active = ~np.isfinite(mind)
        if no_active.any():
            idx = idx.astype(np.int32, copy=True)
            mind = mind.astype(np.float32, copy=True)
            idx[no_active] = -1
            mind[no_active] = np.nan

        return idx.astype(np.int32), mind.astype(np.float32)

    if debug:
        print(f"[load_grs_tracks] computing nearest active site in chunks of {chunk_size} rows")

    i0 = 0
    while i0 < N:
        i1 = min(N, i0 + chunk_size)
        # build the [L,S] active-site mask for this chunk’s timestamps
        active_mask = _active_mask_for_chunk(t_ns_arr[i0:i1], site_ranges)

        idxs, dists = _compute_chunk(i0, i1, active_mask)
        nearest_sites_idx[i0:i1] = idxs
        nearest_dists[i0:i1] = dists
        if debug:
            na = int(np.sum(np.isnan(dists)))
            print(f"[load_grs_tracks] processed rows {i0}:{i1} (rows w/ no active site: {na})")
        i0 = i1

    # map indices to names; -1 -> None
    nearest_sites = [site_names[i] if i >= 0 else None for i in nearest_sites_idx]
    grs["radar_site"] = nearest_sites
    grs["distance_to_site"] = nearest_dists.astype(float)

    # drop rows beyond max_distance_km
    before = len(grs)
    grs = grs[grs["distance_to_site"] <= max_distance_km].reset_index(drop=True)
    if debug:
        print(f"[load_grs_tracks] dropped {before - len(grs)} rows with distance > {max_distance_km} km; remaining {len(grs)}")
    if grs.empty:
        if debug:
            print("[load_grs_tracks] no rows left after distance filtering")
        return grs

    # VERY FAST vectorized split for multi-site storms
    grs['_orig_storm_number'] = grs['_orig_storm_number'].astype(int)
    site_counts = grs.groupby('_orig_storm_number')['radar_site'].nunique()
    multi_mask_series = grs['_orig_storm_number'].map(site_counts) > 1

    # For rows that need splitting (multi-site storms), factorize (orig, site) pairs and assign new ids
    mask_multi = multi_mask_series.fillna(False).values  # boolean numpy array
    if mask_multi.any():
        # factorize the pair tuples for only those rows
        pairs = list(zip(grs.loc[mask_multi, '_orig_storm_number'].values,
                         grs.loc[mask_multi, 'radar_site'].values))
        codes, uniques = pd.factorize(pairs)
        max_orig = int(grs['_orig_storm_number'].max())
        next_id = max_orig + 1
        # map codes to new storm ids (vectorized)
        new_ids_for_multi = next_id + codes
        # create an array for storm_id
        storm_id_arr = grs['_orig_storm_number'].values.copy().astype(int)
        storm_id_arr[mask_multi] = new_ids_for_multi
        grs['storm_id'] = storm_id_arr.astype(int)
    else:
        # no splits needed, preserve original numbers
        grs['storm_id'] = grs['_orig_storm_number'].values.astype(int)

    # drop storm groups with fewer than min_rows rows
    counts = grs.groupby('storm_id').size()
    keep_ids = counts[counts >= min_rows].index.tolist()
    before2 = len(grs)
    grs = grs[grs['storm_id'].isin(keep_ids)].reset_index(drop=True)
    if debug:
        dropped = before2 - len(grs)
        print(f"[load_grs_tracks] dropped {dropped} rows belonging to storm groups with < {min_rows} rows")

    # remove original storm-like columns and helper column
    storm_like = [c for c in grs.columns if re.match(r'^\s*storm(?:\s*number)?\s*$', c, flags=re.I)]
    if storm_like:
        if debug:
            print(f"[load_grs_tracks] dropping original storm-like columns: {storm_like}")
        grs = grs.drop(columns=storm_like, errors='ignore')
    if '_orig_storm_number' in grs.columns:
        grs = grs.drop(columns=['_orig_storm_number'], errors='ignore')


    # snake_case all column names
    def _snake_case(name: str) -> str:
        s = str(name).strip()
        s = re.sub(r'[^0-9a-zA-Z]+', '_', s)
        s = re.sub(r'__+', '_', s)
        s = s.strip('_').lower()
        return s or name
    grs.columns = [_snake_case(c) for c in grs.columns]

    # final ordering: keep time/latitude/longitude/storm_id/radar_site/distance_to_site up front when present
    front = []
    for k in ("time", "latitude", "longitude", "storm_id", "radar_site", "distance_to_site"):
        if k in grs.columns:
            front.append(k)
    rest = [c for c in grs.columns if c not in front]
    grs = grs[front + rest]


    ######################################################### FILTER DATAFRAME FOR DISCONTINUITIES #########################################################


    # Continuity & splitting:
    # 1) If NO cluster in a storm covers ≥50% of that storm's rows → drop the entire storm.
    # 2) Otherwise (there IS a majority cluster), keep every cluster with size ≥ min_rows:
    #       - majority cluster keeps original storm_id
    #       - each other kept cluster becomes a new storm_id
    #    Clusters with size < min_rows are dropped.
    if ("storm_id" in grs.columns) and ("time" in grs.columns):
        t = pd.to_datetime(grs["time"], utc=True, errors="coerce")

        # Sort by (storm_id, time) to compute per-storm diffs and cluster ids
        order = np.lexsort((t.view(np.int64), grs["storm_id"].to_numpy()))
        grs_sorted = grs.iloc[order].copy()
        t_sorted = t.iloc[order]

        dts = t_sorted.groupby(grs_sorted["storm_id"]).diff()
        gap = pd.to_timedelta(max_gap_hours, unit="h")
        new_cluster = dts.isna() | (dts > gap)
        cluster_id = new_cluster.groupby(grs_sorted["storm_id"]).cumsum().astype(np.int32)

        # Cluster sizes and group sizes  (version-agnostic column naming)
        cluster_sizes = (
            grs_sorted
            .groupby(["storm_id", cluster_id], sort=False)
            .size()
            .reset_index(name="cluster_size")
        )
        # The second column is the one created from 'cluster_id' Series; rename by position.
        # (Works regardless of whether pandas called it 'level_1', 0, or anything else.)
        second_col = cluster_sizes.columns[1]
        if second_col != "cluster_id":
            cluster_sizes = cluster_sizes.rename(columns={second_col: "cluster_id"})

        group_sizes = (
            grs_sorted.groupby("storm_id").size().rename("group_size").reset_index()
        )

        # Largest cluster per storm (majority check) using normalized name
        largest = (
            cluster_sizes.sort_values(["storm_id", "cluster_size"], ascending=[True, False])
            .groupby("storm_id", sort=False)
            .head(1)[["storm_id", "cluster_id", "cluster_size"]]
            .rename(columns={"cluster_id": "best_cluster_id", "cluster_size": "best_size"})
        )

        meta = (
            cluster_sizes
            .merge(group_sizes, on="storm_id", how="left")
            .merge(largest, on="storm_id", how="left")
        )
        meta["best_ratio"] = meta["best_size"] / meta["group_size"]

        # Decide which storms survive (must have a majority cluster ≥ 50%)
        storms_have_majority = (
            meta[["storm_id", "best_ratio"]].drop_duplicates().set_index("storm_id")["best_ratio"] >= 0.5
        )

        # Keep clusters only for storms that have a majority; among those, keep clusters with size ≥ min_rows
        meta["keep_cluster"] = storms_have_majority.reindex(meta["storm_id"]).to_numpy() & (meta["cluster_size"] >= min_rows)

        # Build mapping (storm_id, cluster_id) -> new_storm_id for kept clusters
        next_id = int(grs["storm_id"].max()) + 1
        mapping_rows = []
        for sid, sub in meta.groupby("storm_id", sort=False):
            has_majority = bool(storms_have_majority.get(sid, False))
            if not has_majority:
                # drop the entire storm (no mapping rows created)
                continue

            # clusters that pass size threshold
            kept = sub.loc[sub["keep_cluster"]]
            if kept.empty:
                # even majority cluster was < min_rows → drop this storm entirely
                continue

            # Identify majority cluster for this storm (one row in 'largest')
            best_cid = int(sub["best_cluster_id"].iloc[0])

            # Majority cluster keeps original storm_id if it's kept; otherwise the storm is dropped
            if (kept["cluster_id"] == best_cid).any():
                mapping_rows.append({"storm_id": sid, "cluster_id": best_cid, "new_storm_id": sid})
            else:
                # No kept majority cluster → drop storm entirely
                continue

            # All other kept clusters → assign fresh ids
            others = kept[kept["cluster_id"] != best_cid]
            for _, r in others.iterrows():
                mapping_rows.append({"storm_id": sid, "cluster_id": int(r["cluster_id"]), "new_storm_id": next_id})
                next_id += 1

        if mapping_rows:
            map_df = pd.DataFrame(mapping_rows)

            # Attach (storm_id, cluster_id) for each row, then merge to new_storm_id
            key = pd.DataFrame({
                "storm_id": grs_sorted["storm_id"].to_numpy(),
                "cluster_id": cluster_id.to_numpy()
            })
            key = key.merge(map_df, on=["storm_id", "cluster_id"], how="left")

            # Rows not mapped are dropped
            keep_mask_sorted = key["new_storm_id"].notna().to_numpy()
            new_ids_sorted = key["new_storm_id"].fillna(-1).astype(np.int64).to_numpy()

            # Map to original order
            keep_mask = np.empty_like(keep_mask_sorted, dtype=bool)
            keep_mask[order] = keep_mask_sorted
            new_ids = np.empty_like(new_ids_sorted)
            new_ids[order] = new_ids_sorted

            before_cont = len(grs)
            grs = grs.loc[keep_mask].reset_index(drop=True)
            grs["storm_id"] = new_ids[keep_mask]

            if debug:
                dropped = before_cont - len(grs)
                print(f"[load_grs_tracks] continuity filter (max_gap_hours={max_gap_hours}): "
                      f"dropped {dropped} rows; kept clusters with size ≥ {min_rows} "
                      f"and split non-major clusters into new storm_ids; "
                      f"dropped storms with no ≥50% majority cluster")
        else:
            # No storms passed the rules → empty result
            if debug:
                print(f"[load_grs_tracks] continuity filter: no storms had a ≥50% majority cluster; dropping all")
            grs = grs.iloc[0:0].copy().reset_index(drop=True)
    else:
        if debug:
            print("[load_grs_tracks] continuity filter skipped (missing 'storm_id' or 'time')")


    ######################################################################################################################################################


    # save processed cache
    try:
        grs.to_parquet(processed_parquet, index=False)
        if debug:
            print(f"[load_grs_tracks] wrote processed cache {processed_parquet}")
    except Exception as e:
        try:
            grs.to_csv(processed_csv, index=False)
            if debug:
                print(f"[load_grs_tracks] parquet save failed; saved CSV {processed_csv}: {e}")
        except Exception as ee:
            if debug:
                print(f"[load_grs_tracks] failed to save processed cache: {ee}")

    return grs
