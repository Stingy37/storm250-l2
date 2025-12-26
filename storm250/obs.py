

######################################################################## FILTER LSR DATA ###############################################################################


def load_raw_lsr(start: Union[date, datetime],
                 end:   Union[date, datetime],
                 debug: bool = False,
                 cache_dir: Union[str, Path] = "Datasets/surface_obs_datasets/lsr_reports",
                 force_refresh: bool = False) -> pd.DataFrame:
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
      - remarks (text comments, if any)
      - type (the original TYPETEXT)
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Coerce start/end to datetimes (treat date inputs as full-day ranges) ---
    if isinstance(start, date) and not isinstance(start, datetime):
        start_dt = datetime.combine(start, time.min)
    else:
        start_dt = start

    if isinstance(end, date) and not isinstance(end, datetime):
        # Treat end date as inclusive through the end of day
        end_dt = datetime.combine(end, time(23, 59, 59))
    else:
        end_dt = end

    # Build cache filename using only YMD (matches your requested format)
    cache_fname = cache_dir / f"{start_dt:%Y%m%d}_{end_dt:%Y%m%d}.csv"

    # If present and not forcing refresh, try to load cache
    if cache_fname.exists() and not force_refresh:
        if debug:
            print(f"Loading LSRs from cache: {cache_fname}")
        try:
            cached = pd.read_csv(cache_fname, parse_dates=['time'], engine='python')
            # ensure columns are present and return subset to be safe
            cols = ['time', 'lat', 'lon', 'gust', 'type']
            missing = [c for c in cols if c not in cached.columns]
            if missing:
                raise ValueError(f"Cached file missing columns: {missing}")
            return cached[cols]
        except Exception as e:
            if debug:
                print(f"Failed to read cache ({cache_fname}): {e}. Removing and refetching.")
            try:
                cache_fname.unlink()
            except Exception:
                pass
            # fall through to re-fetch

    # Fetch from IEM if cache not used ---
    base_url = "https://mesonet.agron.iastate.edu/cgi-bin/request/gis/lsr.py"

    # Format start/end as ISO8601 with trailing Z (include +1 sec to make end inclusive)
    sts = start_dt.strftime('%Y-%m-%dT%H:%MZ')
    ets = (end_dt + timedelta(seconds=1)).strftime('%Y-%m-%dT%H:%MZ')

    params = {
        'wfo': 'ALL',
        'sts': sts,
        'ets': ets,
        'fmt': 'csv'
    }

    if debug:
        print(f"\n Fetching LSRs from {sts} to {ets} with params {params} \n")

    resp = requests.get(base_url, params=params)
    resp.raise_for_status()
    if debug:
        print("Fetch HTTP", resp.status_code, "| Response snippet:",
              resp.text[:200].replace('\n',' '))

    # Read into pandas via StringIO
    df = pd.read_csv(
        StringIO(resp.text),
        parse_dates=['VALID'],
        engine='python',
        on_bad_lines='skip'
    )
    if debug:
        print(f"{len(df)} total reports before filtering")

    # Filter for thunderstorm-wind events with MG or EG
    if debug:
        display(df['TYPETEXT'].value_counts())
    keep_codes = ['tstm wind',
                  'tstm wnd gst',
                  'non-tstm wnd gst',
                  'marine tstm wind',
                  ]
    df = df[df['TYPETEXT'].str.lower().isin(keep_codes)]
    df = df.reset_index(drop=True)
    if debug:
        print(f" {len(df)} total reports remaining after TYPETEXT filtering")

    # Rename & select
    df = df.rename(columns={
        'VALID':  'time',
        'LAT':    'lat',
        'LON':    'lon',
        'MAG':    'gust',
        'REMARK': 'remarks',
        'TYPETEXT': 'type'
    })

    # coerce gust to numeric then drop NaNs in critical fields
    df['gust'] = pd.to_numeric(df['gust'], errors='coerce')
    df = df.dropna(subset=['time','lat','lon','gust']).reset_index(drop=True)
    if debug:
        print(f" {len(df)} total reports after dropping NaNs in critical fields")

    # keep only needed columns (and preserve order)
    out = df[['time', 'lat', 'lon', 'gust', 'type']].copy()

    # Save to cache atomically ---
    try:
        # write to a temporary file then atomically replace
        with tempfile.NamedTemporaryFile(delete=False, dir=str(cache_dir), suffix='.csv') as tmp:
            tmp_name = Path(tmp.name)
        out.to_csv(tmp_name, index=False)
        tmp_name.replace(cache_fname)
        if debug:
            print(f"Saved LSR cache to {cache_fname}")
    except Exception as e:
        if debug:
            print("Warning: failed to write cache:", e)

    return out



def filter_lsr(lsr_df,
               bounding_lat,
               bounding_lon,
               center_lat,
               center_lon,
               scan_time,
               time_window=timedelta(minutes=5),
               debug=True):
    """
    - Filters pre-loaded lsr_df to time_window & bbox
    - bounding_lat = (min_lat, max_lat)
    - bounding_lon = (min_lon, max_lon)
    Returns DataFrame of {source, time, station_lat, station_lon, gust, obs_distance}
    """
    min_lat, max_lat = bounding_lat
    min_lon, max_lon = bounding_lon

    if debug:
        print(f"[filter_lsr] bounding_lat=({min_lat},{max_lat}), bounding_lon=({min_lon},{max_lon})")
        print(f"[filter_lsr] scan_time={scan_time}, window=±{time_window}")

    # normalize scan_time to python datetime
    if hasattr(scan_time, 'strftime') and not isinstance(scan_time, pd.Timestamp):
        try:
            scan_time_dt = datetime(
                scan_time.year, scan_time.month, scan_time.day,
                scan_time.hour, scan_time.minute, scan_time.second
            )
            if debug:
                print(f"[filter_lsr] converted scan_time to datetime: {scan_time_dt}")
        except Exception:
            scan_time_dt = pd.to_datetime(str(scan_time))
            if debug:
                print(f"[filter_lsr] fallback converted scan_time via to_datetime: {scan_time_dt}")
    else:
        scan_time_dt = pd.Timestamp(scan_time)
        if debug:
            print(f"[filter_lsr] scan_time is Timestamp: {scan_time_dt}")

    # copy dataframe
    df = lsr_df.copy()

    # Normalize time column to pandas Timestamps
    df['time'] = pd.to_datetime(df['time'])
    if debug:
        print(f"[filter_lsr] coerced time dtype = {df['time'].dtype}")

    # Build pandas Timestamps for the window
    start_ts = pd.Timestamp(scan_time_dt) - time_window
    end_ts   = pd.Timestamp(scan_time_dt) + time_window
    if debug:
        print(f"[filter_lsr] time window start={start_ts}, end={end_ts}")

    # Time filter
    df = df[df['time'].between(start_ts, end_ts)]
    if debug:
        print(f"[filter_lsr] after time filter: {len(df)} records")


    # spatial filter using bounding box
    df = df[(df['lat'] >= min_lat) & (df['lat'] <= max_lat) &
            (df['lon'] >= min_lon) & (df['lon'] <= max_lon)]
    if debug:
        print(f"[filter_lsr] records after spatial filter: {len(df)}")

    # compute distance from cell center
    df['obs_distance'] = df.apply(
        lambda r: haversine(center_lon, center_lat, r['lon'], r['lat']),
        axis=1
    )

    # rename station coordinates
    df = df.rename(columns={'lat': 'station_lat', 'lon': 'station_lon'})
    df['source'] = 'lsr_iastate'

    # select and order columns to match filter_synoptic
    result = df[['source', 'time', 'station_lat', 'station_lon', 'gust', 'obs_distance']]
    if debug:
        print(f"[filter_lsr] final returned records: {len(result)}")

    return result


##################################################################################################################################################################



def load_raw_spc(start: datetime,
                 end:   datetime,
                 spc_dir: str = "Datasets/surface_obs_datasets/spc_reports",
                 debug: bool = False) -> pd.DataFrame:
    """
    Load SPC per-year 'wind' CSV(s) from a folder and return a DataFrame with:
      - time (python/pandas datetime)
      - lat, lon (averaged from slat/elat and slon/elon)
      - gust (float; from `mag`)
      - type (string; 'spc_wind')

    Parameters:
    - start, end: datetime objects (inclusive range)
    - spc_dir: path to folder containing YYYY_wind.csv files. Can be relative to
               current working dir or a path under /content/drive/My Drive/...
    - debug: enable debug printing
    """
    # Build candidate folder paths to try (helps with Colab Drive mounts)
    candidates = [spc_dir,
                  os.path.join("/content/drive/My Drive", spc_dir),
                  os.path.join("/content/drive/MyDrive", spc_dir),
                  os.path.expanduser(spc_dir)]
    folder = None
    for p in candidates:
        if p and os.path.isdir(p):
            folder = p
            break

    if folder is None:
        raise FileNotFoundError(
            f"Could not find SPC folder. Tried: {candidates}. "
            "Make sure your Drive is mounted and the folder exists."
        )

    if debug:
        print(f"[load_raw_spc] using folder: {folder}")

    # Determine which year files we need
    years = list(range(start.year, end.year + 1))
    if debug:
        print(f"[load_raw_spc] loading years: {years}")

    dfs = []
    for y in years:
        fname = f"{y}_wind.csv"
        fpath = os.path.join(folder, fname)
        if os.path.isfile(fpath):
            if debug:
                print(f"[load_raw_spc] reading {fpath}")
            # read with pandas
            try:
                dfi = pd.read_csv(fpath, low_memory=False)
                dfi['__source_file'] = fname
                dfs.append(dfi)
            except Exception as e:
                if debug:
                    print(f"[load_raw_spc] failed to read {fpath}: {e}")
        else:
            if debug:
                print(f"[load_raw_spc] file not found: {fpath} (skipping)")

    if not dfs:
        raise FileNotFoundError(f"No SPC wind CSVs found in {folder} for years {years}.")

    df = pd.concat(dfs, ignore_index=True, sort=False)
    if debug:
        print(f"[load_raw_spc] concatenated rows: {len(df)}")

    # Build time column from `date` + `time` (both expected in file)
    # Examples in file: date='2024-12-29', time='03:45:00'
    df['time'] = pd.to_datetime(df['date'].astype(str).str.strip() + ' ' +
                                df['time'].astype(str).str.strip(),
                                errors='coerce')

    if debug:
        n_bad_time = df['time'].isna().sum()
        print(f"[load_raw_spc] parsed time; {n_bad_time} rows failed to parse time")

    # Compute lat/lon as the mean of start/end lat/lon
    # tolerantly coerce columns to numeric
    for c in ['slat','elat','slon','elon','mag']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        else:
            df[c] = np.nan

    df['lat'] = df[['slat','elat']].mean(axis=1)
    df['lon'] = df[['slon','elon']].mean(axis=1)

    # gust from mag
    df['gust'] = df['mag']

    if debug:
        print(f"[load_raw_spc] after coordinate/gust computation: rows={len(df)}")
        print(f"  lat/nulls={df['lat'].isna().sum()}, lon/nulls={df['lon'].isna().sum()}, gust/nulls={df['gust'].isna().sum()}")

    # drop rows missing critical fields
    before = len(df)
    df = df.dropna(subset=['time','lat','lon','gust']).reset_index(drop=True)
    after = len(df)
    if debug:
        print(f"[load_raw_spc] dropped {before-after} rows missing time/lat/lon/gust -> {len(df)} remaining")

    # Keep / rename columns to match your LSR loader
    df_out = pd.DataFrame({
        'time': df['time'],
        'lat': df['lat'],
        'lon': df['lon'],
        'gust': df['gust'].astype(float),
        'type': 'spc_wind'
    })

    return df_out[['time','lat','lon','gust','type']]


def filter_spc(spc_df,
               bounding_lat,
               bounding_lon,
               center_lat,
               center_lon,
               scan_time,
               time_window=timedelta(minutes=5),
               debug=True):
    """
    Filters pre-loaded spc_df to time_window & bbox and returns:
      ['source','time','station_lat','station_lon','gust','obs_distance']

    - bounding_lat = (min_lat, max_lat)
    - bounding_lon = (min_lon, max_lon)
    - scan_time may be datetime or pandas Timestamp
    """
    min_lat, max_lat = bounding_lat
    min_lon, max_lon = bounding_lon

    if debug:
        print(f"[filter_spc] bounding_lat=({min_lat},{max_lat}), bounding_lon=({min_lon},{max_lon})")
        print(f"[filter_spc] scan_time={scan_time}, window=±{time_window}")

    # Normalize scan_time to a naive datetime
    if hasattr(scan_time, 'strftime') and not isinstance(scan_time, pd.Timestamp):
        try:
            scan_time_dt = datetime(
                scan_time.year, scan_time.month, scan_time.day,
                scan_time.hour, scan_time.minute, scan_time.second
            )
            if debug:
                print(f"[filter_spc] converted scan_time to datetime: {scan_time_dt}")
        except Exception:
            scan_time_dt = pd.to_datetime(str(scan_time))
            if debug:
                print(f"[filter_spc] fallback converted scan_time via to_datetime: {scan_time_dt}")
    else:
        scan_time_dt = pd.Timestamp(scan_time)
        if debug:
            print(f"[filter_spc] scan_time is Timestamp: {scan_time_dt}")

    # Work on a copy
    df = spc_df.copy()

    # Normalise and ensure time is datetime
    df['time'] = pd.to_datetime(df['time'])
    if debug:
        print(f"[filter_spc] coerced time dtype = {df['time'].dtype}")

    start_ts = pd.Timestamp(scan_time_dt) - time_window
    end_ts   = pd.Timestamp(scan_time_dt) + time_window
    if debug:
        print(f"[filter_spc] time window start={start_ts}, end={end_ts}")

    # Time filter
    df = df[df['time'].between(start_ts, end_ts)]
    if debug:
        print(f"[filter_spc] after time filter: {len(df)} records")

    # Spatial filter using bounding box
    df = df[(df['lat'] >= min_lat) & (df['lat'] <= max_lat) &
            (df['lon'] >= min_lon) & (df['lon'] <= max_lon)]
    if debug:
        print(f"[filter_spc] records after spatial filter: {len(df)}")

    # compute distance from cell center (center_lon, center_lat)
    df['obs_distance'] = df.apply(
        lambda r: haversine(center_lon, center_lat, r['lon'], r['lat']),
        axis=1
    )

    # rename station coordinates
    df = df.rename(columns={'lat': 'station_lat', 'lon': 'station_lon'})
    df['source'] = 'spc'

    result = df[['source', 'time', 'station_lat', 'station_lon', 'gust', 'obs_distance']].reset_index(drop=True)
    if debug:
        print(f"[filter_spc] final returned records: {len(result)}")

    return result
