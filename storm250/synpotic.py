
def filter_synoptic(bounding_lat,
                    bounding_lon,
                    center_lat,
                    center_lon,
                    scan_time,
                    time_window=timedelta(minutes=5),
                    debug=True):
    """
    - bounding_lat = (min_lat, max_lat)
    - bounding_lon = (min_lon, max_lon)
    - scan_time in UTC
    Returns DataFrame of {source, time, lat, lon, gust, obs_distance}
    """
    min_lat, max_lat = bounding_lat
    min_lon, max_lon = bounding_lon

    if debug:
        print(f"[filter_synoptic] bbox lat=({min_lat},{max_lat}), lon=({min_lon},{max_lon})")
        print(f"[filter_synoptic] scan_time={scan_time.isoformat()} window=±{time_window}")

    # Build bounding box string param
    bbox = f"{min_lon},{min_lat},{max_lon},{max_lat}"
    if debug:
        print(f"[filter_synoptic] bbox string = '{bbox}'")

    # Format time window in UTC as YYYYmmddHHMM
    start = (scan_time - time_window).strftime("%Y%m%d%H%M")
    end   = (scan_time + time_window).strftime("%Y%m%d%H%M")
    if debug:
        print(f"[filter_synoptic] time window start={start}, end={end}")

    # Request the time series
    url = "https://api.synopticdata.com/v2/stations/timeseries"
    params = {
      "token": SYNOPTIC_TOKEN,
      "bbox":  bbox,
      "start": start,
      "end":   end,
      "vars":   "wind_gust",

      # ASOS/AWOS -> High Quality, airport data, sparse  |  ID = 1
      # CWOP/APRSWXNET -> Individual stations, dense     |  ID = 65
      # RAWS -> Off grid states, dense                   |  ID = 2
      # HADS ->                                          |  ID = 106
      # [Add more networks as needed]
      "network": "1, 65, 106, 2",

      "qc":     "on",
      "showemptystations": 1,
      "units":  "english"
    }

    if debug:
        print(f"[filter_synoptic] GET {url} with params={params}")
    r = requests.get(url, params=params)
    r.raise_for_status()
    resp = r.json()


    if debug:
        # print top-level summary
        print("SUMMARY:", resp.get("SUMMARY", {}))

        # print how many stations
        n = len(resp.get("STATION", []))
        print("STATION count:", n)



    if debug:
        n_stations = len(resp.get("STATION", []))
        print(f"[filter_synoptic] received data for {n_stations} stations")

    # Parse out each station’s observations
    recs = []
    for st in resp.get("STATION", []):
        st_lat = float(st.get("LATITUDE"))
        st_lon = float(st.get("LONGITUDE"))
        obs = st.get("OBSERVATIONS", {})

        # detect network id from common station fields (MNET_ID seen in dumps)
        network_id = (
            st.get("MNET_ID")
            or st.get("MNET")
            or st.get("NETWORK")
            or st.get("NETWORK_ID")
        )
        # normalize to simple string (or None)
        network_id = str(network_id) if network_id not in (None, "", "None") else None

        # build source label like "synoptic_1" or fallback "synoptic"
        src_label = f"synoptic_{network_id}" if network_id else "synoptic"

        # Prefer wind_gust_set_1, fall back to wind_gust
        gusts = obs.get("wind_gust_set_1") or obs.get("wind_gust") or []

        # Try several possible time arrays, then fall back to date_time
        times = (
            obs.get("wind_gust_time_utc_1")
            or obs.get("wind_gust_time_utc")
            or obs.get("wind_gust_time")
            or obs.get("date_time")
            or []
        )

        if debug:
            print(f"[filter_synoptic] Station {st['STID']} @ ({st_lat},{st_lon}): gusts={len(gusts)}, times={len(times)}")

        # zip will iterate up to the shortest list; that's fine
        for t_str, g in zip(times, gusts):
            # skip missing gusts
            if g is None:
                continue

            # parse timestamp
            try:
                t = datetime.strptime(t_str, "%Y-%m-%dT%H:%M:%SZ")
            except Exception:
                if debug:
                    print(f"[filter_synoptic] skipping bad timestamp '{t_str}' for {st['STID']}")
                continue

            # ensure numeric gust
            try:
                g_val = float(g)
            except Exception:
                if debug:
                    print(f"[filter_synoptic] skipping non-numeric gust '{g}' for {st['STID']}")
                continue

            dist = haversine(st_lon, st_lat, center_lon, center_lat)

            # NOTE -> this must be aligned with dataframe returned by filter_lsr
            recs.append({
                "source":      src_label,
                "time":        t,
                "station_lat": st_lat,
                "station_lon": st_lon,
                "gust":        g_val,
                "obs_distance": dist
            })

    df = pd.DataFrame(recs)
    if debug:
        print(f"[filter_synoptic] total records = {len(df)}")
    return df
