

def link_obs_data(files,
                  lsr_df,
                  spc_df,
                  debug,
                  cache_dir="Datasets/surface_obs_datasets/linked_obs_cache",
):
    """
    files: List of (lvl2_key, CompositeReflectivity) tuples
    lsr_df: pre-loaded LSR DataFrame
    debug:  whether to print/display intermediate results

    - For each radar scan, create dataframes containing LSR, synoptic, and spc data, as well as cell metadata
    - Repeat for all cells in the radar scan
    - Return a DataFrame containing observations for all cells over all radar scans
    """
    # Store all storm cell dataframes for all scans
    all_records = []

    # For each radar scan, create dataframes containing LSR and synoptic data + cell metadata, for all cells in the radar scan
    for radar_file_name, comp_scan in files:
        # build simple cache path (based on radar_file_name)
        safe_name = re.sub(r'[^A-Za-z0-9._-]', '_', radar_file_name)
        cache_path = os.path.join(cache_dir, f"{safe_name}.csv")

        # if cached, load and continue
        if os.path.exists(cache_path):
            if debug:
                print(f"[link_obs_data] Loading cached scan for {radar_file_name} -> {cache_path}")
            try:
                scan_df = pd.read_csv(cache_path)

                if 'scan_time' in scan_df.columns:
                    scan_df['scan_time'] = pd.to_datetime(scan_df['scan_time'], utc=True, errors='coerce')
                all_records.append(scan_df)
                continue
            except Exception as e:
                print(f"[link_obs_data] WARNING: failed to read cache {cache_path} ({e}), will recompute scan.")


        ############################################# IF CACHE DOESN'T EXIST, PROCEED WITH SURFACE-OBS LOGIC #####################################################


        # Extract scan_time from the radar_scan.time field.
        raw_times = comp_scan.time['data']             # e.g. array([1625694000.0, …])
        time_units = comp_scan.time['units']           # e.g. "seconds since 1970-01-01T00:00:00Z"
        calendar = getattr(comp_scan.time, 'calendar', 'standard')

        # Convert the first sweep’s time to a datetime
        scan_time = num2date(raw_times[0], time_units, calendar)

        if debug:
            print(f"[link_obs_data] No cached scan found, processing normally for {radar_file_name} @ {scan_time}")

        # Get cells from the Level-III composite (where each cell is an ID, center_lat, center_lon, bounding_lat [tuple], bounding_lon [tuple])
        cells, cell_fig = get_cell_centers(
            comp_scan=comp_scan,
            sweep=0,
            class_field='reflectivity',
            threshold=35,
            min_size=500,
            debug=debug
        )

        # Store per-scan dataframes from individual cells
        scan_records = []

        # Build observational data for each cell
        for cell_id, center_lat, center_lon, bounding_lat, bounding_lon in cells:
            # Get obs data from synoptic
            df_synoptic = filter_synoptic(
                bounding_lat=bounding_lat,
                bounding_lon=bounding_lon,
                center_lat=center_lat,
                center_lon=center_lon,
                scan_time=scan_time,
                time_window=timedelta(minutes=5),
                debug=debug
            )
            if debug:
                print(f"\n synoptic reports for the cell {cell_id} in the radar_file_name {radar_file_name} at the time {scan_time}: \n")
                display(df_synoptic)

            # Get obs data from lsr
            df_lsr = filter_lsr(
                lsr_df=lsr_df,
                bounding_lat=bounding_lat,
                bounding_lon=bounding_lon,
                center_lat=center_lat,
                center_lon=center_lon,
                scan_time=scan_time,
                debug=debug
            )
            if debug:
                print(f"\n lsr reports for the cell {cell_id} in the radar_file_name {radar_file_name} at the time {scan_time}: \n")
                display(df_lsr)

            df_spc = filter_spc(
                spc_df=spc_df,
                bounding_lat=bounding_lat,
                bounding_lon=bounding_lon,
                center_lat=center_lat,
                center_lon=center_lon,
                scan_time=scan_time,
            )
            if debug:
                print(f"\n spc reports for the cell {cell_id} in the radar_file_name {radar_file_name} at the time {scan_time}: \n")
                display(df_spc)

            # For each cell, concatenate the three dataframes observational dataframes
            df_cell = pd.concat([df_synoptic, df_lsr, df_spc], ignore_index=True)

            # If no observations, skip and move on to next cell
            if df_cell.empty:
                continue

            # If not empty, then add columns containing cell metadata
            df_cell['radar_file_name'] = radar_file_name
            df_cell['scan_time'] = scan_time
            df_cell['cell_id']   = cell_id
            df_cell['cell_lat']  = center_lat
            df_cell['cell_lon']  = center_lon

            # Unpack tuples so operations are easier later
            min_lat, max_lat = bounding_lat
            min_lon, max_lon = bounding_lon

            df_cell['bounding_lat_min'] = min_lat
            df_cell['bounding_lat_max'] = max_lat
            df_cell['bounding_lon_min'] = min_lon
            df_cell['bounding_lon_max'] = max_lon


            # Add each cell's dataframe into scan_records
            scan_records.append(df_cell)

            if debug:
                print(f"\n [link_obs_data] Complete dataframe for the cell {cell_id} in the radar_file_name {radar_file_name}: \n")
                display(df_cell)
                print(f"\n [link_obs_data] Overlayed observation data for cell {cell_id}: ")
                _plot_observations_on_fig(cell_fig, df_cell, lon_col='station_lon', lat_col='station_lat', ms=50, alpha=0.85)
                section_seperator(4)


        # Write scan_records to cache
        if not scan_records:
            scan_df = pd.DataFrame(columns=[
                "source", "time", "station_lat", "station_lon", "gust", "distance_km",
                "radar_file_name", "scan_time", "cell_id", "cell_lat", "cell_lon",
                "bounding_lat_min", "bounding_lat_max", "bounding_lon_min", "bounding_lon_max"
            ])
            if debug:
                print(f"[link_obs_data] No obs for scan {radar_file_name}; will cache empty dataframe.")
        else:
            scan_df = pd.concat(scan_records, ignore_index=True)

        try:
            tmp_path = cache_path + ".tmp"

            if 'scan_time' in scan_df.columns:
                scan_df['scan_time'] = pd.to_datetime(scan_df['scan_time'], utc=True, errors='coerce')
            scan_df.to_csv(tmp_path, index=False)
            os.replace(tmp_path, cache_path)

            if debug:
                print(f"[link_obs_data] Wrote cache for {radar_file_name} -> {cache_path}")
                section_seperator(4)

        except Exception as e:
            print(f"[link_obs_data] WARNING: failed to write cache ({e})")
        all_records.append(scan_df)


    # Finally, handle all_records after looping through all radar scans
    if not all_records:
        # Return an empty DataFrame with expected columns
        full = pd.DataFrame(columns=[
            # Obs columns
            "source",
            "time",
            "station_lat",
            "station_lon",
            "gust",
            "obs_distance",

            # Cell metadata
            "radar_file_name",
            "scan_time",
            "cell_id",
            "cell_lat",     # Where cell_lat and cell_lon are columns for center of cell
            "cell_lon",
            "bounding_lat_min", # The bounds of the cell (use for cropping our level 2 data later)
            "bounding_lat_max",
            "bounding_lon_min",
            "bounding_lon_max"
        ])
    else:
        full = pd.concat(all_records, ignore_index=True)
    return full


def _plot_observations_on_fig(
    fig,
    obs_df,
    lon_col='station_lon',
    lat_col='station_lat',
    marker='o',
    ms=40,
    alpha=0.9,
    edgecolor='k',
    zorder=6):
    """
    Robust overlay for Colab:
     - prints debug info (how many obs will be plotted)
     - plots points on the first axis of `fig`
     - calls ax.relim()/ax.autoscale_view() to ensure points are visible
     - displays the updated figure via IPython.display (works reliably in Colab)
    """
    # Safety guards
    if fig is None:
        print("[_plot_observations_on_fig] got fig=None -> skipping overlay")
        return

    if obs_df is None or obs_df.empty:
        print("[_plot_observations_on_fig] obs_df empty -> showing base figure")
        ipy_display(fig)
        return

    # get or create axis
    ax = fig.axes[0] if fig.axes else fig.add_subplot(111)

    # make sure we have numeric lon/lat
    lons = pd.to_numeric(obs_df[lon_col], errors='coerce')
    lats = pd.to_numeric(obs_df[lat_col], errors='coerce')
    valid = lons.notna() & lats.notna()
    n_valid = int(valid.sum())

    print(f"[_plot_observations_on_fig] plotting {n_valid} observation(s) (from {len(obs_df)} rows)")

    if n_valid == 0:
        # show the base figure so you still get the blob-only output
        ipy_display(fig)
        return

    # scatter them on the axis
    sc = ax.scatter(
        lons[valid], lats[valid],
        s=ms, marker=marker, alpha=alpha,
        edgecolors=edgecolor, linewidths=0.5,
        zorder=zorder, label='obs'
    )

    # Ensure new points fall inside the visible area:
    try:
        # recompute data limits and autoscale the view
        ax.relim()
        ax.autoscale_view()

        # optionally pad the limits a little (5%)
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        padx = 0.05 * (xmax - xmin) if (xmax - xmin) != 0 else 0.01
        pady = 0.05 * (ymax - ymin) if (ymax - ymin) != 0 else 0.01
        ax.set_xlim(xmin - padx, xmax + padx)
        ax.set_ylim(ymin - pady, ymax + pady)
    except Exception as e:
        # non-fatal; continue to display figure
        print(f"[_plot_observations_on_fig] autoscale failed: {e}")

    # tidy legend (dedupe)
    try:
        handles, labels = ax.get_legend_handles_labels()
        uniq = {}
        for h, l in zip(handles, labels):
            if l not in uniq:
                uniq[l] = h
        if uniq:
            ax.legend(list(uniq.values()), list(uniq.keys()), fontsize='small', loc='best')
    except Exception:
        pass

    # show the updated figure in Colab
    ipy_display(fig)
    # also call plt.pause(0.001) to flush (usually not necessary in Colab but harmless)
    plt.pause(0.001)


