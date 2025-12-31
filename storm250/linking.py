from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from netCDF4 import num2date
from typing import List, Tuple

import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)


def link_obs_data(
    files,
    lsr_df,
    spc_df,
    debug,
    cache_dir="Datasets/surface_obs_datasets/linked_obs_cache",
):
    """
    files: List of (lvl2_key, CompositeReflectivity) tuples
    lsr_df: pre-loaded LSR DataFrame
    spc_df: pre-loaded SPC DataFrame
    debug:  whether to log/display intermediate results

    - For each radar scan, create dataframes containing LSR, synoptic, and spc data, as well as cell metadata
    - Repeat for all cells in the radar scan
    - Return a DataFrame containing observations for all cells over all radar scans
    """
    # Local imports to keep module import lightweight and avoid circular deps
    from .obs import filter_lsr, filter_spc
    from .utils import section_seperator

    # synoptic may or may not exist in your current pipeline; handle gracefully
    try:
        from .synoptic import filter_synoptic  # type: ignore
    except Exception:
        filter_synoptic = None  # type: ignore

    # get_cell_centers location in your codebase
    try:
        from .cropping import get_cell_centers  # type: ignore
    except Exception as e:
        raise ImportError("Could not import get_cell_centers (expected in storm250.cropping).") from e

    if num2date is None:
        raise ImportError("num2date is unavailable (install netCDF4 or cftime).")

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Folder for debug plots (created per-call)
    plot_dir: Path | None = None
    if debug:
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        plot_dir = Path("Datasets") / "Logs" / "plots" / f"overlay_obs_data_{ts}"
        plot_dir.mkdir(parents=True, exist_ok=True)
        logger.info("[link_obs_data] debug plot dir: %s", plot_dir)

    # Store all storm cell dataframes for all scans
    all_records = []

    # For each radar scan, create dataframes containing LSR and synoptic data + cell metadata, for all cells in the radar scan
    for radar_file_name, comp_scan in files:
        # build simple cache path (based on radar_file_name)
        safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", str(radar_file_name))
        cache_path = cache_dir / f"{safe_name}.csv"

        # if cached, load and continue
        if cache_path.exists():
            if debug:
                logger.info("[link_obs_data] Loading cached scan for %s -> %s", radar_file_name, cache_path)
            try:
                scan_df = pd.read_csv(cache_path)
                if "scan_time" in scan_df.columns:
                    scan_df["scan_time"] = pd.to_datetime(scan_df["scan_time"], utc=True, errors="coerce")
                all_records.append(scan_df)
                continue
            except Exception:
                if debug:
                    logger.exception("[link_obs_data] failed to read cache %s; will recompute scan", cache_path)

        ############################################# IF CACHE DOESN'T EXIST, PROCEED WITH SURFACE-OBS LOGIC #####################################################

        # Extract scan_time from the radar_scan.time field.
        raw_times = comp_scan.time["data"]   # e.g. array([1625694000.0, …])
        time_units = comp_scan.time["units"] # e.g. "seconds since 1970-01-01T00:00:00Z"
        calendar = getattr(comp_scan.time, "calendar", "standard")

        # Convert the first sweep’s time to a datetime
        scan_time = num2date(raw_times[0], time_units, calendar)

        if debug:
            logger.info("[link_obs_data] No cached scan found; processing %s @ %s", radar_file_name, scan_time)

        # Get cells from the Level-III composite
        cells, cell_fig = get_cell_centers(
            comp_scan=comp_scan,
            sweep=0,
            class_field="reflectivity",
            threshold=35,
            min_size=500,
            debug=debug,
        )

        # Store per-scan dataframes from individual cells
        scan_records = []

        # Build observational data for each cell
        for cell_id, center_lat, center_lon, bounding_lat, bounding_lon in cells:
            # Get obs data from synoptic (if available)
            if filter_synoptic is not None:
                df_synoptic = filter_synoptic(
                    bounding_lat=bounding_lat,
                    bounding_lon=bounding_lon,
                    center_lat=center_lat,
                    center_lon=center_lon,
                    scan_time=scan_time,
                    time_window=timedelta(minutes=5),
                    debug=debug,
                )
                if debug:
                    logger.info(
                        "[link_obs_data] synoptic rows=%d for cell=%s scan=%s",
                        len(df_synoptic),
                        cell_id,
                        radar_file_name,
                    )
            else:
                df_synoptic = pd.DataFrame(columns=["source", "time", "station_lat", "station_lon", "gust", "obs_distance"])
                if debug:
                    logger.info("[link_obs_data] filter_synoptic unavailable; using empty synoptic df")

            # Get obs data from lsr
            df_lsr = filter_lsr(
                lsr_df=lsr_df,
                bounding_lat=bounding_lat,
                bounding_lon=bounding_lon,
                center_lat=center_lat,
                center_lon=center_lon,
                scan_time=scan_time,
                debug=debug,
            )
            if debug:
                logger.info(
                    "[link_obs_data] lsr rows=%d for cell=%s scan=%s",
                    len(df_lsr),
                    cell_id,
                    radar_file_name,
                )

            # Get obs data from spc
            df_spc = filter_spc(
                spc_df=spc_df,
                bounding_lat=bounding_lat,
                bounding_lon=bounding_lon,
                center_lat=center_lat,
                center_lon=center_lon,
                scan_time=scan_time,
            )
            if debug:
                logger.info(
                    "[link_obs_data] spc rows=%d for cell=%s scan=%s",
                    len(df_spc),
                    cell_id,
                    radar_file_name,
                )

            # For each cell, concatenate the three dataframes observational dataframes
            df_cell = pd.concat([df_synoptic, df_lsr, df_spc], ignore_index=True)

            # If no observations, skip and move on to next cell
            if df_cell.empty:
                continue

            # If not empty, then add columns containing cell metadata
            df_cell["radar_file_name"] = radar_file_name
            df_cell["scan_time"] = scan_time
            df_cell["cell_id"] = cell_id
            df_cell["cell_lat"] = center_lat
            df_cell["cell_lon"] = center_lon

            # Unpack tuples so operations are easier later
            min_lat, max_lat = bounding_lat
            min_lon, max_lon = bounding_lon

            df_cell["bounding_lat_min"] = min_lat
            df_cell["bounding_lat_max"] = max_lat
            df_cell["bounding_lon_min"] = min_lon
            df_cell["bounding_lon_max"] = max_lon

            # Add each cell's dataframe into scan_records
            scan_records.append(df_cell)

            if debug:
                logger.info(
                    "[link_obs_data] built df for cell=%s scan=%s rows=%d",
                    cell_id,
                    radar_file_name,
                    len(df_cell),
                )

                # Save overlay plot(s) to per-call plot_dir
                if plot_dir is not None:
                    file_stem = f"{safe_name}_cell{cell_id}"
                    _plot_observations_on_fig(
                        cell_fig,
                        df_cell,
                        lon_col="station_lon",
                        lat_col="station_lat",
                        ms=50,
                        alpha=0.85,
                        save_dir=plot_dir,
                        file_stem=file_stem,
                    )

                section_seperator(4)

        # Write scan_records to cache
        if not scan_records:
            scan_df = pd.DataFrame(
                columns=[
                    "source",
                    "time",
                    "station_lat",
                    "station_lon",
                    "gust",
                    "distance_km",
                    "radar_file_name",
                    "scan_time",
                    "cell_id",
                    "cell_lat",
                    "cell_lon",
                    "bounding_lat_min",
                    "bounding_lat_max",
                    "bounding_lon_min",
                    "bounding_lon_max",
                ]
            )
            if debug:
                logger.info("[link_obs_data] No obs for scan %s; will cache empty dataframe.", radar_file_name)
        else:
            scan_df = pd.concat(scan_records, ignore_index=True)

        try:
            tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
            cache_path.parent.mkdir(parents=True, exist_ok=True)

            if "scan_time" in scan_df.columns:
                scan_df["scan_time"] = pd.to_datetime(scan_df["scan_time"], utc=True, errors="coerce")
            scan_df.to_csv(tmp_path, index=False)
            os.replace(str(tmp_path), str(cache_path))

            if debug:
                logger.info("[link_obs_data] Wrote cache for %s -> %s", radar_file_name, cache_path)
                section_seperator(4)

        except Exception:
            if debug:
                logger.exception("[link_obs_data] failed to write cache for %s", radar_file_name)

        all_records.append(scan_df)

    # Finally, handle all_records after looping through all radar scans
    if not all_records:
        # Return an empty DataFrame with expected columns
        full = pd.DataFrame(
            columns=[
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
                "cell_lat",
                "cell_lon",
                "bounding_lat_min",
                "bounding_lat_max",
                "bounding_lon_min",
                "bounding_lon_max",
            ]
        )
    else:
        full = pd.concat(all_records, ignore_index=True)

    return full


def _plot_observations_on_fig(
    fig,
    obs_df,
    lon_col="station_lon",
    lat_col="station_lat",
    marker="o",
    ms=40,
    alpha=0.9,
    edgecolor="k",
    zorder=6,
    save_dir: Path | None = None,
    file_stem: str | None = None,
):
    """
    Overlay observation points onto the first axis of `fig`, and (optionally) save to disk.

    Notes:
      - This is only called when debug=True in link_obs_data().
      - If save_dir is provided, saves a PNG into that folder.
    """
    if fig is None:
        logger.info("[_plot_observations_on_fig] fig=None -> skipping overlay")
        return

    if obs_df is None or obs_df.empty:
        logger.info("[_plot_observations_on_fig] obs_df empty -> skipping overlay")
        return

    # get or create axis
    ax = fig.axes[0] if getattr(fig, "axes", None) else fig.add_subplot(111)

    # make sure we have numeric lon/lat
    lons = pd.to_numeric(obs_df[lon_col], errors="coerce")
    lats = pd.to_numeric(obs_df[lat_col], errors="coerce")
    valid = lons.notna() & lats.notna()
    n_valid = int(valid.sum())

    logger.info("[_plot_observations_on_fig] plotting %d observation(s) (from %d rows)", n_valid, len(obs_df))

    if n_valid == 0:
        return

    ax.scatter(
        lons[valid],
        lats[valid],
        s=ms,
        marker=marker,
        alpha=alpha,
        edgecolors=edgecolor,
        linewidths=0.5,
        zorder=zorder,
        label="obs",
    )

    # Ensure new points fall inside the visible area:
    try:
        ax.relim()
        ax.autoscale_view()

        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        padx = 0.05 * (xmax - xmin) if (xmax - xmin) != 0 else 0.01
        pady = 0.05 * (ymax - ymin) if (ymax - ymin) != 0 else 0.01
        ax.set_xlim(xmin - padx, xmax + padx)
        ax.set_ylim(ymin - pady, ymax + pady)
    except Exception:
        logger.exception("[_plot_observations_on_fig] autoscale failed")

    # tidy legend (dedupe)
    try:
        handles, labels = ax.get_legend_handles_labels()
        uniq = {}
        for h, l in zip(handles, labels):
            if l not in uniq:
                uniq[l] = h
        if uniq:
            ax.legend(list(uniq.values()), list(uniq.keys()), fontsize="small", loc="best")
    except Exception:
        pass

    # Save to disk (preferred for EC2)
    if save_dir is not None:
        try:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            stem = file_stem or "obs_overlay"
            stem = re.sub(r"[^A-Za-z0-9._-]", "_", stem)
            out_path = save_dir / f"{stem}.png"
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            logger.info("[_plot_observations_on_fig] saved %s", out_path)
        except Exception:
            logger.exception("[_plot_observations_on_fig] failed to save plot")

    if plt is not None:
        try:
            plt.pause(0.001)
        except Exception:
            pass
