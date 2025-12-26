


############################################################################# MISC. ###########################################################################



# Haversine distance (km) between two lon/lat points
def haversine(lon1, lat1, lon2, lat2):
    R = 6371.0  # Earth radius in km
    dlon = radians(lon2 - lon1)
    dlat = radians(lat2 - lat1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return 2*R*asin(sqrt(a))


def section_seperator(lines_to_print):
    for _ in range(lines_to_print):
      print()

# Helper: ensure directory exists
def _ensure_dir(path: str):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)



# Helper: weighted dimensions
def _compute_weighted_dimensions(widths_m: np.ndarray, heights_m: np.ndarray, nbins=20, debug=False):
    """
    Compute weighted average of dimensions (width, height in meters).
    Weight is higher for rows whose width AND height are nearer the mode (histogram mode),
    with robustness via MAD.
    Returns (avg_width_m, avg_height_m, weights_array).
    """
    widths = np.asarray(widths_m, dtype=float)
    heights = np.asarray(heights_m, dtype=float)
    assert widths.shape == heights.shape

    # histogram-mode helper
    def hist_mode(vals, nbins_local):
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return np.nan
        counts, edges = np.histogram(vals, bins=nbins_local)
        idx = int(np.argmax(counts))
        return 0.5 * (edges[idx] + edges[idx + 1])

    mode_w = hist_mode(widths, nbins)
    mode_h = hist_mode(heights, nbins)
    if debug:
        print(f"[_compute_weighted_dimensions] mode_w={mode_w:.1f}, mode_h={mode_h:.1f}")

    # MAD scales (robust)
    def mad_scale(vals):
        v = vals[np.isfinite(vals)]
        if v.size == 0:
            return 1.0
        med = np.median(v)
        mad = np.median(np.abs(v - med))
        return max(mad, 1.0)  # avoid near-zero scale

    scale_w = mad_scale(widths)
    scale_h = mad_scale(heights)

    # deviation scores -> convert into weight scores (higher for smaller dev)
    dev_w = np.abs(widths - mode_w)
    dev_h = np.abs(heights - mode_h)

    w_score = 1.0 / (1.0 + (dev_w / (scale_w + 1e-12)))
    h_score = 1.0 / (1.0 + (dev_h / (scale_h + 1e-12)))

    combined = w_score * h_score
    # fallback to equal weights if something weird happens
    if np.sum(combined) <= 0:
        combined = np.ones_like(combined)

    weights = combined / float(np.sum(combined))
    avg_w = float(np.sum(weights * widths))
    avg_h = float(np.sum(weights * heights))

    if debug:
        print(f"[_compute_weighted_dimensions] avg_w={avg_w:.1f} m, avg_h={avg_h:.1f} m")
    return avg_w, avg_h, weights



######################################################################## RAM CLEANUP ########################################################################



def _safe_close(obj):
    # Best-effort close for things that may hold file handles or buffers
    for name in ("close", "shutdown", "stop", "terminate", "flush"):
        try:
            getattr(obj, name)()
            return
        except Exception:
            pass

def aggressive_memory_cleanup(locals_dict):
    # 1) Close plotting state (matplotlib holds a LOT)
    try:
        plt.close('all')
    except Exception:
        pass

    # 2) Shut down any executors/threads/queues you created
    for k, v in list(locals_dict.items()):
        # queues, threads, executors, file systems, netCDF datasets, s3fs/gcsfs, etc.
        try:
            _safe_close(v)
        except Exception:
            pass

    # 3) Drop large objects by name (customize this list for your code)
    big_names = [
        "lsr_df", "spc_df", "grs_df",
        "grouped", "linked_radar_df", "bboxed_df",
        "files", "full_df",
        "raw_queue", "plot_thread",
        # add anything else large that might linger (arrays, caches, etc.)
    ]
    for name in big_names:
        if name in locals_dict:
            locals_dict[name] = None

    # 4) Clear module-level caches if you have them
    for cache_name in [
        "_GRID_PLAN_CACHE",     # your own cache variable (example from your earlier code)
        "_s3fs", "_s3client",   # s3fs/boto (if you created globals)
    ]:
        if cache_name in globals():
            try:
                obj = globals()[cache_name]
                _safe_close(obj)
            except Exception:
                pass
            globals()[cache_name] = None

    # 5) Free TF/PyTorch contexts if they exist (safe no-ops if not installed)
    try:
        import tensorflow as tf
        tf.keras.backend.clear_session()
    except Exception:
        pass
    try:
        import torch
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    except Exception:
        pass

    # 6) Force Python GC
    gc.collect()
    gc.collect()  # call twice sometimes helps drop finalizers/cycles

    # 7) Ask glibc to return free arenas to the OS (Linux only; works on Colab)
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass
