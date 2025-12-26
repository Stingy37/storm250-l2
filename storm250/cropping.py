def _find_blob_for_point(comp_scan,
                         center_lat, center_lon,
                         class_field='reflectivity',
                         threshold=20,
                         min_size=2500,
                         pad_m=5000.0,
                         grid_res_m=1000.0,
                         include_nearby_km=10.0,
                         debug=False,
                         plot_dir: str | None = None,
                         plot_stub: str | None = None):
    """
    (fast) Find the labeled blob containing (center_lat, center_lon) and return
    bbox with recursive merging of nearby blobs. Returns (minlat,maxlat,minlon,maxlon, centers_list).
    """
    from scipy import ndimage as ndi  # local import to keep module deps light

    # timing debug
    if debug:
        from time import perf_counter
        _t0 = perf_counter()

    # normalize center lon to canonical range [-180,180)
    center_lon = ((center_lon + 180.0) % 360.0) - 180.0

    # grid & labeling (unchanged helper)
    info = _compute_metric_grid_and_labels(
        comp_scan, class_field=class_field, threshold=threshold,
        pad_m=pad_m, grid_res_m=grid_res_m, debug=debug
    )
    labeled_grid = info['labeled_grid']      # (ny, nx), labels 0..N
    grid_x = info['grid_x']                  # metric x coords (nx,)
    grid_y = info['grid_y']                  # metric y coords (ny,)
    t_xy2geog = info['t_xy2geog']
    t_geog2xy = info['t_geog2xy']
    grid_mask = info['grid_mask']            # boolean mask of valid > threshold

    # center in metric coords (meters)
    cx_m, cy_m = t_geog2xy.transform(center_lon, center_lat)

    # grid cell containing the center (fast, vectorized edges)
    xi = np.searchsorted(grid_x, cx_m) - 1
    yi = np.searchsorted(grid_y, cy_m) - 1
    xi = max(0, min(xi, grid_x.size - 1))
    yi = max(0, min(yi, grid_y.size - 1))

    # ----- FAST region statistics -----
    # counts per label (O(N))
    lab_flat = labeled_grid.ravel()
    if debug: _t_stats0 = perf_counter()
    counts = np.bincount(lab_flat)  # index 0 is background
    # keep labels with enough pixels
    valid_labels = np.nonzero(counts >= max(1, int(min_size)))[0]
    valid_labels = valid_labels[valid_labels != 0]  # drop background

    if valid_labels.size == 0:
        if debug:
            print("[_find_blob_for_point] no blobs passed the min_size filter")
            _t_end = perf_counter()
            print(f"[_find_blob_for_point] TOTAL: {(_t_end - _t0):.3f}s (early exit: no blobs)")
        return None, None, None, None, []

    # quick bbox per label via slices (O(N))
    obj_slices = ndi.find_objects(labeled_grid, max_label=int(labeled_grid.max()))
    # center-of-mass for all valid labels in one go (fast C-impl)
    # (use grid_mask as input so COM is over "inside" pixels only)
    coms = ndi.center_of_mass(grid_mask.astype(np.float32), labeled_grid, index=valid_labels)
    coms = np.asarray(coms, dtype=float)  # shape (k, 2) -> (y, x) in index space

    # convert COM (index space) -> metric coords
    # Using linear mapping from index to coord: x = grid_x[idx], y = grid_y[idx]
    # center_of_mass returns fractional index positions; map by interpolation on the arrays
    x_idx = coms[:, 1]
    y_idx = coms[:, 0]
    # Faster than np.interp per row: use uniform spacing if grid is uniform; otherwise fall back to interp.
    # grid_x/y are produced via linspace -> uniform
    dx = grid_x[1] - grid_x[0]
    dy = grid_y[1] - grid_y[0]
    cx = grid_x[0] + x_idx * dx
    cy = grid_y[0] + y_idx * dy

    # Build per-label bboxes in metric coords using the object slices
    # find_objects returns slices indexed by label-1
    minx = np.empty(valid_labels.size, dtype=float)
    maxx = np.empty(valid_labels.size, dtype=float)
    miny = np.empty(valid_labels.size, dtype=float)
    maxy = np.empty(valid_labels.size, dtype=float)

    # vectorizable using slice arrays
    # Each slice is (slice_y, slice_x) with .start/.stop indices
    sl_y = np.fromiter((obj_slices[lab-1][0].start for lab in valid_labels), dtype=np.int64, count=valid_labels.size)
    el_y = np.fromiter((obj_slices[lab-1][0].stop  for lab in valid_labels), dtype=np.int64, count=valid_labels.size)
    sl_x = np.fromiter((obj_slices[lab-1][1].start for lab in valid_labels), dtype=np.int64, count=valid_labels.size)
    el_x = np.fromiter((obj_slices[lab-1][1].stop  for lab in valid_labels), dtype=np.int64, count=valid_labels.size)

    # Convert index bounds to metric coords (stop is exclusive; use stop-1)
    minx[:] = grid_x[sl_x]
    maxx[:] = grid_x[np.maximum(el_x - 1, 0)]
    miny[:] = grid_y[sl_y]
    maxy[:] = grid_y[np.maximum(el_y - 1, 0)]

    # transform all centroids -> lon/lat in one call (vectorized)
    clon, clat = t_xy2geog.transform(cx, cy)
    # normalize longitudes
    clon = ((np.asarray(clon) + 180.0) % 360.0) - 180.0

    # Build centers list (same schema as before)
    centers = [
        {
            'label': int(lab),
            'clat': float(clat[i]),
            'clon': float(clon[i]),
            'cx':   float(cx[i]),
            'cy':   float(cy[i]),
            'size': int(counts[lab]),
            'minx': float(minx[i]), 'maxx': float(maxx[i]),
            'miny': float(miny[i]), 'maxy': float(maxy[i]),
        }
        for i, lab in enumerate(valid_labels)
    ]
    if debug:
        _t_stats1 = perf_counter()
        print(f"[_find_blob_for_point] region stats: {(_t_stats1 - _t_stats0):.3f}s  "
              f"(labels={valid_labels.size}, total_cells={labeled_grid.size})")

    # ----- choose initial label -----
    label_at_center = int(labeled_grid[yi, xi]) if (0 <= yi < labeled_grid.shape[0] and 0 <= xi < labeled_grid.shape[1]) else 0
    if label_at_center != 0 and (label_at_center in valid_labels):
        chosen_label = int(label_at_center)
        if debug:
            print(f"[_find_blob_for_point] center cell belongs to label {chosen_label}")
    else:
        # vectorized nearest centroid in metric space
        dxv = cx - cx_m
        dyv = cy - cy_m
        nearest_idx = int(np.argmin(dxv*dxv + dyv*dyv))
        chosen_label = int(valid_labels[nearest_idx])
        if debug:
            best_dist = float(np.hypot(dxv[nearest_idx], dyv[nearest_idx]))
            print(f"[_find_blob_for_point] chosen nearest label {chosen_label} (dist {best_dist:.1f} m)")
    if debug:
        _t_choose = perf_counter()
        print(f"[_find_blob_for_point] choose-initial-label: {(_t_choose - _t_stats1):.3f}s (label={chosen_label})")

    # ----- iterative merging (same logic; operates on arrays) -----
    expand_m = float(include_nearby_km) * 1000.0
    included = {chosen_label}
    if debug:
        print(f"[_find_blob_for_point] starting merge with label {chosen_label}, expand_km={include_nearby_km}")

    # index map from label -> row index in our arrays
    lab_to_idx = {int(lab): int(i) for i, lab in enumerate(valid_labels)}

    def union_bbox(indices):
        # vectorized union over selected indices
        return (minx[indices].min(), maxx[indices].max(),
                miny[indices].min(), maxy[indices].max())

    iteration = 0
    if debug: _t_merge0 = perf_counter()
    while True:
        iteration += 1
        idxs = np.fromiter((lab_to_idx[l] for l in included), dtype=int)
        uminx, umaxx, uminy, umaxy = union_bbox(idxs)

        # expanded union bbox
        exp_minx = uminx - expand_m
        exp_maxx = umaxx + expand_m
        exp_miny = uminy - expand_m
        exp_maxy = umaxy + expand_m

        if debug:
            print(f"[_find_blob_for_point] iter {iteration}: included={sorted(included)}, "
                  f"union_bbox_m=[{uminx:.1f},{umaxx:.1f}] x [{uminy:.1f},{umaxy:.1f}], "
                  f"expanded by {expand_m:.1f} m")

        # Compute bbox-to-bbox distance to all *not yet included* candidates (vectorized)
        remaining = [l for l in valid_labels if l not in included]
        if not remaining:
            break
        r_idx = np.array([lab_to_idx[int(l)] for l in remaining], dtype=int)

        # dx: horizontal gap (0 if overlapping)
        dx_left  = np.maximum(0.0, minx[r_idx] - exp_maxx)
        dx_right = np.maximum(0.0, exp_minx - maxx[r_idx])
        dx_gap = np.where(dx_left > 0, dx_left, dx_right)

        # dy: vertical gap (0 if overlapping)
        dy_low   = np.maximum(0.0, miny[r_idx] - exp_maxy)
        dy_high  = np.maximum(0.0, exp_miny - maxy[r_idx])
        dy_gap = np.where(dy_low > 0, dy_low, dy_high)

        dist = np.hypot(dx_gap, dy_gap)
        add_mask = (dist <= expand_m)  # overlap or within expand distance
        if not np.any(add_mask):
            if debug:
                print(f"[_find_blob_for_point] iter {iteration}: no new blobs added; merging finished")
            break

        for l in np.asarray(remaining)[add_mask]:
            included.add(int(l))
            if debug:
                print(f"[_find_blob_for_point] iter {iteration}: added label {int(l)} (bbox-dist {float(dist[add_mask][0]):.1f} m)")
    if debug:
        _t_merge1 = perf_counter()
        print(f"[_find_blob_for_point] merge nearby: {(_t_merge1 - _t_merge0):.3f}s  "
              f"(iters={iteration}, included={len(included)})")

    # final union bbox (metric coords)
    idxs = np.fromiter((lab_to_idx[l] for l in included), dtype=int)
    final_minx = float(minx[idxs].min())
    final_maxx = float(maxx[idxs].max())
    final_miny = float(miny[idxs].min())
    final_maxy = float(maxy[idxs].max())

    if debug: _t_back0 = perf_counter()

    # convert final metric bbox -> lon/lat (single vectorized call)
    lon_pair, lat_pair = t_xy2geog.transform(
        np.array([final_minx, final_maxx], dtype=float),
        np.array([final_miny, final_maxy], dtype=float)
    )
    minlon = ((float(lon_pair[0]) + 180.0) % 360.0) - 180.0
    maxlon = ((float(lon_pair[1]) + 180.0) % 360.0) - 180.0
    minlat = float(lat_pair[0])
    maxlat = float(lat_pair[1])
    if debug:
        _t_back1 = perf_counter()
        print(f"[_find_blob_for_point] bbox metric→geo transform: {(_t_back1 - _t_back0)*1000:.1f} ms")
        print(f"[_find_blob_for_point] final merged bbox lat [{minlat:.4f}, {maxlat:.4f}] lon [{minlon:.4f}, {maxlon:.4f}]")
        print(f"[_find_blob_for_point] merged labels: {sorted(list(included))}")

        '''
        ################################################### COMMENT PLOTTING BLOCK OUT ON PRODUCTION RUNS #######################################################
        #                                                             (VERY SLOW OTHERWISE)


        # plotting block unchanged except we already know the host sweep
        try:
            from pyart.graph import RadarDisplay
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            ax0, ax1 = axes
            try:
                display = RadarDisplay(comp_scan)
                sweep = int(getattr(comp_scan, "metadata", {}).get("pseudo_host_sweep", 0))
                display.plot(class_field, sweep=sweep, ax=ax0, title=f"Reflectivity (sweep {sweep})")
            except Exception as e:
                if debug:
                    print(f"[_find_blob_for_point] RadarDisplay plotting failed: {e}")
                ax0.text(0.5, 0.5, "RadarDisplay failed", ha='center')
                ax0.set_title("Reflectivity (not available)")

            GX, GY = np.meshgrid(grid_x, grid_y)
            lon_grid, lat_grid = t_xy2geog.transform(GX, GY)
            try:
                radar_lon = float(comp_scan.longitude['data']) if np.isscalar(comp_scan.longitude['data']) else float(comp_scan.longitude['data'][0])
            except Exception:
                radar_lon = np.nanmedian(comp_scan.gate_longitude['data'])

            # Use the *same* normalization scheme as everywhere else:
            lon_grid = _normalize_lons_to_center(lon_grid, radar_lon)
            ax1.pcolormesh(lon_grid, lat_grid, grid_mask.astype(int),
                           shading='auto', cmap='viridis', alpha=0.5)

            # plot every candidate bbox (light gray)
            # vectorized draw: still loop to draw rectangles
            for i, lab in enumerate(valid_labels):
                lon_min_c, lat_min_c = t_xy2geog.transform(minx[i], miny[i])
                lon_max_c, lat_max_c = t_xy2geog.transform(maxx[i], maxy[i])
                lon_min_c = ((lon_min_c + 180.0) % 360.0) - 180.0
                lon_max_c = ((lon_max_c + 180.0) % 360.0) - 180.0
                w = lon_max_c - lon_min_c
                if w < 0: w += 360.0
                h = lat_max_c - lat_min_c
                rect = Rectangle((lon_min_c, lat_min_c), w, h,
                                 fill=False, linewidth=1.0, linestyle='-', edgecolor='lightgray', alpha=0.9, zorder=2)
                ax1.add_patch(rect)
                ax1.text(centers[i]['clon'], centers[i]['clat'], str(int(lab)),
                         fontsize=8, ha='center', va='center', zorder=11)

            # highlight included bboxes + final bbox
            for l in included:
                i = lab_to_idx[int(l)]
                lon_min_c, lat_min_c = t_xy2geog.transform(minx[i], miny[i])
                lon_max_c, lat_max_c = t_xy2geog.transform(maxx[i], maxy[i])
                lon_min_c = ((lon_min_c + 180.0) % 360.0) - 180.0
                lon_max_c = ((lon_max_c + 180.0) % 360.0) - 180.0
                w = lon_max_c - lon_min_c
                if w < 0: w += 360.0
                h = lat_max_c - lat_min_c
                rect = Rectangle((lon_min_c, lat_min_c), w, h,
                                 fill=True, linewidth=1.5, linestyle='-',
                                 edgecolor='blue', facecolor='blue', alpha=0.15, zorder=5)
                ax1.add_patch(rect)

            # final bbox
            width = maxlon - minlon
            if width < 0: width += 360.0
            height = maxlat - minlat
            rect_final = Rectangle((minlon, minlat), width, height,
                                   fill=False, linewidth=3.0, linestyle='--', edgecolor='red', zorder=12)
            ax1.add_patch(rect_final)

            # track center
            plot_center_lon = ((center_lon + 180.0) % 360.0) - 180.0
            ax1.scatter(plot_center_lon, center_lat, s=120, marker='X', edgecolor='k',
                        facecolor='yellow', zorder=15, label='track center')

            ax1.set_xlabel("Longitude"); ax1.set_ylabel("Latitude")
            ax1.set_title("Gridded mask + candidate cell bboxes (gray), included (blue), final bbox (red)")
            ax1.set_aspect('equal', adjustable='box')
            ax1.legend(loc='upper right', fontsize='small')
            plt.tight_layout()

            # Save to directory
            stub = plot_stub or "find_blob"
            _save_plot(fig=fig,
                       plot_dir=plot_dir,
                       func_name="_find_blob_for_point",
                       stub=stub,
                       debug=debug,
                       )

        except Exception as e:
            if debug:
                print(f"[_find_blob_for_point] debug plotting (reflectivity + mask) failed: {e}")


        ################################################################### END BLOCK ###################################################################
        '''

    if debug:
        _t_total = perf_counter()
        print(f"[_find_blob_for_point] TOTAL: {(_t_total - _t0):.3f}s")
    return minlat, maxlat, minlon, maxlon, centers, info


def _bbox_center(minlat, maxlat, minlon, maxlon):
    # robust center even if the box crosses the dateline
    width = maxlon - minlon
    if width < 0: width += 360.0
    clon = minlon + 0.5 * width
    clon = ((clon + 180.0) % 360.0) - 180.0
    clat = 0.5 * (minlat + maxlat)
    return clat, clon



# Helper: fit averaged box to a scan
def _fit_box_to_scan_max_overlap(comp_scan,
                                 width_m, height_m,
                                 class_field='reflectivity',
                                 threshold=20,
                                 search_km=20.0, step_km=2.5,
                                 center_hint=None,      # <-- used now
                                 debug=False):
    """
    Grid-search candidate box centers around a chosen center (by default the
    blob centroid; if center_hint is given, use that), and return
    (minlat,maxlat,minlon,maxlon) of the best-scoring box (max fraction of
    dbz>=threshold gates inside the box).
    """
    # Build gate-level blob mask and gate coords
    sweep = int(getattr(comp_scan, "metadata", {}).get("pseudo_host_sweep", 0))
    start = comp_scan.sweep_start_ray_index['data'][sweep]
    stop  = comp_scan.sweep_end_ray_index['data'][sweep]
    dbz_field = comp_scan.fields[class_field]['data'][start:stop, :]
    if isinstance(dbz_field, np.ma.MaskedArray):
        dbz_arr = dbz_field.filled(np.nan)
    else:
        dbz_arr = dbz_field

    comp_scan.init_gate_longitude_latitude()
    gate_lats = comp_scan.gate_latitude['data'][start:stop, :]
    gate_lons = comp_scan.gate_longitude['data'][start:stop, :]

    # normalize gate longitudes around radar lon
    try:
        radar_lon = float(comp_scan.longitude['data']) if np.isscalar(comp_scan.longitude['data']) else float(comp_scan.longitude['data'][0])
    except Exception:
        radar_lon = None
    if radar_lon is not None:
        radar_lon_norm = ((radar_lon + 180.0) % 360.0) - 180.0
        gate_lons = _normalize_lons_to_center(gate_lons, radar_lon_norm)
    else:
        radar_lon_norm = None

    blob_mask = (dbz_arr >= threshold) & np.isfinite(dbz_arr)
    total_blob = float(blob_mask.sum())
    # if no blob data, fallback to scan centroid area near radar
    if total_blob <= 0:
        try:
            rlat = float(comp_scan.latitude['data']) if np.isscalar(comp_scan.latitude['data']) else float(comp_scan.latitude['data'][0])
            rlon = float(comp_scan.longitude['data']) if np.isscalar(comp_scan.longitude['data']) else float(comp_scan.longitude['data'][0])
        except Exception:
            rlat = float(np.nanmean(gate_lats[np.isfinite(gate_lats)]))
            rlon = float(np.nanmean(gate_lons[np.isfinite(gate_lons)]))
        return rlat - 0.5, rlat + 0.5, rlon - 0.5, rlon + 0.5

    # centroid (in lon/lat) of ALL gates >= threshold (baseline)
    valid = blob_mask & np.isfinite(gate_lats) & np.isfinite(gate_lons)
    if valid.sum() == 0:
        centroid_lat = float(np.nanmean(gate_lats[np.isfinite(gate_lats)]))
        centroid_lon = float(np.nanmean(gate_lons[np.isfinite(gate_lons)]))
    else:
        centroid_lat = float(np.nanmean(gate_lats[valid]))
        centroid_lon = float(np.nanmean(gate_lons[valid]))

    # choose center: override with center_hint if provided
    used_lat, used_lon = centroid_lat, centroid_lon
    if center_hint is not None:
        try:
            hl, ho = float(center_hint[0]), float(center_hint[1])
            # normalize hint lon consistently with gate_lons
            if radar_lon_norm is not None:
                ho = _normalize_lons_to_center(ho, radar_lon_norm)
            else:
                ho = ((ho + 180.0) % 360.0) - 180.0
            used_lat, used_lon = hl, ho
            if debug:
                print(f"[_fit_box_to_scan_max_overlap] using center_hint=({used_lat:.4f},{used_lon:.4f}) "
                      f"(raw centroid was {centroid_lat:.4f},{centroid_lon:.4f})")
        except Exception as e:
            if debug:
                print(f"[_fit_box_to_scan_max_overlap] center_hint ignored due to error: {e}")

    # prepare projection to metric coordinates (centered at radar)
    if debug: _t_grid0 = perf_counter()
    info = _compute_metric_grid_and_labels(
        comp_scan, class_field=class_field, threshold=threshold,
        pad_m=0.0, grid_res_m=1000.0, debug=False
    )
    if debug:
        _t_grid1 = perf_counter()
        ny = int(info['grid_y'].size); nx = int(info['grid_x'].size)
        print(f"[_find_blob_for_point] grid+labeling: {(_t_grid1 - _t_grid0):.3f}s  (ny={ny}, nx={nx}, pixels={ny*nx})")

    t_geog2xy = info['t_geog2xy']
    t_xy2geog = info['t_xy2geog']

    # search around the chosen center
    cx_m, cy_m = t_geog2xy.transform(used_lon, used_lat)

    search_m = float(search_km) * 1000.0
    step_m = float(step_km) * 1000.0
    offsets = np.arange(-search_m, search_m + 1e-9, step_m)

    # flatten gate coords & blob mask for fast checks
    gate_lons_flat = gate_lons.ravel()
    gate_lats_flat = gate_lats.ravel()
    blob_flat = blob_mask.ravel()
    total_blob_pts = float(blob_flat.sum())

    best_score = -1.0
    best_box = None

    for dx in offsets:
        for dy in offsets:
            center_x = cx_m + dx
            center_y = cy_m + dy
            minx = center_x - (width_m / 2.0)
            maxx = center_x + (width_m / 2.0)
            miny = center_y - (height_m / 2.0)
            maxy = center_y + (height_m / 2.0)
            # convert corners back to lon/lat for gate arithmetic
            lon_min, lat_min = t_xy2geog.transform(minx, miny)
            lon_max, lat_max = t_xy2geog.transform(maxx, maxy)
            if radar_lon_norm is not None:
                lon_min = _normalize_lons_to_center(lon_min, radar_lon_norm)
                lon_max = _normalize_lons_to_center(lon_max, radar_lon_norm)
            else:
                lon_min = ((lon_min + 180.0) % 360.0) - 180.0
                lon_max = ((lon_max + 180.0) % 360.0) - 180.0

            # gate inclusion test (degrees, approximate)
            inside = ((gate_lats_flat >= min(lat_min, lat_max)) &
                      (gate_lats_flat <= max(lat_min, lat_max)) &
                      (gate_lons_flat >= min(lon_min, lon_max)) &
                      (gate_lons_flat <= max(lon_min, lon_max)))
            if inside.sum() == 0:
                score = 0.0
            else:
                score = float((blob_flat & inside).sum()) / (total_blob_pts + 1e-12)

            if score > best_score:
                best_score = score
                best_box = (min(lat_min, lat_max),
                            max(lat_min, lat_max),
                            min(lon_min, lon_max),
                            max(lon_min, lon_max))

    if debug:
        print(f"[_fit_box_to_scan_max_overlap] best_score={best_score:.3f}")
        if best_box is not None:
            bminlat, bmaxlat, bminlon, bmaxlon = best_box
            print(f"[_fit_box_to_scan_max_overlap] best_box "
                  f"lat [{bminlat:.4f}, {bmaxlat:.4f}] "
                  f"lon [{bminlon:.4f}, {bmaxlon:.4f}] "
                  f"(center_used {used_lat:.4f},{used_lon:.4f})")

    return best_box  # minlat, maxlat, minlon, maxlon



def _crop_comp_scan_to_bbox(comp_scan, minlat, maxlat, minlon, maxlon,
                            buffer_km=5.0, debug=False, inplace=True,
                            drop_gate_coords=True, prefer_nan_fill=True):
    """
    In-place crop via masking (or NaN-fill) outside bbox+buffer, optimized for speed & bandwidth:
      - builds the inside mask in *metric space* (no gate lon/lat fetch),
      - early exit if the mask keeps everything,
      - applies masks per-sweep slice if fields span all sweeps,
      - reuse a single shared mask across fields,
      - avoid array-wide dtype casts/writes for non-floating fields.
    """
    _t0 = perf_counter()

    # Copy vs in-place
    _t_copy0 = perf_counter()
    comp = comp_scan if inplace else copy.deepcopy(comp_scan)
    _t_copy1 = perf_counter()

    # Bbox + buffer (deg)
    _t_bbox0 = perf_counter()
    deg_per_lat_km = 1.0 / 110.574
    midlat = (minlat + maxlat) * 0.5
    deg_per_lon_km = 1.0 / (111.320 * np.cos(np.deg2rad(midlat))) if np.isfinite(midlat) else (1.0 / 111.320)
    lat_min = minlat - buffer_km * deg_per_lat_km
    lat_max = maxlat + buffer_km * deg_per_lat_km
    lon_min = minlon - buffer_km * deg_per_lon_km
    lon_max = maxlon + buffer_km * deg_per_lon_km
    _t_bbox1 = perf_counter()

    # Build inside mask in metric space (no gate lon/lat)
    _t_mask0 = perf_counter()
    inside_mask = _fast_metric_inside_mask(comp, lat_min, lat_max, lon_min, lon_max, buffer_km)
    _t_mask1 = perf_counter()

    # Compute sweep slice used by the mask
    sweep = int(getattr(comp, "metadata", {}).get("pseudo_host_sweep", 0))
    s = int(comp.sweep_start_ray_index['data'][sweep])
    e = int(comp.sweep_end_ray_index['data'][sweep])

    # Fast no-op escape
    _t_noop0 = perf_counter()
    if isinstance(inside_mask, np.ndarray) and inside_mask.dtype == bool and inside_mask.all():
        if debug:
            try:
                rays = e - s
            except Exception:
                rays = inside_mask.shape[0] if inside_mask.ndim == 2 else -1
            try:
                gates = int(np.asarray(comp.range['data']).size)
            except Exception:
                gates = inside_mask.shape[1] if inside_mask.ndim == 2 else -1
            print(f"[_crop_comp_scan_to_bbox] dims: rays={rays} gates={gates} fields={len(getattr(comp, 'fields', {}))} "
                  f"inplace={bool(inplace)} prefer_nan_fill={bool(prefer_nan_fill)} drop_gate_coords={bool(drop_gate_coords)}")
            print(f"    copy/deepcopy:        {(_t_copy1 - _t_copy0)*1000:.1f} ms")
            print(f"    bbox math:            {(_t_bbox1 - _t_bbox0)*1000:.1f} ms")
            print(f"    compute metric mask:  {(_t_mask1 - _t_mask0)*1000:.1f} ms")
            print(f"    no-op check:          {(perf_counter() - _t_noop0)*1000:.1f} ms")
            print(f"    TOTAL:                {(perf_counter() - _t0   )*1000:.1f} ms")
            print("[_crop_comp_scan_to_bbox] bbox covers full radar -> no-op (metric mask all True)")
        return comp
    _t_noop1 = perf_counter()

    # Turn inside→outside (shared mask)
    _t_inv0 = perf_counter()
    np.logical_not(inside_mask, out=inside_mask)
    mask_outside = inside_mask
    _t_inv1 = perf_counter()

    gate_shape = mask_outside.shape  # = (e-s, ngates)
    shared_bcast = {}

    # Per-field timing accumulators
    _t_fields0 = perf_counter()
    n_fields = 0; n_masked = 0; n_float = 0; n_intbool = 0
    t_masked_merge = 0.0; t_float_fill = 0.0; t_int_wrap = 0.0; t_bcast_build = 0.0

    # Helper: apply mask to a MaskedArray that spans *all* rays (slice [s:e,:])
    def _merge_mask_slice(arr_ma, mask_slice):
        # Prepare a full-size mask (only slice gets OR-ed with mask_slice)
        if (arr_ma.mask is np.ma.nomask) or (arr_ma.mask is False):
            full = np.zeros(arr_ma.shape, dtype=bool)
        elif isinstance(arr_ma.mask, np.ndarray) and arr_ma.mask.shape == arr_ma.shape:
            full = arr_ma.mask
        else:
            try:
                full = np.array(arr_ma.mask, dtype=bool, copy=True)
            except Exception:
                full = np.zeros(arr_ma.shape, dtype=bool)
            if full.shape != arr_ma.shape:
                full = np.zeros(arr_ma.shape, dtype=bool)
        np.logical_or(full[s:e, :], mask_slice, out=full[s:e, :])
        arr_ma.mask = full

    for fname, fdict in comp.fields.items():
        arr = fdict.get("data", None)
        if arr is None:
            continue
        n_fields += 1
        _t_one0 = perf_counter()

        # -------- MaskedArray fields --------
        if isinstance(arr, np.ma.MaskedArray):
            n_masked += 1
            if arr.ndim == 2 and arr.shape == gate_shape:
                # exact per-sweep array
                if (arr.mask is np.ma.nomask) or (arr.mask is False):
                    arr.mask = mask_outside
                else:
                    if isinstance(arr.mask, np.ndarray) and arr.mask.shape == gate_shape:
                        np.logical_or(arr.mask, mask_outside, out=arr.mask)
                    else:
                        real = mask_outside.copy(order='C')
                        np.logical_or(real, arr.mask, out=real)
                        arr.mask = real
                try:
                    if arr.dtype != np.float32:
                        arr._data = arr._data.astype(np.float32, copy=False)
                except Exception:
                    pass
                fdict["data"] = arr

            elif arr.ndim == 2 and arr.shape[1] == gate_shape[1] and e <= arr.shape[0]:
                # full-radar array: apply only to [s:e, :]
                _merge_mask_slice(arr, mask_outside)
                try:
                    if arr.dtype != np.float32:
                        arr._data = arr._data.astype(np.float32, copy=False)
                except Exception:
                    pass
                fdict["data"] = arr

            else:
                # other shapes -> try broadcast (may be rare)
                if arr.shape not in shared_bcast:
                    _tb0 = perf_counter()
                    bm = np.broadcast_to(mask_outside, arr.shape).copy(order='C')
                    _tb1 = perf_counter()
                    shared_bcast[arr.shape] = bm
                    t_bcast_build += (_tb1 - _tb0)
                bm = shared_bcast[arr.shape]
                if (arr.mask is np.ma.nomask) or (arr.mask is False):
                    arr.mask = bm
                else:
                    if isinstance(arr.mask, np.ndarray) and arr.mask.shape == arr.shape:
                        np.logical_or(arr.mask, bm, out=arr.mask)
                    else:
                        real = bm.copy(order='C')
                        np.logical_or(real, arr.mask, out=real)
                        arr.mask = real
                try:
                    if arr.dtype != np.float32:
                        arr._data = arr._data.astype(np.float32, copy=False)
                except Exception:
                    pass
                fdict["data"] = arr

            t_masked_merge += (perf_counter() - _t_one0)
            continue

        # -------- Plain ndarray fields --------
        if prefer_nan_fill and np.issubdtype(arr.dtype, np.floating):
            n_float += 1
            _tf0 = perf_counter()
            if arr.dtype != np.float32:
                arr = arr.astype(np.float32, copy=False)

            if arr.ndim == 2 and arr.shape == gate_shape:
                # per-sweep
                np.copyto(arr, np.nan, where=mask_outside)
            elif arr.ndim == 2 and arr.shape[1] == gate_shape[1] and e <= arr.shape[0]:
                # full-radar: slice
                np.copyto(arr[s:e, :], np.nan, where=mask_outside)
            else:
                # fallback broadcast (rare)
                bm = shared_bcast.get(arr.shape)
                if bm is None:
                    _tb0 = perf_counter()
                    bm = np.broadcast_to(mask_outside, arr.shape).copy(order='C')
                    _tb1 = perf_counter()
                    shared_bcast[arr.shape] = bm
                    t_bcast_build += (_tb1 - _tb0)
                np.copyto(arr, np.nan, where=bm)

            fdict["data"] = arr
            t_float_fill += (perf_counter() - _tf0)
        else:
            n_intbool += 1
            _ti0 = perf_counter()
            if arr.ndim == 2 and arr.shape == gate_shape:
                fdict["data"] = _ma.MaskedArray(arr, mask=mask_outside, copy=False)
            elif arr.ndim == 2 and arr.shape[1] == gate_shape[1] and e <= arr.shape[0]:
                # full array -> build a full-size mask and fill only [s:e, :]
                bm_full = np.zeros(arr.shape, dtype=bool)
                bm_full[s:e, :] = mask_outside
                fdict["data"] = _ma.MaskedArray(arr, mask=bm_full, copy=False)
            else:
                bm = shared_bcast.get(arr.shape)
                if bm is None:
                    _tb0 = perf_counter()
                    bm = np.broadcast_to(mask_outside, arr.shape).copy(order='C')
                    _tb1 = perf_counter()
                    shared_bcast[arr.shape] = bm
                    t_bcast_build += (_tb1 - _tb0)
                fdict["data"] = _ma.MaskedArray(arr, mask=bm, copy=False)
            t_int_wrap += (perf_counter() - _ti0)

    _t_fields1 = perf_counter()

    # Optional: drop large cached geo arrays if desired
    _t_drop0 = perf_counter()
    if drop_gate_coords:
        for gate_key in ("gate_longitude", "gate_latitude", "gate_altitude"):
            try: delattr(comp, gate_key)
            except Exception: pass
    _t_drop1 = perf_counter()

    # Clear locals
    _t_clean0 = perf_counter()
    try: del mask_outside, inside_mask
    except Exception: pass
    _t_clean1 = perf_counter()

    if debug:
        rays = e - s
        try:
            gates = int(np.asarray(comp.range['data']).size)
        except Exception:
            gates = gate_shape[1] if isinstance(gate_shape, tuple) and len(gate_shape) >= 2 else -1

        print(f"[_crop_comp_scan_to_bbox] dims: rays={rays} gates={gates} fields={n_fields} "
              f"inplace={bool(inplace)} prefer_nan_fill={bool(prefer_nan_fill)} drop_gate_coords={bool(drop_gate_coords)}")
        print(f"    copy/deepcopy:        {(_t_copy1  - _t_copy0 )*1000:.1f} ms")
        print(f"    bbox math:            {(_t_bbox1  - _t_bbox0 )*1000:.1f} ms")
        print(f"    compute metric mask:  {(_t_mask1  - _t_mask0 )*1000:.1f} ms")
        print(f"    no-op check:          {(_t_noop1  - _t_noop0 )*1000:.1f} ms")
        print(f"    invert to outside:    {(_t_inv1   - _t_inv0  )*1000:.1f} ms")
        print(f"    per-field apply:      {(_t_fields1- _t_fields0)*1000:.1f} ms")
        print(f"        masked merge (n={n_masked}):   {t_masked_merge*1000:.1f} ms")
        print(f"        float NaN fill (n={n_float}):  {t_float_fill*1000:.1f} ms")
        print(f"        int/bool wrap (n={n_intbool}): {t_int_wrap*1000:.1f} ms")
        print(f"        broadcast builds:               {t_bcast_build*1000:.1f} ms")
        print(f"    drop gate coords:     {(_t_drop1  - _t_drop0 )*1000:.1f} ms")
        print(f"    cleanup locals:       {(_t_clean1 - _t_clean0)*1000:.1f} ms")
        print(f"    TOTAL:                {(perf_counter() - _t0   )*1000:.1f} ms")
        print(f"[_crop_comp_scan_to_bbox] cropped to {minlat:.4f}-{maxlat:.4f}, {minlon:.4f}-{maxlon:.4f} (buffer_km={buffer_km})")

    return comp




def build_bboxes_for_linked_df(linked_df: pd.DataFrame,
                               class_field='reflectivity',
                               threshold=20,
                               min_size=500,
                               pad_km=5.0,
                               grid_res_m=1000.0,
                               buffer_km=5.0,
                               include_nearby_km=8.0,
                               debug=False,
                               debug_plot_dir: str | None = None,
                               debug_plot_limit: int = 2) -> pd.DataFrame:

    """
    High-level handler (minimal multi-product changes):
      - Detect product keys in linked_df by columns that end with '_scan' (e.g. 'dhr_scan', 'dpa_scan').
      - Choose a reference product key for bbox computation: prefer 'dhr' if present, otherwise the first product key,
        falling back to legacy 'radar_scan' if present.
      - Use the reference composite to run the exact same bounding logic as before.
      - Crop EVERY product composite present in the row (for each <key>_scan) to the stationary bbox (loading from
        <key>_cache_member_name if needed) and store the cropped composite back in <key>_scan in the output rows.
    """
    # set up debug plot dir if requested
    if debug:
        if debug_plot_dir is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            debug_plot_dir = os.path.join("Logs", "plots", f"bbox_{ts}_pid{os.getpid()}")
        os.makedirs(debug_plot_dir, exist_ok=True)
        if debug:
            print(f"[build_bboxes] debug_plot_dir={debug_plot_dir}")

    # detect product keys (columns like '<key>_scan')
    product_keys = sorted({c[:-5] for c in linked_df.columns if c.endswith('_scan')})
    # legacy fallback: if no product-specific scan columns, but legacy 'radar_scan' exists, treat it as 'radar' key
    legacy_mode = False
    if not product_keys:
        if 'radar_scan' in linked_df.columns:
            product_keys = ['radar']
            legacy_mode = True
        else:
            raise KeyError("linked_df must contain at least one '<key>_scan' column (e.g. 'dhr_scan') or 'radar_scan'")

    # choose reference key: prefer dhr (reflectivity) if present
    reference_key = 'dhr' if 'dhr' in product_keys else product_keys[0]

    if debug:
        print(f"[build_bboxes] product_keys={product_keys}, reference_key={reference_key}, legacy_mode={legacy_mode}")

    collected = []
    pseudocomps_by_idx = {}

    for idx, row in linked_df.iterrows():
        try:
            # Load the reference composite (try in-memory first, then cache)
            ref_scan_col = (f"{reference_key}_scan" if not legacy_mode else 'radar_scan')
            ref_cache_col = (f"{reference_key}_cache_member_name" if not legacy_mode else 'cache_member_name')

            comp_scan = row.get(ref_scan_col, None)

            # Move on to the next row if comp_scan fails, for whatever reason
            if comp_scan is None:
                if debug:
                    print(f"[build_bboxes] idx {idx}: no reference scan ({ref_scan_col}) -> skipping")
                continue

            # ### NEW: build a pseudo-composite reflectivity for Level II (or pass-through for Level III)
            try:
                pseudo = _make_reflectivity_pseudocomposite(
                    comp_scan,
                    field_name=class_field,
                    out_field=class_field,     # keep same so downstream code uses class_field unchanged
                    max_tilts=3,
                    chunk_size=2048,
                    debug=debug,
                    plot_dir=(debug_plot_dir if debug else None),
                    plot_stub=(f"row_{idx}_pseudo")
                )
                pseudocomps_by_idx[idx] = pseudo
                comp_scan = pseudo      # use pseudo for bbox finding
            except Exception as e:
                if debug:
                    print(f"[build_bboxes] idx {idx}: pseudo-composite failed ({e}); falling back to given scan")
                pseudocomps_by_idx[idx] = None
                comp_scan = comp_scan   # fallback (e.g., Level III single-tilt/composite)

            # obtain center lat/lon
            center_lat = float(row['latitude'])
            center_lon = float(row['longitude'])
            center_lon = ((center_lon + 180.0) % 360.0) - 180.0

            if debug:
                print(f"[build_bboxes] processing idx {idx} center=({center_lat:.4f},{center_lon:.4f}) using ref='{reference_key}'")

            # find blob bbox using the reference composite (identical call as before)
            minlat, maxlat, minlon, maxlon, centers, info = _find_blob_for_point(
                comp_scan,
                center_lat, center_lon,
                class_field=class_field,
                threshold=threshold,
                min_size=min_size,
                pad_m=pad_km * 1000.0,
                grid_res_m=grid_res_m,
                include_nearby_km=include_nearby_km,
                debug=debug,
                plot_dir=(debug_plot_dir if debug else None),
                plot_stub=(f"row_{idx}_find_blob")
            )


            if minlat is None:
                if debug:
                    print(f"[build_bboxes] idx {idx}: _find_blob_for_point returned None -> skipping")
                continue

            # NEW: per-scan bbox print
            if debug:
                print(f"[build_bboxes] idx {idx}: per-scan bbox "
                      f"lat [{minlat:.4f}, {maxlat:.4f}] lon [{minlon:.4f}, {maxlon:.4f}]")

            t_geog2xy = info['t_geog2xy']

            # convert lon/lat bbox to metric bbox
            minx, miny = t_geog2xy.transform(minlon, minlat)
            maxx, maxy = t_geog2xy.transform(maxlon, maxlat)

            metric_minx, metric_maxx = (min(minx, maxx), max(minx, maxx))
            metric_miny, metric_maxy = (min(miny, maxy), max(miny, maxy))

            collected.append({
                'idx': idx, 'row': row, 'comp_ref': comp_scan,
                'minlat': minlat, 'maxlat': maxlat, 'minlon': minlon, 'maxlon': maxlon,
                'minx': metric_minx, 'maxx': metric_maxx, 'miny': metric_miny, 'maxy': metric_maxy,
                'centers': centers
            })
        except Exception as e:
            if debug:
                print(f"[build_bboxes] error idx {idx}: {e}")
            continue

    if len(collected) == 0:
        if debug:
            print("[build_bboxes] no valid rows after scanning")
        cols = list(linked_df.columns) + ['min_lat', 'max_lat', 'min_lon', 'max_lon']
        return pd.DataFrame(columns=cols)

    # compute widths/heights in meters (from the metric boxes we computed)
    widths = np.array([c['maxx'] - c['minx'] for c in collected], dtype=float)
    heights = np.array([c['maxy'] - c['miny'] for c in collected], dtype=float)

    avg_width_m, avg_height_m, weights = _compute_weighted_dimensions(widths, heights, debug=debug)


    # fit averaged box to first and last reference scans
    search_km = 20.0
    step_km = max(1.0, (grid_res_m / 1000.0) / 2.0)

    first_box = (collected[0]['minlat'], collected[0]['maxlat'],
                collected[0]['minlon'], collected[0]['maxlon'])
    last_box  = (collected[-1]['minlat'], collected[-1]['maxlat'],
                collected[-1]['minlon'], collected[-1]['maxlon'])

    first_center_hint = _bbox_center(*first_box)
    last_center_hint  = _bbox_center(*last_box)

    first  = collected[0]['comp_ref']
    last   = collected[-1]['comp_ref']


    first_fit = _fit_box_to_scan_max_overlap(first, avg_width_m, avg_height_m,
                                             class_field=class_field, threshold=threshold,
                                             search_km=search_km, step_km=step_km, center_hint=first_center_hint, debug=debug)
    last_fit  = _fit_box_to_scan_max_overlap(last, avg_width_m, avg_height_m,
                                             class_field=class_field, threshold=threshold,
                                             search_km=search_km, step_km=step_km, center_hint=last_center_hint, debug=debug)

    # first/last fit prints
    if debug:
        if first_fit is not None and last_fit is not None:
            fminlat, fmaxlat, fminlon, fmaxlon = first_fit
            lminlat, lmaxlat, lminlon, lmaxlon = last_fit
            print(f"[build_bboxes] first_fit lat [{fminlat:.4f}, {fmaxlat:.4f}] "
                  f"lon [{fminlon:.4f}, {fmaxlon:.4f}]")
            print(f"[build_bboxes] last_fit  lat [{lminlat:.4f}, {lmaxlat:.4f}] "
                  f"lon [{lminlon:.4f}, {lmaxlon:.4f}]")
        else:
            print("[build_bboxes] first/last fit is None (will fallback to union of per-scan bboxes)")


    # fallback -> union of per-row lon/lat extents if fit fails
    if (first_fit is None) or (last_fit is None):
        if debug:
            print("[build_bboxes] fit failed for first/last; falling back to union of per-row lat/lon extents")
        all_minlat = min(c['minlat'] for c in collected)
        all_maxlat = max(c['maxlat'] for c in collected)
        all_minlon = min(c['minlon'] for c in collected)
        all_maxlon = max(c['maxlon'] for c in collected)
        final_minlat, final_maxlat, final_minlon, final_maxlon = all_minlat, all_maxlat, all_minlon, all_maxlon
    else:
        fminlat, fmaxlat, fminlon, fmaxlon = first_fit
        lminlat, lmaxlat, lminlon, lmaxlon = last_fit
        final_minlat = min(fminlat, lminlat)
        final_maxlat = max(fmaxlat, lmaxlat)
        final_minlon = min(fminlon, lminlon)
        final_maxlon = max(fmaxlon, lmaxlon)

    if debug:
        print(f"[build_bboxes] stationary bbox lat [{final_minlat:.4f}, {final_maxlat:.4f}] lon [{final_minlon:.4f}, {final_maxlon:.4f}]")

    # crop every scan (for every product key) to the stationary bbox (apply buffer_km) and collect output rows
    out_rows = []
    for c in collected:
        try:
            out = c['row'].to_dict()
            out['min_lat'] = final_minlat
            out['max_lat'] = final_maxlat
            out['min_lon'] = final_minlon
            out['max_lon'] = final_maxlon

            # For each product key present in the original linked_df, crop its comp (loading from cache if necessary)
            for key in product_keys:
                scan_col = (f"{key}_scan" if not (legacy_mode and key == 'radar') else 'radar_scan')
                cache_col = (f"{key}_cache_member_name" if not (legacy_mode and key == 'radar') else 'cache_member_name')

                comp = out.get(scan_col, None)
                if comp is None:
                    # attempt to load from cache
                    pkl_path = out.get(cache_col, None)
                    if pkl_path:
                        try:
                            comp = _load_composite_pickle(pkl_path, debug=debug)
                            if debug:
                                print(f"[build_bboxes] idx {c['idx']}: loaded comp for key={key} from cache {pkl_path}")
                        except Exception:
                            comp = None

                if comp is None:
                    # no composite available for this product on this row -> keep column but set to None
                    out[scan_col] = None
                else:
                    try:
                        cropped = _crop_comp_scan_to_bbox(comp, final_minlat, final_maxlat, final_minlon, final_maxlon,
                                                         buffer_km=buffer_km, debug=debug, inplace=True)
                        out[scan_col] = cropped
                    except Exception as e:
                        if debug:
                            print(f"[build_bboxes] cropping fail for idx {c['idx']}, key={key}: {e}")
                        out[scan_col] = None

            # ### NEW: also crop and attach the pseudo-composite (built per-row from reference scan)
            pseudo_comp = pseudocomps_by_idx.get(c['idx'], None)
            if pseudo_comp is not None:
                try:
                    pseudo_cropped = _crop_comp_scan_to_bbox(
                        pseudo_comp, final_minlat, final_maxlat, final_minlon, final_maxlon,
                        buffer_km=buffer_km, debug=debug, inplace=True
                    )
                except Exception as e:
                    if debug:
                        print(f"[build_bboxes] pseudo-crop fail idx {c['idx']}: {e}")
                    pseudo_cropped = None
            else:
                pseudo_cropped = None
            out['reflectivity_composite_scan'] = pseudo_cropped  # <- NEW COLUMN
            out_rows.append(out)

        except Exception as e:
            if debug:
                print(f"[build_bboxes] cropping fail idx {c['idx']}: {e}")

    if len(out_rows) == 0:
        if debug:
            print("[build_bboxes] no rows after final cropping")
        cols = list(linked_df.columns) + ['min_lat', 'max_lat', 'min_lon', 'max_lon']
        return pd.DataFrame(columns=cols)

    out_df = pd.DataFrame(out_rows)

    # final debug plotting: save up to N samples
    if debug and debug_plot_dir:
        try:
            import matplotlib.pyplot as plt
            from pyart.graph import RadarDisplay

            sample_is = list(range(len(collected)))
            if len(sample_is) > debug_plot_limit:
                # keep first, mid, last and a few evenly spaced in-between
                first, last = 0, len(collected)-1
                mids = np.linspace(1, last-1, num=min(debug_plot_limit-2, max(0, len(collected)-2)), dtype=int)
                sample_is = [first] + sorted(set(mids.tolist())) + [last]

            for si in sample_is:
                p = collected[si]
                comp_scan = p['comp_ref']

                fig, axes = _plt.subplots(1, 2, figsize=(14, 6))
                ax0, ax1 = axes
                try:
                    display = RadarDisplay(comp_scan)
                    sweep = int(getattr(comp_scan, "metadata", {}).get("pseudo_host_sweep", 0))
                    display.plot(class_field, sweep=sweep, ax=ax0, title=f"Reflectivity (sample {si})")
                except Exception:
                    ax0.text(0.5, 0.5, "RadarDisplay failed", ha='center')
                    ax0.set_title("Reflectivity (not available)")

                info = _compute_metric_grid_and_labels(comp_scan, class_field=class_field,
                                                      threshold=threshold, pad_m=pad_km*1000.0,
                                                      grid_res_m=grid_res_m, debug=False)
                GX, GY = np.meshgrid(info['grid_x'], info['grid_y'])
                lon_grid, lat_grid = info['t_xy2geog'].transform(GX, GY)
                lon_grid = ((lon_grid + 180.0) % 360.0) - 180.0

                ax1.pcolormesh(lon_grid, lat_grid, info['grid_mask'].astype(int),
                              shading='auto', cmap='viridis', alpha=0.5)

                # per-row boxes (light gray)
                for q in collected:
                    lon_min_c, lat_min_c = q['minlon'], q['minlat']
                    lon_max_c, lat_max_c = q['maxlon'], q['maxlat']
                    w = lon_max_c - lon_min_c
                    if w < 0: w += 360.0
                    h = lat_max_c - lat_min_c
                    rect = Rectangle((lon_min_c, lat_min_c), w, h,
                                    fill=False, linewidth=1.0, linestyle='-',
                                    edgecolor='lightgray', alpha=0.8, zorder=2)
                    ax1.add_patch(rect)

                # stationary bbox (red dashed)
                width = final_maxlon - final_minlon
                if width < 0: width += 360.0
                height = final_maxlat - final_minlat
                rect_final = Rectangle((final_minlon, final_minlat), width, height,
                                      fill=False, linewidth=3.0, linestyle='--',
                                      edgecolor='red', zorder=12)
                ax1.add_patch(rect_final)

                ax1.set_xlabel("Longitude"); ax1.set_ylabel("Latitude")
                ax1.set_title(f"Stationary bbox overlay (sample {si})")
                ax1.set_aspect('equal', adjustable='box')
                _plt.tight_layout()

                # SAVE into subdir named after this function
                stub = f"summary_sample_{si}"
                _save_plot(fig, debug_plot_dir, "build_bboxes_for_linked_df", stub, debug)
        except Exception as e:
            if debug:
                print(f"[build_bboxes] final debug plotting failed: {e}")

    # return DataFrame with bounding box columns
    cols = list(linked_df.columns) + ['min_lat', 'max_lat', 'min_lon', 'max_lon', 'reflectivity_composite_scan']
    cols = [c for c in cols if c in out_df.columns]
    out_df = out_df[cols]
    if debug:
        print(f"[build_bboxes] returning out_df shape: {out_df.shape}")
        print(out_df.head(50))

    return out_df
