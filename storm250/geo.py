# For usage in _compute_metric_grid_and_labels
@functools.lru_cache(maxsize=64)
def _cached_aeqd_transformers(lat0_round6, lon0_round6):
    lat0 = float(lat0_round6)
    lon0 = float(lon0_round6)
    proj_str = f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +units=m +datum=WGS84"
    t_geog2xy = Transformer.from_crs("EPSG:4326", proj_str, always_xy=True)
    t_xy2geog = Transformer.from_crs(proj_str, "EPSG:4326", always_xy=True)
    return t_geog2xy, t_xy2geog


# small helper: normalize longitudes to be continuous around a center lon
def _normalize_lons_to_center(lons, center_lon):
    """
    Normalize lon(s) so they are in the range [center_lon-180, center_lon+180),
    preserving continuity around center_lon (works with scalars or numpy arrays).
    """
    # broadcast to numpy array
    lon_arr = np.array(lons)
    # compute centered difference in (-180,180]
    diff = ((lon_arr - center_lon + 180.0) % 360.0) - 180.0
    return center_lon + diff


def _compute_metric_grid_and_labels(comp_scan,
                                    class_field='reflectivity',
                                    threshold=20,
                                    pad_m=5000.0,
                                    grid_res_m=1000.0,
                                    debug=False):
    _t0 = perf_counter()

    # locate sweep & read field for thresholding
    sweep = int(getattr(comp_scan, "metadata", {}).get("pseudo_host_sweep", 0))
    start = comp_scan.sweep_start_ray_index['data'][sweep]
    stop  = comp_scan.sweep_end_ray_index['data'][sweep]

    _t_fields0 = perf_counter()
    dbz_cls = comp_scan.fields[class_field]['data'][start:stop, :]
    _t_fields1 = perf_counter()

    _t_geo0 = perf_counter()
    comp_scan.init_gate_longitude_latitude()
    lats_full = comp_scan.gate_latitude['data'][start:stop, :]
    lons_full = comp_scan.gate_longitude['data'][start:stop, :]

    # Canonical roll to match plan ordering
    az = np.asarray(comp_scan.azimuth['data'][start:stop], dtype=float)
    roll = _ray_canonical_roll(az)
    lats = np.roll(lats_full, -roll, axis=0)
    lons = np.roll(lons_full, -roll, axis=0)
    _t_geo1 = perf_counter()

    # Fill masked -> ndarray if needed
    _t_fill0 = perf_counter()
    if isinstance(dbz_cls, np.ma.MaskedArray):
        if debug: print("[_compute_metric_grid_and_labels] dbz_cls is MaskedArray -> filled")
        dbz_cls = dbz_cls.filled(0)
    # keep data rolled to canonical order too
    dbz_cls = np.roll(dbz_cls, -roll, axis=0)
    _t_fill1 = perf_counter()

    # threshold mask (in canonical order)
    _t_thr0 = perf_counter()
    mask2d = (dbz_cls >= threshold)
    _t_thr1 = perf_counter()

    # respect invalid geo gates (canonical order)
    _t_bad0 = perf_counter()
    bad_geo = ~np.isfinite(lats) | ~np.isfinite(lons)
    if np.any(bad_geo):
        mask2d = np.where(bad_geo, False, mask2d)
        if debug: print(f"[_compute_metric_grid_and_labels] masked out {int(bad_geo.sum())} invalid geo gates")
    valid_geo = ~bad_geo
    _t_bad1 = perf_counter()

    # plan (order-invariant); pre-check cache hit status for debug
    cache_hit = None
    if debug:
        try:
            _key_probe = _grid_plan_key(comp_scan, grid_res_m, pad_m, class_field)
            cache_hit = _key_probe in _GRID_PLAN_CACHE
        except Exception:
            cache_hit = None

    _t_plan0 = perf_counter()
    plan = _get_or_build_grid_plan(comp_scan, class_field, grid_res_m, pad_m, debug=debug)
    ny, nx = plan['ny'], plan['nx']
    _t_plan1 = perf_counter()

    # map mask → grid
    _t_flat0 = perf_counter()
    # (kept for compatibility with old logs even if not used in new path)
    # valid_geo_n etc. still computed from 'valid_geo'
    _t_flat1 = perf_counter()

    _t_map0 = perf_counter()
    if ('grid_ray' in plan) and ('grid_gate' in plan):
        # order-invariant pair indexing path
        grid_vals = mask2d[plan['grid_ray'], plan['grid_gate']].reshape(ny, nx)
    else:
        # legacy fallback path
        mask_flat = mask2d.ravel()[valid_geo.ravel()].astype(np.uint8, copy=False)
        grid_vals = mask_flat[plan['idx_grid']].reshape(ny, nx)
    _t_map1 = perf_counter()

    # morphology + labeling
    _t_morph0 = perf_counter()
    struct_xy = np.ones((3, 3), dtype=bool)
    grid_mask = binary_closing(grid_vals >= 1, structure=struct_xy)
    _t_morph1 = perf_counter()

    _t_label0 = perf_counter()
    labeled_grid, n_blob = label(grid_mask)
    _t_label1 = perf_counter()

    if debug:
        rays = int(stop - start)
        gates = int(dbz_cls.shape[1]) if hasattr(dbz_cls, "shape") and len(dbz_cls.shape) >= 2 else -1
        valid_geo_n = int(valid_geo.sum()) if hasattr(valid_geo, "sum") else -1
        cache_note = ("hit" if cache_hit else "miss") if cache_hit is not None else "n/a"
        print(f"[_compute_metric_grid_and_labels] dims: sweep={sweep} rays={rays} gates={gates}  "
              f"grid={ny}x{nx} ({ny*nx} px)  valid_geo={valid_geo_n}  plan_cache={cache_note}")
        print(f"    read fields:          {(_t_fields1 - _t_fields0)*1000:.1f} ms")
        print(f"    init/get lon/lat:     {(_t_geo1    - _t_geo0   )*1000:.1f} ms")
        print(f"    fill masked -> array: {(_t_fill1   - _t_fill0  )*1000:.1f} ms")
        print(f"    threshold compare:    {(_t_thr1    - _t_thr0   )*1000:.1f} ms")
        print(f"    bad_geo apply:        {(_t_bad1    - _t_bad0   )*1000:.1f} ms")
        print(f"    get/build plan:       {(_t_plan1   - _t_plan0  )*1000:.1f} ms")
        print(f"    flatten/select:       {(_t_flat1   - _t_flat0  )*1000:.1f} ms")
        # label kept for backward-compatibility even though we now use (ray,gate) pairs
        print(f"    map→grid (idx_grid):  {(_t_map1    - _t_map0   )*1000:.1f} ms")
        print(f"    binary_closing:       {(_t_morph1  - _t_morph0 )*1000:.1f} ms")
        print(f"    labeling:             {(_t_label1  - _t_label0 )*1000:.1f} ms")
        print(f"    TOTAL:                {(perf_counter() - _t0   )*1000:.1f} ms")
        print(f"[_compute_metric_grid_and_labels] labeled blobs on grid: {n_blob}")

    return {
        'labeled_grid': labeled_grid,
        'grid_x': plan['grid_x'],
        'grid_y': plan['grid_y'],
        't_xy2geog': plan['t_xy2geog'],
        't_geog2xy': plan['t_geog2xy'],
        'xmin': plan['xmin'], 'xmax': plan['xmax'], 'ymin': plan['ymin'], 'ymax': plan['ymax'],
        'grid_mask': grid_mask
    }


def _fast_metric_inside_mask(comp, minlat, maxlat, minlon, maxlon, buffer_km):
    # 1) local projection centered at radar
    try:
        rlat = float(comp.latitude['data']) if np.isscalar(comp.latitude['data']) else float(comp.latitude['data'][0])
        rlon = float(comp.longitude['data']) if np.isscalar(comp.longitude['data']) else float(comp.longitude['data'][0])
    except Exception:
        # worst case: fall back to Py-ART’s georef once (still way rarer than per-crop)
        comp.init_gate_longitude_latitude()
        rlat = float(np.nanmean(comp.gate_latitude['data']))
        rlon = float(np.nanmean(comp.gate_longitude['data']))
    from math import radians, sin, cos, isfinite
    t_g2x, _ = _cached_aeqd_transformers(round(rlat, 6), round(((rlon + 180) % 360) - 180, 6))

    # 2) bbox -> metric (use all 4 corners for safety, buffer already folded into min/max)
    xs, ys = t_g2x.transform(
        [minlon, minlon, maxlon, maxlon],
        [minlat, maxlat, minlat, maxlat]
    )
    minx = float(np.min(xs)); maxx = float(np.max(xs))
    miny = float(np.min(ys)); maxy = float(np.max(ys))

    # 3) geometry vectors
    sweep = int(getattr(comp, "metadata", {}).get("pseudo_host_sweep", 0))
    s = int(comp.sweep_start_ray_index['data'][sweep])
    e = int(comp.sweep_end_ray_index['data'][sweep])

    az = np.asarray(comp.azimuth['data'][s:e], dtype=float)
    el = np.asarray(comp.elevation['data'][s:e], dtype=float)
    rng = np.asarray(comp.range['data'], dtype=float)

    # ground-range per gate (outer product) via cos(elev)
    elc = np.cos(np.deg2rad(el)).astype(np.float32, copy=False)   # (rays,)
    rg  = (elc[:, None] * rng[None, :]).astype(np.float32, copy=False)  # (rays, gates)

    # 4) per-ray range bounds from rectangle
    a  = np.deg2rad(az).astype(np.float32, copy=False)
    sa = np.sin(a); ca = np.cos(a)
    eps = 1e-6

    def bounds_1d(vmin, vmax, denom):
        # returns (lo, hi) per ray for r given vmin <= r*denom <= vmax
        lo = np.full(denom.shape, -np.inf, dtype=np.float32)
        hi = np.full(denom.shape,  np.inf, dtype=np.float32)
        mask = np.abs(denom) >= eps
        q1 = (vmin / denom[mask]).astype(np.float32, copy=False)
        q2 = (vmax / denom[mask]).astype(np.float32, copy=False)
        lo[mask] = np.minimum(q1, q2)
        hi[mask] = np.maximum(q1, q2)
        # if |denom|<eps, line is x=0 or y=0; if 0 in [vmin,vmax] → no constraint; else empty
        pass_mask = (~mask) & (vmin <= 0.0) & (0.0 <= vmax)
        # else keep lo=-inf, hi=+inf (no solution will be filtered later by intersection)
        # For the impossible case (0 not in [vmin,vmax]) we leave lo=-inf,hi=+inf
        return lo, hi

    rx_lo, rx_hi = bounds_1d(minx, maxx, sa)
    ry_lo, ry_hi = bounds_1d(miny, maxy, ca)

    # intersect and clamp to r>=0
    r_lo = np.maximum(0.0, np.maximum(rx_lo, ry_lo))   # (rays,)
    r_hi = np.minimum(     rx_hi, ry_hi)               # (rays,)

    # 5) final inside mask
    inside_mask = (rg >= r_lo[:, None]) & (rg <= r_hi[:, None])
    return inside_mask

def _grid_cache_set(key, plan):
    # move-to-end semantics; evict oldest when full
    _GRID_PLAN_CACHE[key] = plan
    _GRID_PLAN_CACHE.move_to_end(key)
    if len(_GRID_PLAN_CACHE) > _GRID_PLAN_CACHE_MAX:
        _GRID_PLAN_CACHE.popitem(last=False)  # evict LRU


def _grid_plan_key(comp_scan, grid_res_m, pad_m, class_field):
    sweep = int(getattr(comp_scan, "metadata", {}).get("pseudo_host_sweep", 0))

    # radar center
    try:
        rlat = float(comp_scan.latitude['data']) if np.isscalar(comp_scan.latitude['data']) else float(comp_scan.latitude['data'][0])
        rlon = float(comp_scan.longitude['data']) if np.isscalar(comp_scan.longitude['data']) else float(comp_scan.longitude['data'][0])
    except Exception:
        rlat = float(np.nanmean(comp_scan.gate_latitude['data']))
        rlon = float(np.nanmean(comp_scan.gate_longitude['data']))
    rlon = ((rlon + 180.0) % 360.0) - 180.0
    rlat_r = round(rlat, 4); rlon_r = round(rlon, 4)

    start = comp_scan.sweep_start_ray_index['data'][sweep]
    stop  = comp_scan.sweep_end_ray_index['data'][sweep]
    n_rays  = int(stop - start)
    n_gates = int(comp_scan.fields[class_field]['data'].shape[1])

    # Range fingerprint (start and spacing rounded)
    try:
        rng = np.asarray(comp_scan.range['data'][:n_gates], dtype=float)
        r0 = float(rng[0]) if rng.size else 0.0
        dr = float(np.nanmean(np.diff(rng))) if rng.size > 1 else 0.0
    except Exception:
        r0, dr = 0.0, 0.0
    r0_r = round(r0, 1); dr_r = round(dr, 1)

    # NOTE: intentionally no azimuth hash here (too jittery scan-to-scan)
    return (rlat_r, rlon_r, sweep, n_rays, n_gates, float(grid_res_m), float(pad_m), r0_r, dr_r)



def _get_or_build_grid_plan(comp_scan, class_field, grid_res_m, pad_m, debug=False):
    key = _grid_plan_key(comp_scan, grid_res_m, pad_m, class_field)
    if key in _GRID_PLAN_CACHE:
        plan = _GRID_PLAN_CACHE[key]
        _GRID_PLAN_CACHE.move_to_end(key)
        if debug:
            print(f"[_grid_plan] cache HIT → grid={plan['ny']}x{plan['nx']}")
        return plan

    if debug:
        print("[_grid_plan] cache MISS → building plan…")

    sweep = key[2]
    start = comp_scan.sweep_start_ray_index['data'][sweep]
    stop  = comp_scan.sweep_end_ray_index['data'][sweep]

    # Canonicalize ray order (roll so the smallest azimuth ray is first)
    az = np.asarray(comp_scan.azimuth['data'][start:stop], dtype=float)
    roll = _ray_canonical_roll(az)

    comp_scan.init_gate_longitude_latitude()
    lats_full = comp_scan.gate_latitude['data'][start:stop, :]
    lons_full = comp_scan.gate_longitude['data'][start:stop, :]

    # Roll ray axis to canonical order
    lats = np.roll(lats_full, -roll, axis=0)
    lons = np.roll(lons_full, -roll, axis=0)

    # radar center (from key rounding)
    radar_lat, radar_lon = key[0], key[1]

    t_geog2xy, t_xy2geog = _cached_aeqd_transformers(round(radar_lat, 6), round(radar_lon, 6))

    # valid mask in canonical order
    valid_geo_2d = np.isfinite(lons) & np.isfinite(lats)
    if not np.any(valid_geo_2d):
        raise RuntimeError("[_get_or_build_grid_plan] no valid geo gates found")

    # project valid gates once (lon normalized around radar_lon for stability)
    lon_valid = _normalize_lons_to_center(lons[valid_geo_2d], radar_lon)
    lat_valid = lats[valid_geo_2d]
    xx_flat, yy_flat = t_geog2xy.transform(lon_valid, lat_valid)

    # grid extents + arrays
    xmin, xmax = xx_flat.min() - pad_m, xx_flat.max() + pad_m
    ymin, ymax = yy_flat.min() - pad_m, yy_flat.max() + pad_m
    nx = int(np.ceil((xmax - xmin) / grid_res_m)) + 1
    ny = int(np.ceil((ymax - ymin) / grid_res_m)) + 1
    grid_x = np.linspace(xmin, xmax, nx)
    grid_y = np.linspace(ymin, ymax, ny)

    # KDTree over valid gates
    pts = np.column_stack((xx_flat, yy_flat))
    tree = cKDTree(pts)

    # Prepare per-valid-point (ray, gate) indices in canonical order
    n_rays = lats.shape[0]; n_gates = lats.shape[1]
    ray_idx_2d = np.broadcast_to(np.arange(n_rays)[:, None], (n_rays, n_gates))
    gate_idx_2d = np.broadcast_to(np.arange(n_gates)[None, :], (n_rays, n_gates))
    valid_ray_idx = ray_idx_2d[valid_geo_2d].astype(np.int32, copy=False)
    valid_gate_idx = gate_idx_2d[valid_geo_2d].astype(np.int32, copy=False)

    # Query nearest for every grid node in stripes
    grid_ray = np.empty(ny * nx, dtype=np.int32)
    grid_gate = np.empty(ny * nx, dtype=np.int32)
    block_rows = 512 if ny > 512 else ny
    write_ptr = 0
    for y0 in range(0, ny, block_rows):
        y1 = min(y0 + block_rows, ny)
        Xb = np.tile(grid_x, y1 - y0)
        Yb = np.repeat(grid_y[y0:y1], nx)
        _, idx = tree.query(np.column_stack((Xb, Yb)), k=1, workers=-1)
        grid_ray[write_ptr:write_ptr + idx.size]  = valid_ray_idx[idx]
        grid_gate[write_ptr:write_ptr + idx.size] = valid_gate_idx[idx]
        write_ptr += idx.size

    plan = {
        't_geog2xy': t_geog2xy,
        't_xy2geog': t_xy2geog,
        'grid_x': grid_x, 'grid_y': grid_y,
        'nx': nx, 'ny': ny,
        # NEW: order-invariant mapping
        'grid_ray': grid_ray,          # (ny*nx,) int32 → ray index in canonical order
        'grid_gate': grid_gate,        # (ny*nx,) int32 → gate index
        'ray_order_rule': 'min_az_first',  # for clarity/debug
        'xmin': xmin, 'xmax': xmax, 'ymin': ymin, 'ymax': ymax,
        'n_rays': n_rays, 'n_gates': n_gates,
    }
    _GRID_PLAN_CACHE[key] = plan
    if debug:
        print(f"[_grid_plan] built new plan: grid={ny}x{nx}, valid_gates={pts.shape[0]}, roll={roll}, cache_size={len(_GRID_PLAN_CACHE)}")
    return plan
