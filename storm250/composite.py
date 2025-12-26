
def _make_reflectivity_pseudocomposite(
    radar: "pyart.core.Radar",
    field_name: str = "reflectivity",
    out_field: str = "reflectivity",      # keep same so downstream class_field='reflectivity' just works
    max_tilts: int = 3,                   # unused here but kept for signature stability
    chunk_size: int = 2048,               # rays per chunk to bound memory; None = no chunking
    debug: bool = False,
    plot_dir: str | None = None,
    plot_stub: str | None = None
) -> "pyart.core.Radar":
    """
    Build a pseudo-composite reflectivity from a subset of tilts:
      - choose all tilts to build true composite,
      - per-gate max across the chosen tilts,
      - returns a *new* Radar object with a single field `out_field`.
    """
    from time import perf_counter

    _t0 = perf_counter()

    if field_name not in radar.fields:
        raise KeyError(f"[pseudo-comp] field '{field_name}' not found in radar.fields")

    nsw = int(radar.nsweeps)
    sw_start = radar.sweep_start_ray_index['data'].astype(int)
    sw_end   = radar.sweep_end_ray_index['data'].astype(int)

    # fixed_angle is the canonical way to order tilts
    _t_fixed0 = perf_counter()
    try:
        fixed = np.asarray(radar.fixed_angle['data']).astype(float)
        if fixed.size != nsw:
            fixed = np.array([
                np.nanmean(radar.elevation['data'][sw_start[i]:sw_end[i]])
                for i in range(nsw)
            ], dtype=float)
    except Exception:
        fixed = np.array([
            np.nanmean(radar.elevation['data'][sw_start[i]:sw_end[i]])
            for i in range(nsw)
        ], dtype=float)
    _t_fixed1 = perf_counter()

    # --- DEBUG: per-sweep timing / size summary ---------------------------------
    _t_summ0 = perf_counter()
    tsec = np.full(nsw, np.nan, dtype=float)
    try:
        t_all = np.asarray(radar.time['data'], dtype=float)
        for i in range(nsw):
            s, e = int(sw_start[i]), int(sw_end[i])
            tsec[i] = float(np.nanmedian(t_all[s:e])) if e > s else np.nan
    except Exception:
        pass

    if debug:
        print("[_make_reflectivity_pseudocomposite] sweep summary (idx  elev(deg)  rays  gates  median_t[s]):")
        for i in range(nsw):
            s, e = int(sw_start[i]), int(sw_end[i])
            rays = int(max(0, e - s))
            try:
                gates = int(radar.fields[field_name]['data'].shape[1])
            except Exception:
                gates = -1
            print(f"    {i:02d}   {float(fixed[i]):5.2f}    {rays:5d}   {gates:5d}   {tsec[i]:10.1f}")
    _t_summ1 = perf_counter()

    # use ALL available tilts, sorted by fixed angle (lowest → highest)
    _t_choose0 = perf_counter()
    order = np.argsort(fixed)
    if order.size == 0:
        raise RuntimeError("[pseudo-comp] no sweeps available")
    chosen = order  # ← all sweeps
    _t_choose1 = perf_counter()

    if debug:
        fa = [float(fixed[i]) for i in chosen]
        ts = [float(tsec[i]) for i in chosen]
        print(f"[_make_reflectivity_pseudocomposite] chosen sweeps={chosen.tolist()} "
              f"fixed_angles={fa} median_t[s]={ts}")
        '''
        #################################################### COMMENT PLOTTING BLOCK OUT ON PRODUCTION RUNS ###############################################################

        # Plot each contributing sweep (as-is in native file) for visual parity
        _t_dbgplotA0 = perf_counter()
        try:
            from pyart.graph import RadarDisplay
            disp = RadarDisplay(radar)
            n = int(len(chosen))
            fig, axes = _plt.subplots(1, n, figsize=(6*n, 5), squeeze=False)
            for j, sw in enumerate(chosen):
                ax = axes[0, j]
                title = f"{field_name} sweep {int(sw)} ({float(fixed[sw]):.1f}°)"
                try:
                    disp.plot(field_name, sweep=int(sw), ax=ax, title=title)
                    ax.set_aspect('equal', adjustable='box')
                except Exception as e:
                    ax.text(0.5, 0.5, f"Plot failed: {e}", ha='center', va='center')
                    ax.set_title(title + " [failed]")
            _plt.tight_layout();
            stub = f"{(plot_stub or 'pseudo')}_contrib_sweeps"
            _save_plot(fig=fig, plot_dir=plot_dir, func_name="_make_reflectivity_pseudocomposite", stub=stub, debug=debug,)

        except Exception as e:
            print(f"[_make_reflectivity_pseudocomposite] debug plotting failed: {e}")
        _t_dbgplotA1 = perf_counter()

        ##############################################################################################################################################################
        '''
    # --- geometry / host sweep selection ----------------------------------------
    _t_host0 = perf_counter()
    rays_per_sweep, gates_per_sweep = [], []
    for i in chosen:
        s, e = int(sw_start[i]), int(sw_end[i])
        rays_per_sweep.append(int(max(0, e - s)))
        gates_per_sweep.append(int(radar.fields[field_name]['data'].shape[1]))

    host_idx_in_chosen = int(np.argmax(rays_per_sweep))
    host_sweep = int(chosen[host_idx_in_chosen])

    ngates_common = int(min(gates_per_sweep))
    if ngates_common <= 0:
        raise RuntimeError("[pseudo-comp] no common gate count across chosen sweeps")
    _t_host1 = perf_counter()

    if debug:
        print(f"[_make_reflectivity_pseudocomposite] host_sweep={host_sweep} "
              f"(elev={float(fixed[host_sweep]):.2f}°), rays_host={rays_per_sweep[host_idx_in_chosen]}, "
              f"ngates_common={ngates_common}")

    base_ma = radar.fields[field_name]['data']
    if not isinstance(base_ma, _ma.MaskedArray):
        base_ma = _ma.MaskedArray(base_ma, mask=np.zeros_like(base_ma, dtype=bool))

    _t_hostaz0 = perf_counter()
    hs, he = int(sw_start[host_sweep]), int(sw_end[host_sweep])
    rays_host = int(max(0, he - hs))
    host_az = np.array(radar.azimuth['data'][hs:he], dtype=float)
    host_u = np.unwrap(np.deg2rad(host_az))  # radians, ~ span ≈ 2π
    if debug:
        dhost = np.diff(host_u)
        nonmono = int(np.sum(dhost < 0))
        print(f"[_make_reflectivity_pseudocomposite] host az unwrap: rays={host_u.size}, "
              f"span(deg)={np.degrees(host_u[-1]-host_u[0]):.2f}, non-monotonic diffs={nonmono}")
    _t_hostaz1 = perf_counter()

    # --- accumulation & (optional) per-sweep reindexed debug stacks -------------
    out_mask = np.ones((rays_host, ngates_common), dtype=bool)
    acc_data = None
    acc_valid_any = None

    # Optional: store reindexed fields for "winner" map (guard memory)
    store_reindexed = bool(debug) and (rays_host * ngates_common <= 4_000_000)  # ~16MB/sweep (float32)
    reindexed_stack = [] if store_reindexed else None

    _t_accum0 = perf_counter()
    per_sweep_map_ms = []
    per_sweep_accum_ms = []
    for i in chosen:
        s, e = int(sw_start[i]), int(sw_end[i])

        # ---------- Mapping (source sweep → host rays) ----------
        _t_map0 = perf_counter()

        # Source sweep azimuths (unwrapped absolute angles)
        src_az = np.array(radar.azimuth['data'][s:e], dtype=float)
        src_u  = np.unwrap(np.deg2rad(src_az))  # radians, length ≈ 2π but offset vs host

        # Align only by integer 2π cycles so src & host live on same unwrap "cycle"
        two_pi = 2.0 * np.pi
        k = int(np.round((host_u.mean() - src_u.mean()) / two_pi))
        src_u_aligned = src_u + k * two_pi

        # Sort once; keep mapping to original ray indices
        order_idx = np.argsort(src_u_aligned)
        src_sorted = src_u_aligned[order_idx]
        src_sorted = np.maximum.accumulate(src_sorted)  # guard tiny non-monotonic dips
        idx_sorted = order_idx  # original ray index at each sorted position
        N = src_sorted.size

        # ---------- PERIODIC EXTENSION (prevents endpoint smearing) ----------
        src_ext = np.concatenate([src_sorted - two_pi, src_sorted, src_sorted + two_pi])
        idx_ext = np.concatenate([idx_sorted,       idx_sorted,       idx_sorted      ])
        M = src_ext.size  # = 3N

        # Nearest-neighbor on the extended axis (no clamping to endpoints)
        j = np.searchsorted(src_ext, host_u)  # insertion index
        j0 = np.clip(j - 1, 0, M - 1)
        j1 = np.clip(j,     0, M - 1)
        d0 = np.abs(host_u - src_ext[j0])
        d1 = np.abs(host_u - src_ext[j1])
        pick_ext = np.where(d0 <= d1, j0, j1)     # nearest neighbor in the extended axis
        src_idx  = idx_ext[pick_ext]              # map back to ORIGINAL ray indices (0..N_src-1)

        # ---------- OPTIONAL: GAP GUARD (don’t bridge real azimuth holes) ----------
        gap_deg = 6.0  # tune as needed; 4–8° works well
        gap_rad = np.deg2rad(gap_deg)

        prev_ext = np.clip(pick_ext - 1, 0, M - 1)
        next_ext = np.clip(pick_ext + 1, 0, M - 1)
        left_span  = np.abs(src_ext[pick_ext] - src_ext[prev_ext])
        right_span = np.abs(src_ext[next_ext] - src_ext[pick_ext])
        local_gap  = np.maximum(left_span, right_span)  # conservative local spacing
        bad_bridge = local_gap > gap_rad                # too sparse => likely a real gap

        if debug:
            # How bad was it before the fix?
            rr = src_idx.copy()
            rr[bad_bridge] = -1
            max_run = 0
            run = 0
            prev = None
            for v in rr:
                if v >= 0 and v == prev:
                    run += 1
                else:
                    max_run = max(max_run, run)
                    run = 1
                prev = v
            max_run = max(max_run, run)
            frac_bad = float(np.mean(bad_bridge)) if bad_bridge.size else 0.0
            print(f"[_make_reflectivity_pseudocomposite] map sweep {i:02d}→host {host_sweep:02d}: "
                  f"k={k:+d}, max_identical_src_run={max_run}, gap_mask_frac={frac_bad:.3f}")

        _t_map1 = perf_counter()

        # ---------- Accumulation (max over sweeps) ----------
        _t_acc0 = perf_counter()

        src_field = base_ma[s:e, :ngates_common]

        # For bad_bridge positions, we’ll force invalid by clearing 'valid' later.
        src_idx_clip = np.clip(src_idx, 0, src_field.shape[0]-1)

        # Accumulator for winner map (if enabled)
        sweep_acc = None
        if store_reindexed:
            sweep_acc = np.full((rays_host, ngates_common), -np.inf, dtype=np.float32)

        gate_chunk = ngates_common if (chunk_size is None or chunk_size <= 0) else int(max(1, min(chunk_size, ngates_common)))
        for g0 in range(0, ngates_common, gate_chunk):
            g1 = min(g0 + gate_chunk, ngates_common)

            ch     = src_field[:, g0:g1]           # (rays_src, gates_chunk)
            ch_re  = ch[src_idx_clip, :]           # (rays_host, gates_chunk)
            ch_dat = ch_re.filled(np.nan).astype(np.float32, copy=False)
            ch_msk = _ma.getmaskarray(ch_re)

            # Valid data from the source AND not crossing a detected gap
            valid_map = (~ch_msk) & np.isfinite(ch_dat) & (~bad_bridge[:, None])

            if acc_data is None:
                acc_data      = np.full((rays_host, ngates_common), -np.inf, dtype=np.float32)
                acc_valid_any = np.zeros((rays_host, ngates_common), dtype=bool)

            work = np.where(valid_map, ch_dat, -np.inf)
            acc_data[:, g0:g1]      = np.maximum(acc_data[:, g0:g1], work)
            acc_valid_any[:, g0:g1] |= valid_map

            if store_reindexed:
                sweep_acc[:, g0:g1] = work  # -inf where invalid

        if store_reindexed:
            reindexed_stack.append(sweep_acc)

        _t_acc1 = perf_counter()

        per_sweep_map_ms.append((_t_map1 - _t_map0) * 1000.0)
        per_sweep_accum_ms.append((_t_acc1 - _t_acc0) * 1000.0)

    _t_accum1 = perf_counter()

    # finalize masked result on HOST grid
    _t_finalize0 = perf_counter()
    out_data = acc_data
    out_mask = ~acc_valid_any
    out_ma = _ma.MaskedArray(out_data, mask=out_mask)
    _t_finalize1 = perf_counter()
    '''
    ######################################################## COMMENT PLOTTING BLOCK OUT ON PRODUCTION RUNS ########################################################

    # --- DEBUG: winner map (which sweep contributed the max per gate) -----------
    _t_winner0 = perf_counter()
    if debug and store_reindexed and len(reindexed_stack) >= 1:
        stack = np.stack(reindexed_stack, axis=0)  # (ns, rays_host, ngates_common)
        winner = np.argmax(stack, axis=0)          # (rays_host, ngates_common)
        maxval = np.max(stack, axis=0)
        # gates with no valid contributor: mark as -1
        winner = np.where(np.isfinite(maxval) & (maxval > -1e20), winner, -1)

        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 1, figsize=(7, 5))
            im = ax.imshow(winner.T, origin='lower', aspect='auto', interpolation='nearest')
            ax.set_title("Winner map (which sweep provides max) [index in chosen]")
            ax.set_xlabel("ray (host az index)"); ax.set_ylabel("gate")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            plt.tight_layout();

            # Save to local directory
            stub = f"{(plot_stub or 'pseudo')}_winner_map"
            _save_plot(fig, plot_dir, "_make_reflectivity_pseudocomposite", stub, debug)

        except Exception as e:
            print(f"[_make_reflectivity_pseudocomposite] winner-map plotting failed: {e}")
    _t_winner1 = perf_counter()

    #############################################################################################################################################################
    '''
    # skeleton radar with single host sweep
    _t_skel0 = perf_counter()
    skel = radar.deepcopy() if hasattr(radar, "deepcopy") else pickle.loads(pickle.dumps(radar, -1))

    # Slice per-ray series down to the host sweep
    for key in ("time", "azimuth", "elevation", "sweep_number"):
        if key in skel.__dict__ and "data" in skel.__dict__[key]:
            arr = skel.__dict__[key]["data"]
            try:
                skel.__dict__[key]["data"] = arr[hs:he]
            except Exception:
                pass

    try:
        skel.fixed_angle["data"] = np.array([float(radar.fixed_angle["data"][host_sweep])], dtype=float)
    except Exception:
        skel.fixed_angle["data"] = np.array([float(np.nanmean(radar.elevation["data"][hs:he]))], dtype=float)

    try:
        skel.range["data"] = np.array(radar.range["data"][:ngates_common], copy=True)
    except Exception:
        pass

    skel.nsweeps = 1
    skel.sweep_start_ray_index["data"] = np.array([0], dtype=int)
    skel.sweep_end_ray_index["data"]   = np.array([rays_host], dtype=int)

    for gate_key in ("gate_longitude", "gate_latitude", "gate_altitude"):
        if hasattr(skel, gate_key):
            try:
                delattr(skel, gate_key)
            except Exception:
                pass
    _t_skel1 = perf_counter()

    _t_gate0 = perf_counter()
    skel.init_gate_longitude_latitude()
    if not hasattr(skel, 'gate_altitude'):
        skel.init_gate_altitude()
    _t_gate1 = perf_counter()

    _t_fields0 = perf_counter()
    skel.fields = {
        out_field: {
            "data": out_ma.astype(np.float32, copy=False),
            "long_name": "composite reflectivity (host sweep only)",
            "standard_name": out_field,
            "units": radar.fields[field_name].get("units", "dBZ")
        }
    }
    skel.metadata = dict(getattr(skel, "metadata", {}) or {})
    skel.metadata["pseudo_host_sweep"] = 0
    _t_fields1 = perf_counter()
    '''
    ######################################################## COMMENT PLOTTING BLOCK OUT ON PRODUCTION RUNS ########################################################

    # --- DEBUG: final product sanity plots --------------------------------------
    _t_dbgplotB0 = perf_counter()
    if debug:
        try:
            from pyart.graph import RadarDisplay
            import matplotlib.pyplot as plt
            disp2 = RadarDisplay(skel)
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            axA, axB = axes

            # Final pseudo-comp in km coords
            disp2.plot(out_field, sweep=0, ax=axA, title="Pseudo-composite (host grid)")
            axA.set_aspect('equal', adjustable='box')

            # Threshold scatter of the final host product in lon/lat
            skel.init_gate_longitude_latitude()
            glat = skel.gate_latitude['data'][0: rays_host, :ngates_common]
            glon = skel.gate_longitude['data'][0: rays_host, :ngates_common]
            data = skel.fields[out_field]['data']
            data_arr = data.filled(-9999.0) if isinstance(data, _ma.MaskedArray) else data

            thr = 20.0  # or pass through from caller if you prefer
            mask = (data_arr >= thr) & np.isfinite(glat) & np.isfinite(glon)
            idx = np.flatnonzero(mask.ravel())
            if idx.size > 100000:
                idx = idx[:: int(idx.size // 100000) + 1]

            axB.scatter(glon.ravel()[idx], glat.ravel()[idx], s=2, marker='.', alpha=0.6, zorder=5)
            axB.set_title(f"Final host gates ≥ {thr} dBZ (lon/lat)")
            axB.set_xlabel("Longitude"); axB.set_ylabel("Latitude")
            axB.set_aspect('equal', adjustable='box')

            plt.tight_layout();
            stub = f"{(plot_stub or 'pseudo')}_final_host"
            _save_plot(fig, plot_dir, "_make_reflectivity_pseudocomposite", stub, debug)

        except Exception as e:
            print(f"[_make_reflectivity_pseudocomposite] final plotting failed: {e}")
    _t_dbgplotB1 = perf_counter()

    ################################################################################################################################################################
    '''

    # --------- PERF SUMMARY ------------------------------------------------------
    if debug:
        total_ms = (perf_counter() - _t0) * 1000.0
        print(f"[_make_reflectivity_pseudocomposite] PERF:")
        print(f"    fixed-angle compute:         {(_t_fixed1  - _t_fixed0 )*1000:.1f} ms")
        print(f"    sweep summary (timing):      {(_t_summ1   - _t_summ0  )*1000:.1f} ms")
        print(f"    choose sweeps:               {(_t_choose1 - _t_choose0)*1000:.1f} ms")
        print(f"    host selection + dims:       {(_t_host1   - _t_host0  )*1000:.1f} ms")
        print(f"    host azimuth unwrap:         {(_t_hostaz1 - _t_hostaz0)*1000:.1f} ms")
        print(f"    accumulation loop (total):   {(_t_accum1  - _t_accum0 )*1000:.1f} ms")
        if per_sweep_map_ms:
            print(f"        per-sweep mapping avg:   {np.mean(per_sweep_map_ms):.1f} ms  (n={len(per_sweep_map_ms)})")
            print(f"        per-sweep mapping sum:   {np.sum(per_sweep_map_ms):.1f} ms")
        if per_sweep_accum_ms:
            print(f"        per-sweep accumulate avg:{np.mean(per_sweep_accum_ms):.1f} ms  (n={len(per_sweep_accum_ms)})")
            print(f"        per-sweep accumulate sum:{np.sum(per_sweep_accum_ms):.1f} ms")
        print(f"    finalize masked result:      {(_t_finalize1- _t_finalize0)*1000:.1f} ms")
        print(f"    winner-map (debug):          {((_t_winner1 - _t_winner0 )*1000 if '_t_winner1' in locals() else 0):.1f} ms")
        print(f"    skeleton build/slice:        {(_t_skel1   - _t_skel0  )*1000:.1f} ms")
        print(f"    gate lon/lat/alt init:       {(_t_gate1   - _t_gate0  )*1000:.1f} ms")
        print(f"    set fields/metadata:         {(_t_fields1 - _t_fields0)*1000:.1f} ms")
        print(f"    debug plots (chosen sweeps): {((_t_dbgplotA1 - _t_dbgplotA0)*1000 if '_t_dbgplotA1' in locals() else 0):.1f} ms")
        print(f"    debug plots (final host):    {((_t_dbgplotB1 - _t_dbgplotB0)*1000 if '_t_dbgplotB1' in locals() else 0):.1f} ms")
        print(f"    TOTAL:                       {total_ms:.1f} ms")

    return skel
