import os, shutil, gc, warnings, json
from datetime import datetime

import yaml
import h5py
import numpy as np
import numpy.ma as ma
import pandas as pd

def save_df_for_training(
    df: pd.DataFrame,
    base_dir: str,
    year_override: int | None = None,
    radar_site_col: str = "radar_site",
    storm_id_col: str = "storm_id",
    time_col: str = "time",
    debug: bool = True,
    scan_order: list | None = None,
    drop_scans_after_save: bool = True,
    *,
    dataset_version: str = "1.0.0",
    context_schema_yaml_path: str | None = None,
    product_schema_yaml_path: str | None = None,
):
    """
    Save a SEVIR-like training package per storm folder:
      base_dir/{year}/{RADAR_SITE}/storm_{storm_id}/

    Inside each storm folder:
      - one CONTEXT HDF (rows=times; all bbox_df columns except *_scan and *_cache_volume_path)
      - one product HDF per *_scan column, each containing dataset '/data'
        shaped (T, H, W, C) with float32 + NaN, chunked & compressed.

    Notes
    -----
    - H, W are determined per product within the storm by taking the maximum
      (rays, gates) across available sweeps and scans; smaller frames are NaN-padded.
    - Channels C:
        - If comp.metadata['pseudo_host_sweep'] exists  → C = 1 (use sweep 0).
        - Otherwise                                   → C = number of sweeps (nsweeps) in that scan.
      The file’s C is the maximum channels seen across the storm for that product;
      scans with fewer sweeps are padded with NaN channels.
    """
    from time import perf_counter


    def _apply_dataset_attrs(dset_obj, name: str):
        """
        Copy yaml attributes into HDF attributes. 
        
        Allows a user to check the meaning of a value in hdf (ex. f["azimuth_deg"].attrs)
        without looking ay yaml file, while still keeping yaml file as single point of truth;
        yaml is copied into HDF attributes.
        """
        spec = (product_schema_base.get("datasets") or {}).get(name, {}) or {}
        attrs = (spec.get("attrs") or {}) or {}
        for k, v in attrs.items():
            dset_obj.attrs[k] = v
        def _fmt_bytes(n: int) -> str:
            for unit in ["B", "KB", "MB", "GB", "TB"]:
                if abs(n) < 1024.0:
                    return f"{n:.1f} {unit}"
                n /= 1024.0
            return f"{n:.1f} PB"
    
    def _fmt_bytes(n: int) -> str:
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if abs(n) < 1024.0:
                return f"{n:.1f} {unit}"
            n /= 1024.0
        return f"{n:.1f} PB"

    def _load_yaml_schema(path: str | None, *, what: str) -> dict:
        if not path:
            raise ValueError(f"{what} schema YAML path is required.")
        with open(path, "r", encoding="utf-8") as f:
            d = yaml.safe_load(f) or {}
        if "schema_version" not in d or "schema_name" not in d:
            raise ValueError(f"{what} schema YAML must define schema_name and schema_version: {path}")
        return d

    def _pretty_attr(val):
        import numpy as _np
        try:
            if isinstance(val, (bytes, _np.bytes_)):
                return val.decode("utf-8", errors="replace")
            if isinstance(val, (_np.bool_, _np.integer, _np.floating)):
                return val.item()
            if isinstance(val, _np.ndarray):
                if val.ndim == 0:
                    return val.item()
                if val.size > 16:
                    return f"array(shape={val.shape}, dtype={val.dtype})"
                return _np.array2string(val, threshold=16)
            return val
        except Exception:
            return str(val)

    t_all0 = perf_counter()

    if df is None or len(df) == 0:
        if debug:
            print("[save] empty df — nothing to save.")
        return []

    # ---------- load schemes from yaml file ----------
    context_schema_base = _load_yaml_schema(context_schema_yaml_path, what="context")
    product_schema_base = _load_yaml_schema(product_schema_yaml_path, what="product")

    context_schema_version = str(context_schema_base["schema_version"])
    product_schema_name    = str(product_schema_base["schema_name"])
    product_schema_version = str(product_schema_base["schema_version"])

    # ---------- discover product prefixes ----------
    t_disc0 = perf_counter()
    scan_cols = [c for c in df.columns if c.endswith("_scan")]
    if not scan_cols:
        if debug:
            print("[save] no *_scan columns detected — nothing to write.")
        return []
    prefixes = [c[:-5] for c in scan_cols]  # strip _scan
    if scan_order:
        ordered = [p for p in scan_order if p in prefixes]
        ordered += [p for p in prefixes if p not in ordered]
        prefixes = ordered
    t_disc1 = perf_counter()
    if debug:
        print(f"[save] detected products (in order): {prefixes}")
        print(f"[save] discovery time: {(t_disc1 - t_disc0)*1000:.1f} ms")

    # helper: derive year for a row
    def _row_year(row):
        if year_override is not None:
            return int(year_override)
        t = row.get(time_col)
        try:
            tt = pd.to_datetime(t)
            return int(tt.year)
        except Exception:
            return int(datetime.utcnow().year)

    # group by (year, site, storm_id) so each folder has one context HDF + per-product HDFs
    t_grp0 = perf_counter()
    groups = df.groupby([df.apply(_row_year, axis=1), df[radar_site_col], df[storm_id_col]], sort=False)
    t_grp1 = perf_counter()
    if debug:
        print(f"[save] grouping produced {len(groups)} storm group(s) in {(t_grp1 - t_grp0)*1000:.1f} ms")

    saved = []

    # NEW: in-memory debug snapshots
    debug_product_attrs = []   # list of {"path", "file_attrs", "data_attrs"}
    debug_context_peek = []    # list of {"context_path", "columns", "row"}

    # -------------- utilities --------------

    def _choose_field_key(radar_obj, prefix: str):
        """Pick a field key from radar.fields robustly."""
        fkeys = list(getattr(radar_obj, "fields", {}).keys())
        if not fkeys:
            return None
        # exact prefix match
        if prefix in fkeys:
            return prefix
        # common aliases
        prefer = []
        low = prefix.lower()
        if "refl" in low or "dbz" in low or "reflect" in low:
            prefer += ["reflectivity", "corrected_reflectivity"]
        if "vel" in low:
            prefer += ["velocity", "corrected_velocity", "dealiased_velocity"]
        if "spectrum" in low or "width" in low:
            prefer += ["spectrum_width"]
        for cand in prefer:
            if cand in fkeys:
                return cand
        # substring match against available keys
        for k in fkeys:
            if low in k.lower():
                return k
        # fallback: reflectivity if present
        if "reflectivity" in fkeys:
            return "reflectivity"
        # final fallback: first field
        return fkeys[0]

    def _is_pseudo(radar_obj) -> bool:
        try:
            meta = getattr(radar_obj, "metadata", {}) or {}
        except Exception:
            meta = {}
        return "pseudo_host_sweep" in meta

    def _get_sweep_bounds(radar_obj, sw: int):
        """Return (start, end) indices for rays of sweep sw; falls back gracefully."""
        try:
            s = int(radar_obj.sweep_start_ray_index["data"][sw])
            e = int(radar_obj.sweep_end_ray_index["data"][sw])
            return s, e
        except Exception:
            # best effort: whole series
            nrays = int(getattr(radar_obj, "nrays", 0)) or 0
            return 0, nrays

    def _frames_from_scan_all_sweeps(radar_obj, field_key):
        """
        Returns a list of 2-D float32 arrays, one per sweep (or [sweep0] for pseudo),
        each shaped (rays, gates) with NaNs where masked/invalid.
        """
        fdict = radar_obj.fields.get(field_key)
        if fdict is None:
            return []

        data = fdict.get("data")
        if data is None:
            return []

        # Which sweeps?
        if _is_pseudo(radar_obj):
            sweep_indices = [0]
        else:
            try:
                ns = int(getattr(radar_obj, "nsweeps", len(radar_obj.sweep_number["data"])))
            except Exception:
                ns = 1
            sweep_indices = list(range(max(0, ns)))

        out = []
        for sw in sweep_indices:
            s, e = _get_sweep_bounds(radar_obj, sw)
            try:
                sub = data[s:e, :]
            except Exception:
                sub = data  # fallback

            if isinstance(sub, ma.MaskedArray):
                arr = sub.filled(np.nan).astype(np.float32, copy=False)
            else:
                arr = sub.astype(np.float32, copy=False)
                if not np.issubdtype(arr.dtype, np.floating):
                    arr = arr.astype(np.float32, copy=False)

            # ignore degenerate empty slices
            if arr.size == 0 or arr.ndim != 2:
                continue
            out.append(arr)

        return out

    # -------------- main loop --------------
    for (yr, site, storm), sub in groups:
        t_group0 = perf_counter()
        sub = sub.sort_values(time_col).reset_index(drop=True)
        T = len(sub)

        # group summary
        if debug:
            try:
                t0_dbg = pd.to_datetime(sub[time_col].iloc[0]).strftime("%Y-%m-%d %H:%M:%S")
                t1_dbg = pd.to_datetime(sub[time_col].iloc[-1]).strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                t0_dbg = t1_dbg = "NA"
            print(f"\n[save] ===== Group: year={yr} site={site} storm={storm} =====")
            print(f"[save] rows(T)={T} time-span=[{t0_dbg} .. {t1_dbg}]")

        storm_dir = os.path.join(base_dir, str(yr), str(site), f"storm_{storm}")
        os.makedirs(storm_dir, exist_ok=True)

        # ---------- write context HDF ----------
        # context HDF -> one per storm ID, contains all (non-radar) metadata per sweep 
        #       - no *_scan (in memory Py-ART radar) or *_cache_volume_path (points to the pickled skeleton for the field *)
        # 
        #                   /- i.e. what labels fundamentally mean 
        # We don't store semantics of the dataset itself inside the hdf's attributes (unlike with tensor products)
        #    |                                                                               \- these attrs are merely describing the value... doesn't fundamentally change dataset
        #    \- foundational (context hdf) vs descriptive (product hdf)
        t_meta0 = perf_counter()
        context_df = sub.copy()
        drop_cols = [c for c in context_df.columns if c.endswith("_scan") or c.endswith("_cache_volume_path")]
        context_df.drop(columns=drop_cols, inplace=True, errors="ignore")

        # add primary-key component time_unix_ms (UTC, int64 ms since epoch) to storm context hdf
        try:
            t_utc = pd.to_datetime(context_df[time_col], utc=True, errors="coerce")
            context_df["time_unix_ms"] = (t_utc.astype("int64") // 1_000_000)
        except Exception:
            # last-resort fallback; you shouldn't hit this
            context_df["time_unix_ms"] = -1

        # FIX: normalize longitudes to [-180, 180)
        def _wrap180_series(s):
            vals = pd.to_numeric(s, errors="coerce")
            return ((vals + 180.0) % 360.0) - 180.0

        for _lon_col in ["longitude", "tor_lon", "wind_lon", "hail_lon", "min_lon", "max_lon"]:
            if _lon_col in context_df.columns:
                context_df[_lon_col] = _wrap180_series(context_df[_lon_col])

        # FIX: normalize tor_endtime to "" or ISO8601 UTC
        if "tor_endtime" in context_df.columns:
            def _norm_endtime(x):
                if pd.isna(x):
                    return ""
                s = str(x).strip()
                if s in ("", "-", "—", "–", "None", "NA", "nan", "--------------"):
                    return ""
                try:
                    return pd.to_datetime(s, utc=True).strftime("%Y-%m-%dT%H:%M:%SZ")
                except Exception:
                    return ""
            context_df["tor_endtime"] = context_df["tor_endtime"].apply(_norm_endtime)


        # Nice file names: include counts + time span
        try:
            t0 = pd.to_datetime(context_df[time_col].iloc[0]).strftime("%Y%m%dT%H%M%SZ")
            t1 = pd.to_datetime(context_df[time_col].iloc[-1]).strftime("%Y%m%dT%H%M%SZ")
        except Exception:
            t0 = t1 = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

        context_name = f"{site}_{storm}_context_T{T:03d}_{t0}_{t1}.h5"
        context_path = os.path.join(storm_dir, context_name)

        try:
            context_df.to_hdf(context_path, key="context", mode="w", format="table")
            if debug:
                print(f"[save] wrote context → {context_path} (cols={len(context_df.columns)}, rows={len(context_df)})")
        except Exception as e:
            if debug:
                print(f"[save] to_hdf failed for context ({context_path}): {e}. Falling back to pickle.")
            import pickle as _pickle
            with open(os.path.join(storm_dir, f"{site}_{storm}_context_T{T:03d}_{t0}_{t1}.pkl"), "wb") as f:
                _pickle.dump(context_df.to_dict(orient="list"), f, protocol=_pickle.HIGHEST_PROTOCOL)

        # Capture a one-row preview of the context for debug (first row)
        if debug:
            if len(context_df) > 0:
                row0 = context_df.iloc[0].to_dict()
            else:
                row0 = {}
            debug_context_peek.append({
                "context_path": context_path,
                "columns": list(context_df.columns),
                "row": row0,
            })

        t_meta1 = perf_counter()
        if debug:
            try:
                sz = os.path.getsize(context_path)
                print(f"[save] context write time: {(t_meta1 - t_meta0)*1000:.1f} ms, size={_fmt_bytes(sz)}")
            except Exception:
                print(f"[save] context write time: {(t_meta1 - t_meta0)*1000:.1f} ms")

        # compute SHA of the just-written context file so that semantic meaning is encoded. 
        try:
            def _sha256_local(p, chunk=1024*1024):
                import hashlib
                h = hashlib.sha256()
                with open(p, "rb") as f:
                    for b in iter(lambda: f.read(chunk), b""):
                        h.update(b)
                return h.hexdigest()
            context_sha256 = _sha256_local(context_path)
        except Exception:
            context_sha256 = ""

        # ---------- write context schema sidecar (.schema.json) ----------
        schema_name = os.path.splitext(context_name)[0] + ".schema.json"
        schema_path = os.path.join(storm_dir, schema_name)

        # Build a sidecar that is ALWAYS aligned to the actual written context_df columns.
        # Use YAML definitions when available; infer for unknown columns.
        def _infer_dtype_desc(series: pd.Series) -> tuple[str, str]:
            dt = str(series.dtype)
            if "datetime" in dt:
                return "datetime64", "Datetime-like column (inferred)"
            if "int" in dt:
                return "int", "Integer column (inferred)"
            if "float" in dt:
                return "float", "Float column (inferred)"
            if "bool" in dt:
                return "bool", "Boolean column (inferred)"
            return "string", "String-like column (inferred)"

        yaml_cols = (context_schema_base.get("columns") or {})
        sidecar_cols = {}
        for col in context_df.columns:
            if col in yaml_cols:
                sidecar_cols[col] = dict(yaml_cols[col])
            else:
                dtype_guess, desc_guess = _infer_dtype_desc(context_df[col])
                sidecar_cols[col] = {
                    "required": False,
                    "dtype": dtype_guess,
                    "unit": "",
                    "desc": desc_guess,
                }

        schema_sidecar = dict(context_schema_base)
        schema_sidecar.update({
            "applies_to": context_name,
            "applies_to_sha256": context_sha256,
            "generated_at_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "columns": sidecar_cols,
        })

        try:
            with open(schema_path, "w", encoding="utf-8") as f:
                json.dump(schema_sidecar, f, indent=2, sort_keys=True)
            if debug:
                print(f"[save] wrote context schema sidecar → {schema_path}")
        except Exception as e:
            if debug:
                print(f"[save][warn] failed to write schema sidecar: {e}")


        # precompute relpaths for product HDF attrs
        # (relative from product file directory to context/schema files)
        def _rel(p):
            try:
                return os.path.relpath(p, start=storm_dir)
            except Exception:
                return os.path.basename(p)

        context_relpath = _rel(context_path)
        context_schema_relpath = _rel(schema_path)

        # ---------- per-product tensor HDF ----------
        for prefix in prefixes:
            t_prod0 = perf_counter()
            scan_col = f"{prefix}_scan"

            # first pass: determine (H, W, C_max) by scanning available frames across all sweeps
            t_dim0 = perf_counter()
            H = 0
            W = 0
            C = 0
            chosen_field = None
            any_present = False
            min_nsweeps, max_nsweeps = 10**9, 0

            for _, row in sub.iterrows():
                scan = row.get(scan_col, None)
                if scan is None:
                    continue
                any_present = True
                if chosen_field is None:
                    chosen_field = _choose_field_key(scan, prefix)

                arr_list = _frames_from_scan_all_sweeps(scan, chosen_field)
                if not arr_list:
                    continue
                # update dims over all sweeps in this scan
                for a in arr_list:
                    H = max(H, int(a.shape[0]))
                    W = max(W, int(a.shape[1]))
                C = max(C, len(arr_list))
                min_nsweeps = min(min_nsweeps, len(arr_list))
                max_nsweeps = max(max_nsweeps, len(arr_list))

            t_dim1 = perf_counter()

            if not any_present:
                if debug:
                    print(f"[save] storm ({site},{storm},{yr}) → no data for '{prefix}' — skipping file.")
                continue

            if H == 0 or W == 0:
                if debug:
                    print(f"[save] storm ({site},{storm},{yr}) → '{prefix}': could not infer shape — skipping.")
                continue

            # build file name with dims/time span (+ channels)
            prod_name = f"{site}_{storm}_{prefix}_T{T:03d}_{H}x{W}x{C}ch_{t0}_{t1}.h5"
            prod_path = os.path.join(storm_dir, prod_name)

            if debug:
                print(f"[save] product '{prefix}': chosen_field='{chosen_field}', H={H}, W={W}, C={C} "
                      f"(nsweeps per time ~ min={min_nsweeps if min_nsweeps<10**9 else 'NA'}, max={max_nsweeps}) "
                      f"[dimension pass {(t_dim1 - t_dim0)*1000:.1f} ms]")
                print(f"[save] writing → {prod_path}")

            import h5py
            # ---- discover one representative scan (for attrs like units/site coords) ----
            first_scan = None
            for _, _row in sub.iterrows():
                _sc = _row.get(scan_col, None)
                if _sc is not None:
                    first_scan = _sc
                    break

            # Pull units (fallback dBZ) and site coords if we can
            _units = "dBZ"
            _site_lat = _site_lon = _site_alt = np.nan
            _is_pseudo_any = False
            if first_scan is not None:
                try:
                    if chosen_field and chosen_field in first_scan.fields:
                        _units = first_scan.fields[chosen_field].get("units", _units)
                except Exception:
                    pass
                try:
                    _site_lat = float(first_scan.latitude["data"][0])
                    _site_lon = float(first_scan.longitude["data"][0])
                    _site_alt = float(first_scan.altitude["data"][0])
                except Exception:
                    pass
                try:
                    _is_pseudo_any = bool(_is_pseudo(first_scan))
                except Exception:
                    _is_pseudo_any = False

            # ---- open file & create datasets ----
            t_h5open0 = perf_counter()
            with h5py.File(prod_path, "w") as h5:
                # ---- /data dataset ----
                dset = h5.create_dataset(
                    "data",
                    shape=(T, H, W, C),
                    dtype="float32",
                    chunks=(1, H, W, C),
                    compression="gzip",
                    compression_opts=4,
                    shuffle=True,
                    fillvalue=np.nan,
                )

                # Copy *all* YAML-declared attrs for /data (includes unit_source, description, missing_value, etc.)
                _apply_dataset_attrs(dset, "data")

                # Now override/normalize the dynamic bits
                dset.attrs["units"] = str(_units)  # dynamic from Py-ART

                # Normalize missing_value if YAML used "NaN"
                mv = dset.attrs.get("missing_value", "NaN")
                if isinstance(mv, (str, bytes)) and str(mv).lower() == "nan":
                    dset.attrs["missing_value"] = np.nan

                # file-level attrs: dataset + schema identity 
                h5.attrs["dataset_version"] = str(dataset_version)
                h5.attrs["schema_name"] = str(product_schema_base.get("schema_name", "storm250_product"))
                h5.attrs["schema_version"] = str(product_schema_base.get("schema_version", "1.0.0"))

                # apply static defaults from schema (license/source/grid_type/etc.) 
                for k, v in (product_schema_base.get("h5_file_attrs") or {}).items():
                    h5.attrs[k] = v

                # dynamic provenance / identity attrs 
                h5.attrs["radar_site"] = str(site)
                h5.attrs["storm_id"] = str(storm)
                h5.attrs["year"] = int(yr)
                h5.attrs["product_prefix"] = str(prefix)
                h5.attrs["shape_T_H_W_C"] = (int(T), int(H), int(W), int(C))
                if chosen_field:
                    h5.attrs["field_key"] = str(chosen_field)

                h5.attrs["channels_are_sweeps"] = (not _is_pseudo_any)
                h5.attrs["composite_method"] = "MAX_ALL_SWEEPS" if _is_pseudo_any else "NONE"

                h5.attrs["site_lat"] = float(_site_lat)
                h5.attrs["site_lon"] = float(_site_lon)
                h5.attrs["site_alt_m"] = float(_site_alt)

                # link to context + schema sidecar
                h5.attrs["context_relpath"] = context_relpath
                h5.attrs["context_schema_relpath"] = context_schema_relpath
                h5.attrs["context_sha256"] = str(context_sha256 or "")

                # add which specific schema versions we used 
                h5.attrs["context_schema_id"] = os.path.basename(context_schema_yaml_path or "")
                h5.attrs["product_schema_id"] = os.path.basename(product_schema_yaml_path or "")


                # time vectors
                try:
                    times_utf8 = [pd.to_datetime(t).strftime("%Y-%m-%dT%H:%M:%SZ") for t in sub[time_col]]
                    time_ds = h5.create_dataset("time", data=np.asarray(times_utf8, dtype="S20"))
                    _apply_dataset_attrs(time_ds, "time")
                except Exception:
                    pass
                try:
                    t_ms = np.array([int(pd.to_datetime(t).value // 1_000_000) for t in sub[time_col]], dtype="int64")
                    tms_ds = h5.create_dataset("time_unix_ms", data=t_ms)
                    _apply_dataset_attrs(tms_ds, "time_unix_ms")
                except Exception:
                    pass

                # geometry
                az_dset = h5.create_dataset(
                    "azimuth_deg", shape=(T, H), dtype="float32",
                    chunks=(1, H), compression="gzip", compression_opts=4, shuffle=True, fillvalue=np.nan
                )
                _apply_dataset_attrs(az_dset, "azimuth_deg") # now we also add hdf-specific attributes, based on the yaml file 

                rng_dset = h5.create_dataset(
                    "range_m", shape=(T, W), dtype="float32",
                    chunks=(1, W), compression="gzip", compression_opts=4, shuffle=True, fillvalue=np.nan
                )
                _apply_dataset_attrs(rng_dset, "range_m")

                elv_dset = h5.create_dataset(
                    "elevation_deg", shape=(T, C), dtype="float32",
                    chunks=(1, C), compression="gzip", compression_opts=4, shuffle=True, fillvalue=np.nan
                )
                _apply_dataset_attrs(elv_dset, "elevation_deg")

                host_idx_dset = h5.create_dataset(
                    "azimuth_host_sweep_index", shape=(T,), dtype="int16",
                    chunks=True, compression="gzip", compression_opts=4, shuffle=True
                )
                _apply_dataset_attrs(host_idx_dset, "azimuth_host_sweep_index")

                # per-frame bbox datasets (should be the same though, because its )
                def _get_series(name):
                    try:
                        return np.asarray(context_df[name].to_numpy(), dtype=np.float32)
                    except Exception:
                        if debug:
                            print(f"[save][warn] context missing column '{name}', filling NaNs")
                        return np.full((T,), np.nan, dtype=np.float32)

                bbox_min_lat = _get_series("min_lat")
                bbox_max_lat = _get_series("max_lat")
                bbox_min_lon = _get_series("min_lon")
                bbox_max_lon = _get_series("max_lon")

                bbox_min_lat_dset = h5.create_dataset("bbox_min_lat", data=bbox_min_lat, dtype="float32", chunks=True, compression="gzip", compression_opts=4, shuffle=True)
                _apply_dataset_attrs(bbox_min_lat_dset, "bbox_min_lat")

                bbox_max_lat_dset = h5.create_dataset("bbox_max_lat", data=bbox_max_lat, dtype="float32", chunks=True, compression="gzip", compression_opts=4, shuffle=True)
                _apply_dataset_attrs(bbox_max_lat_dset, "bbox_max_lat")

                bbox_min_lon_dset = h5.create_dataset("bbox_min_lon", data=bbox_min_lon, dtype="float32", chunks=True, compression="gzip", compression_opts=4, shuffle=True)
                _apply_dataset_attrs(bbox_min_lon_dset, "bbox_min_lon")

                bbox_max_lon_dset = h5.create_dataset("bbox_max_lon", data=bbox_max_lon, dtype="float32", chunks=True, compression="gzip", compression_opts=4, shuffle=True)
                _apply_dataset_attrs(bbox_max_lon_dset, "bbox_max_lon")

                # provenance
                try:
                    vlen = h5py.special_dtype(vlen=str)
                    srckey_dset = h5.create_dataset("source_key", shape=(T,), dtype=vlen)
                    _apply_dataset_attrs(srckey_dset, "source_key")
                except Exception:
                    srckey_dset = None

                t_h5open1 = perf_counter()
                if debug:
                    print(f"[save] HDF open+create: {(t_h5open1 - t_h5open0)*1000:.1f} ms "
                          f"(chunk=(1,{H},{W},{C}), gzip=4)")

                # snapshot attrs while file is open (no extra I/O later)
                if debug:
                    file_attrs_snapshot = {k: _pretty_attr(v) for k, v in h5.attrs.items()}
                    data_attrs_snapshot = {k: _pretty_attr(v) for k, v in dset.attrs.items()}
                    debug_product_attrs.append({
                        "path": prod_path,
                        "file_attrs": file_attrs_snapshot,
                        "data_attrs": data_attrs_snapshot,
                    })


                ############################### WRITE FRAMES ###############################
                #       (write the tensor product for each field we keep for a storm)
                #           \- ex. reflectivity, velocity, reflectivity_composite, etc. 
                #
                # each tensor file for one specific field:
                #       - /data with shape T, H, W, C
                #           T    = number of rows (volume across time) for this field
                #           H, W = max rays/gates seen in this field
                #           C    = max number of sweeps for this field 
                #       - field-specific metadata inherited from Py-ART radar object + processing steps
                #           azimuth_deg, 
                #           range_m, elevation_deg,
                #           azimuth_host_sweep_index, 
                #           bbox dimensions,
                #           source_key (points to original nexrad s3 level 2 volume) if possible

                t_write_total0 = perf_counter()
                nan_probe_indices = {0, T-1, T//2}  # sample a few frames for NaN frac probe

                for i, row in sub.iterrows():
                    t_frame0 = perf_counter()
                    scan = row.get(scan_col, None)

                    cube = np.empty((H, W, C), dtype=np.float32)
                    cube[:] = np.nan

                    # Defaults for geometry writes at this time index
                    H_host = 0
                    W_common = 0
                    host_idx = 0
                    host_az = None
                    rng = None
                    lim_sweeps_written = 0

                    if scan is not None:
                        if chosen_field is None:
                            chosen_field = _choose_field_key(scan, prefix)

                        if _is_pseudo(scan):
                            try:
                                fld = scan.fields[chosen_field]["data"]
                                if not isinstance(fld, ma.MaskedArray):
                                    fld = ma.MaskedArray(fld, mask=np.zeros_like(fld, dtype=bool))
                                s0 = int(scan.sweep_start_ray_index["data"][0])
                                e0 = int(scan.sweep_end_ray_index["data"][0])
                                rng = np.asarray(scan.range["data"], dtype=np.float32)
                                W_common = min(W, (rng.shape[0] if rng is not None else W))
                                host_block = fld[s0:e0, :W_common].filled(np.nan).astype(np.float32, copy=False)
                                h_host = min(H, host_block.shape[0])
                                w_host = min(W, host_block.shape[1])
                                cube[:h_host, :w_host, 0] = host_block[:h_host, :w_host]
                                host_idx = 0
                                host_az = np.asarray(scan.azimuth["data"][s0:e0], dtype=np.float32)
                                H_host = min(H, host_az.size)
                                lim_sweeps_written = 1
                            except Exception as e:
                                if debug:
                                    print(f"[save][warn] pseudo write failed at t={i}: {e}")

                            # elevation: single value
                            try:
                                el0 = float(scan.fixed_angle["data"][0])
                            except Exception:
                                el0 = float(np.nanmean(np.asarray(scan.elevation["data"][s0:e0], dtype=float))) if H_host > 0 else np.nan
                            elv_dset[i, 0] = el0

                        else:
                            # --- per-time host geometry for multi-tilt ---
                            try:
                                sidx = scan.sweep_start_ray_index["data"].astype(int)
                                eidx = scan.sweep_end_ray_index["data"].astype(int)
                                nsweeps_t = int(len(sidx))
                            except Exception as e:
                                if debug:
                                    print(f"[save][warn] sweep bounds missing at t={i}: {e}")
                                nsweeps_t = 0
                                sidx = eidx = np.array([], dtype=int)

                            if nsweeps_t > 0:
                                rays_per = [int(max(0, eidx[j] - sidx[j])) for j in range(nsweeps_t)]
                                host_idx = int(np.argmax(rays_per))
                                hs, he = sidx[host_idx], eidx[host_idx]

                                host_az = np.unwrap(np.deg2rad(np.asarray(scan.azimuth["data"][hs:he], dtype=float)))
                                H_host = min(H, host_az.size)

                                try:
                                    rng = np.asarray(scan.range["data"], dtype=np.float32)
                                except Exception:
                                    rng = None
                                W_common = min(W, (rng.shape[0] if rng is not None else W))

                                # Elevations per sweep (for first C channels)
                                try:
                                    fixed_angles = np.asarray(scan.fixed_angle["data"], dtype=np.float32)
                                except Exception:
                                    try:
                                        fixed_angles = np.array([
                                            float(np.nanmean(np.asarray(scan.elevation["data"][sidx[j]:eidx[j]], dtype=float)))
                                            for j in range(nsweeps_t)
                                        ], dtype=np.float32)
                                    except Exception:
                                        fixed_angles = None
                                if fixed_angles is not None and fixed_angles.size > 0:
                                    elv_dset[i, :min(C, fixed_angles.size)] = fixed_angles[:min(C, fixed_angles.size)]

                                lim_sweeps = min(C, nsweeps_t)
                                lim_sweeps_written = lim_sweeps

                                for c in range(lim_sweeps):
                                    s, e = int(sidx[c]), int(eidx[c])
                                    # source azimuths
                                    src_az = np.unwrap(np.deg2rad(np.asarray(scan.azimuth["data"][s:e], dtype=float)))

                                    # 0/2π-aware nearest-neighbor to host rays
                                    two_pi = 2.0 * np.pi
                                    k = int(np.round((host_az.mean() - src_az.mean()) / two_pi))
                                    src_u = src_az + k * two_pi
                                    order_idx = np.argsort(src_u)
                                    src_sorted = np.maximum.accumulate(src_u[order_idx])
                                    idx_sorted = order_idx
                                    src_ext = np.concatenate([src_sorted - two_pi, src_sorted, src_sorted + two_pi])
                                    idx_ext = np.concatenate([idx_sorted,          idx_sorted,          idx_sorted])
                                    j = np.searchsorted(src_ext, host_az[:H_host])
                                    j0 = np.clip(j - 1, 0, src_ext.size - 1)
                                    j1 = np.clip(j,       0, src_ext.size - 1)
                                    pick = np.where(np.abs(host_az[:H_host] - src_ext[j0]) <=
                                                    np.abs(host_az[:H_host] - src_ext[j1]), j0, j1)
                                    src_idx = idx_ext[pick]
                                    src_idx = np.clip(src_idx, 0, max(0, (e - s) - 1))

                                    # slice & map gates to host rays (use common W)
                                    fld = scan.fields[chosen_field]["data"]
                                    if not isinstance(fld, ma.MaskedArray):
                                        fld = ma.MaskedArray(fld, mask=np.zeros_like(fld, dtype=bool))
                                    field = fld[s:e, :W_common]
                                    ch_re = field[src_idx, :]
                                    cube[:H_host, :W_common, c] = ch_re.filled(np.nan).astype(np.float32, copy=False)
                            else:
                                if debug:
                                    print(f"[save][warn] no sweeps at t={i}")

                    # write main tensor
                    dset[i, :, :, :] = cube

                    # write geometry/provenance
                    if host_az is not None and H_host > 0:
                        az_dset[i, :H_host] = np.degrees(host_az[:H_host]).astype(np.float32, copy=False)
                    if rng is not None and W_common > 0:
                        rng_dset[i, :W_common] = rng[:W_common]
                    host_idx_dset[i] = int(host_idx)

                    if srckey_dset is not None:
                        keycol = f"{prefix}_matched_volume_s3_key"
                        try:
                            srckey_dset[i] = str(row.get(keycol, "") or "")
                        except Exception:
                            srckey_dset[i] = ""

                    # ---- per-frame debug summary (cheap) ----
                    if debug and (i in nan_probe_indices or i < 3):
                        view = cube
                        n_tot = view.size
                        n_nan = int(np.isnan(view).sum())
                        frac_nan = n_nan / n_tot if n_tot else 0.0
                        print(f"[save] t={i:04d} "
                              f"H_host={H_host} W_common={W_common} host_idx={host_idx} "
                              f"channels_written={lim_sweeps_written} NaN_frac={frac_nan:.3f} "
                              f"frame_write={(perf_counter()-t_frame0)*1000:.1f} ms")

                t_write_total1 = perf_counter()
                if debug:
                    print(f"[save] wrote {T} frames in {(t_write_total1 - t_write_total0):.3f} s "
                          f"(avg {(t_write_total1 - t_write_total0)*1000/T:.1f} ms/frame)")

            # ---- after closing file, report size ----
            t_prod1 = perf_counter()
            if debug:
                try:
                    fsz = os.path.getsize(prod_path)
                    print(f"[save] closed '{prefix}' file in {(t_prod1 - t_prod0):.3f} s, size={_fmt_bytes(fsz)}")
                except Exception:
                    print(f"[save] closed '{prefix}' file in {(t_prod1 - t_prod0):.3f} s")

            saved.append({"storm_dir": storm_dir, "product": prefix, "path": prod_path})

        # encourage GC per storm
        t_gc0 = perf_counter()
        gc.collect()
        t_gc1 = perf_counter()
        if debug:
            print(f"[save] GC after group: {(t_gc1 - t_gc0)*1000:.1f} ms")
            print(f"[save] ===== End Group: site={site} storm={storm} ===== "
                  f"(elapsed {(perf_counter() - t_group0):.3f} s)")

    # ---------- optionally drop in-memory scans ----------
    if drop_scans_after_save:
        t_drop0 = perf_counter()
        if debug:
            print("[save] dropping in-memory *_scan objects from dataframe.")
        for p in prefixes:
            col = f"{p}_scan"
            if col in df.columns:
                df[col] = None
        gc.collect()
        t_drop1 = perf_counter()
        if debug:
            print(f"[save] drop scans + GC: {(t_drop1 - t_drop0)*1000:.1f} ms")

    # FINAL DEBUG DUMP (no file reads): tensor HDF attrs + one-row context preview 
    if debug:
        base_abs = os.path.abspath(base_dir)
        print(f"[save] completed. Wrote context + products for {len(groups)} storm group(s) into {base_abs}")
        print(f"[save] TOTAL elapsed: {(perf_counter() - t_all0):.3f} s")

        print("\n[save] ===== DEBUG: Tensor HDF Attributes (captured in-memory) =====")
        for snap in debug_product_attrs:
            print(f"[save] --- {snap['path']} ---")
            print("[save] file attrs:")
            for k in sorted(snap["file_attrs"].keys()):
                print(f"[save]   {k} = {snap['file_attrs'][k]!r}")
            print("[save] /data attrs:")
            for k in sorted(snap["data_attrs"].keys()):
                print(f"[save]   {k} = {snap['data_attrs'][k]!r}")

        print("\n[save] ===== DEBUG: Context HDF Columns (first row values; captured in-memory) =====")
        for ctx in debug_context_peek:
            print(f"[save] --- {ctx['context_path']} ---")
            cols = ctx["columns"]
            row  = ctx["row"]
            print(f"[save] columns={len(cols)}")
            for col in cols:
                val = row.get(col, None)
                try:
                    if isinstance(val, (pd.Timestamp, np.datetime64)):
                        sval = pd.to_datetime(val).strftime("%Y-%m-%dT%H:%M:%SZ")
                    elif isinstance(val, (float, np.floating)):
                        sval = f"{float(val):.6g}"
                    else:
                        sval = str(val)
                except Exception:
                    sval = repr(val)
                print(f"[save]   {col}: {sval}")

    return saved