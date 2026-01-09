"""
manifest.csv: Immutable integrity + provenance anchor for each file. 
catalog.csv:  Event-level (storm) discovery + exploration index.
"""

import os
import csv
import json
import re
import hashlib

from datetime import datetime


def _utc_iso(ts):
    return datetime.utcfromtimestamp(ts).strftime("%Y-%m-%dT%H:%M:%SZ")

def _sha256(path, chunk=1024*1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()

def _read_context_sha_from_schema(schema_path, expected_fname=None):
    """
    Read applies_to_sha256 from a context schema JSON if it matches the file.
    Returns sha (str) or None.
    """
    if not os.path.exists(schema_path):
        return None
    try:
        with open(schema_path, "r", encoding="utf-8") as f:
            js = json.load(f)
        applies = js.get("applies_to", "")
        sha = (js.get("applies_to_sha256", "") or "").strip()
        if expected_fname and applies and applies != expected_fname:
            return None
        if len(sha) == 64 and all(c in "0123456789abcdefABCDEF" for c in sha):
            return sha.lower()
    except Exception:
        return None
    return None

_fname_re = re.compile(
    r"^(?P<site>[A-Z0-9]{4})_(?P<storm>\d+?)_"
    r"(?P<kind>context|[a-zA-Z0-9_]+?)"
    r"(?:_T(?P<T>\d+))?"
    r"(?:_(?P<H>\d+)x(?P<W>\d+)x(?P<C>\d+)ch)?"
    r"_(?P<t0>\d{8}T\d{6}Z)_(?P<t1>\d{8}T\d{6}Z)\.h5$"
)

def _parse_file_bits(fname):
    m = _fname_re.match(fname)
    if not m:
        return {}
    d = m.groupdict()
    for k in ("storm","T","H","W","C"):
        if d.get(k) is not None:
            d[k] = int(d[k])
    return d

def _rel(root, path):
    try:
        return os.path.relpath(path, start=root)
    except Exception:
        return os.path.basename(path)

def _debug_preview_csv(path: str, label: str):
    """Print header + first row, like your context preview style."""
    try:
        with open(path, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            print(f"\n[catalog] ===== DEBUG: {label} (columns + first row) =====")
            print(f"[catalog] --- {path} ---")
            if not header:
                print("[catalog][warn] empty file or missing header")
                return
            print(f"[catalog] columns={len(header)}")
            row = next(reader, None)
            if row is None:
                print("[catalog][warn] no data rows")
                for col in header:
                    print(f"[catalog]   {col}: <NA>")
                return
            for col, val in zip(header, row):
                print(f"[catalog]   {col}: {'' if val is None else str(val)}")
    except Exception as e:
        print(f"[catalog][warn] failed preview for {label}: {e}")

# ------------------- main builder -------------------

def build_year_manifest_and_catalog(year_dir: str,
                                    manifest_name: str = "manifest.csv",
                                    catalog_name: str = "catalog.csv",
                                    update_schema_checksums: bool = False,
                                    debug: bool = True):
    """
    Produce:
      - manifest.csv (minimal integrity view)
      - catalog.csv  (event-level exploration view)

    SHA policy (no sidecars):
      * context_hdf → prefer schema's applies_to_sha256 (if filename matches), else compute.
      * product_hdf/other → compute.

    We skip manifest/catalog themselves and any lingering '*.sha256' files.
    """
    if debug:
        print(f"[catalog] scanning year_dir={year_dir}")

    manifest_rows = []
    events = {}  # (site, storm) -> dict

    # ----------- loop to build manifest.csv + catalog.csv
    for root, dirs, files in os.walk(year_dir):
        for fn in files:
            # Skip outputs  
            if fn == manifest_name or fn == catalog_name or fn.endswith(".sha256"):
                continue

            path = os.path.join(root, fn)
            rel  = _rel(year_dir, path)
            try:
                size = os.path.getsize(path)
                mtime_iso = _utc_iso(os.path.getmtime(path))
            except FileNotFoundError:
                continue

            lower = fn.lower()
            if lower.endswith(".schema.json"):
                ftype = "context_schema"
            elif lower.endswith(".h5"):
                ftype = "context_hdf" if "_context_" in fn else "product_hdf"
            elif lower.endswith(".pkl"):
                ftype = "context_pkl" if "_context_" in fn else "other_pkl"
            else:
                ftype = "other"

            bits = _parse_file_bits(fn) if ftype in ("context_hdf","product_hdf") else {}

            # ---------- build SHA for manifest 
            # context_hdf -> special case SHA, because this encodes the physical meaning of the dataset. 
            #                - For each storm, context SHA serves as "ground truth" that semantics stayed the same.
            #                - If our context hdf changes, then the SHA will change as well. We can compare SHAs to see if anything has changed. 
            #                - (Ideally) created as soon as the context.h5 is created, so that it truly represents semantic meaning.
            #                              \- i.e. no differing SHA because corrupted copying, etc. 
            if ftype == "context_hdf":
                schema_path = os.path.join(root, os.path.splitext(fn)[0] + ".schema.json")
                #     /- if a SHA already exists in .schema.json, set the manifest.csv's 
                #     |  corresponding SHA to that. Ensures that the context SHA only encodes semantics. 
                sha = _read_context_sha_from_schema(schema_path, expected_fname=fn) or _sha256(path)

                # CAREFUL: only use this branch for backfilling missing SHA. 
                # Otherwise, it can overwrite a good SHA (encoding only semantics) with an SHA that can also encode corruption, since we do it late 
                # in the pipeline. 
                if update_schema_checksums and os.path.exists(schema_path):
                    try:
                        with open(schema_path, "r", encoding="utf-8") as f:
                            js = json.load(f)
                    except Exception:
                        js = {}
                    js["applies_to"] = fn
                    js["applies_to_sha256"] = sha
                    try:
                        with open(schema_path, "w", encoding="utf-8") as f:
                            json.dump(js, f, indent=2, sort_keys=True)
                        if debug:
                            print(f"[catalog] updated schema checksum: {_rel(year_dir, schema_path)}")
                    except Exception as e:
                        if debug:
                            print(f"[catalog][warn] failed updating schema for {schema_path}: {e}")
            else:
                # for other files, ALL we need to do is provide a SHA for end users to verify against (not semantic meaning,)
                # so we can just calculate the SHA here. 
                sha = _sha256(path)

            # ---------- manifest row (trimmed) 
            # NOTE: drop T,H,W,C,t0_utc,t1_utc here.
            #         - only keep necessary items for integrity (SHA + info about which storm this file is for). 
            row = {
                "relpath": rel,
                "file_type": ftype,
                "size_bytes": size,
                "sha256": sha,
                "mtime_utc": mtime_iso,
                "site": bits.get("site",""),
                "storm_id": bits.get("storm",""),
                "kind": bits.get("kind",""),
            }
            # each file gets a row
            manifest_rows.append(row)

            # ---------- build event aggregation (catalog) 
            if ftype in ("context_hdf","product_hdf"):
                site = bits.get("site")
                storm = bits.get("storm")
                if site and storm is not None:
                    key = (site, int(storm))
                    ev = events.setdefault(key, {           #     |- goal: user friendly, easy way to "know" what the dataset contains 
                                                            # Questions that each field answers:
                        "site": site,                       #   - what storm is this
                        "storm_id": int(storm),             #   ------/
                        "storm_dir": _rel(year_dir, root),  #   - where is it on disk

                        # Keep these in memory to populate t0/t1/T/products/dims,
                        # but don't write context_relpath/context_sha256/product_files to CSV (not needed for catalog). 
                        "context_relpath": "",              
                        "context_sha256": "",

                        "t0_utc": "",                       # when did the storm start
                        "t1_utc": "",                       # when did the storm end 
                        "T": "",                            # how many frames / timesteps
                        "products": [],                     # what products exist
                        "product_files": [],
                        "product_dims": [],
                        "total_bytes": 0,                   # total size of the storm
                    })
                    ev["total_bytes"] += size

                    if ftype == "context_hdf":
                        ev["context_relpath"] = rel  # kept internal only
                        ev["context_sha256"] = sha   # kept internal only
                        ev["t0_utc"] = bits.get("t0","")
                        ev["t1_utc"] = bits.get("t1","")
                        ev["T"] = bits.get("T","")

                    elif ftype == "product_hdf":
                        prefix = bits.get("kind","") or ""
                        if prefix and prefix not in ev["products"]:
                            ev["products"].append(prefix)
                        ev["product_files"].append(rel)  # kept internal only
                        if all(bits.get(k) for k in ("H","W","C")):
                            ev["product_dims"].append(f"{prefix}:{bits['H']}x{bits['W']}x{bits['C']}")

    # ---------- write manifest.csv ----------
    manifest_path = os.path.join(year_dir, manifest_name)
    manifest_fields = [
        "relpath","file_type","size_bytes","sha256","mtime_utc",
        "site","storm_id","kind"
    ]
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=manifest_fields)
        w.writeheader()
        for r in sorted(manifest_rows, key=lambda x: x["relpath"]):
            w.writerow(r)

    # ---------- write catalog.csv ----------
    catalog_path = os.path.join(year_dir, catalog_name)
    catalog_fields = [
        "site","storm_id","storm_dir",
        "t0_utc","t1_utc","T",
        "n_products","products","dims",
        "total_bytes"
    ]
    with open(catalog_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=catalog_fields)
        w.writeheader()
        for (site, storm), ev in sorted(events.items(), key=lambda k: (k[0][0], k[0][1])):
            w.writerow({
                "site": ev["site"],
                "storm_id": ev["storm_id"],
                "storm_dir": ev["storm_dir"],
                "t0_utc": ev["t0_utc"],
                "t1_utc": ev["t1_utc"],
                "T": ev["T"],
                "n_products": len(ev["products"]),
                "products": ",".join(sorted(ev["products"])),
                "dims": ";".join(sorted(ev["product_dims"])),
                "total_bytes": ev["total_bytes"],
            })

    if debug:
        print(f"[catalog] wrote manifest → {manifest_path}")
        print(f"[catalog] wrote catalog  → {catalog_path}")
        print(f"[catalog] events={len(events)}, files={len(manifest_rows)}")
        _debug_preview_csv(manifest_path, "manifest preview")
        _debug_preview_csv(catalog_path, "catalog preview")

    return manifest_path, catalog_path