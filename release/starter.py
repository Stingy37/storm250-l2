# HRSS Starter dataset builder (Colab-friendly; no globals)
# Copies full radar sites into <dst_base>/<year>/<SITE>/storm_*/*,
# then rebuilds manifest.csv + catalog.csv per year (lean, no sidecars).

import os
import re
import time
import json
import shutil


from pathlib import Path
from typing import Iterable, Optional



# -------------------------- helpers --------------------------
def _fmt_bytes(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB", "PB"]:
        if abs(n) < 1024.0:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} EB"

_fname_re = re.compile(
    r"^(?P<site>[A-Z0-9]{4})_(?P<storm>\d+?)_"
    r"(?P<kind>context|[a-zA-Z0-9_]+?)"
    r"(?:_T(?P<T>\d+))?"
    r"(?:_(?P<H>\d+)x(?P<W>\d+)x(?P<C>\d+)ch)?"
    r"_(?P<t0>\d{8}T\d{6}Z)_(?P<t1>\d{8}T\d{6}Z)\.h5$"
)

def _list_sites(year_dir: Path) -> list[str]:
    sites = []
    if not year_dir.exists():
        return sites
    for p in year_dir.iterdir():
        if p.is_dir() and len(p.name) == 4:  # e.g., KDLH
            sites.append(p.name)
    return sorted(sites)

def _copy_site_tree(src_site_dir: Path, dst_site_dir: Path, skip_names: set, skip_suffixes: set, debug: bool = True) -> tuple[int, int]:
    """
    Copy everything under <SITE>/storm_*/*, skipping manifests/sidecars.
    Returns (n_files, total_bytes).
    """
    n_files = 0
    total_bytes = 0
    if not src_site_dir.exists():
        if debug:
            print(f"[starter][warn] missing site: {src_site_dir}")
        return (0, 0)

    dst_site_dir.mkdir(parents=True, exist_ok=True)

    for storm_dir in sorted(src_site_dir.glob("storm_*")):
        if not storm_dir.is_dir():
            continue
        dst_storm_dir = dst_site_dir / storm_dir.name
        dst_storm_dir.mkdir(parents=True, exist_ok=True)

        for fn in sorted(os.listdir(storm_dir)):
            if fn in skip_names:
                if debug:
                    print(f"[starter][skip] {storm_dir/fn} (manifest/catalog)")
                continue
            if any(fn.endswith(suf) for suf in skip_suffixes):
                if debug:
                    print(f"[starter][skip] {storm_dir/fn} (legacy sidecar)")
                continue

            src = storm_dir / fn
            if not src.is_file():
                continue
            dst = dst_storm_dir / fn

            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            n_files += 1
            try:
                total_bytes += os.path.getsize(dst)
            except Exception:
                pass
            if debug:
                print(f"[starter][copy] {src} → {dst}")

    return (n_files, total_bytes)

# -------------------------- main builder --------------------------
def build_dataset_starter(
    src_base: str = "/content/drive/MyDrive/Datasets/training_datasets/level_two",
    dst_base: str = "/content/drive/MyDrive/Datasets/training_datasets/starter",
    years: Iterable[str] = ("2017",),
    sites: Optional[Iterable[str]] = ("KDLH", "KUEX"),  # None → copy ALL sites found for each year
    clear_dest_site_first: bool = True,
    debug: bool = True,
    skip_filenames: Iterable[str] = ("manifest.csv", "catalog.csv"),
    skip_suffixes: Iterable[str] = (".sha256",),
):
    """
    Build a 'starter' subset by copying whole radar sites from src_base into dst_base,
    preserving <year>/<SITE>/storm_*/ layout and then regenerating per-year manifest/catalog.

    Parameters
    ----------
    src_base : str
        Root of the full dataset (e.g., ".../Datasets/training_datasets/level_two").
    dst_base : str
        Root to write the starter copy into (e.g., ".../Datasets/training_datasets/starter").
    years : Iterable[str]
        Years to include (strings).
    sites : Optional[Iterable[str]]
        Specific site IDs to include; None → include all sites present for each year.
    clear_dest_site_first : bool
        If True, remove existing destination site directories before copying.
    debug : bool
        Verbose logging.
    skip_filenames : Iterable[str]
        Exact filenames to skip (e.g., manifest.csv, catalog.csv).
    skip_suffixes : Iterable[str]
        Filename suffixes to skip (e.g., ".sha256").
    """
    t0 = time.perf_counter()
    src_base = Path(src_base)
    dst_base = Path(dst_base)
    year_summaries = []

    skip_names_set = set(skip_filenames)
    skip_suffixes_set = set(skip_suffixes)

    if debug:
        print(f"[starter] src_base={src_base}")
        print(f"[starter] dst_base={dst_base}")
        print(f"[starter] years={list(years)}")
        print(f"[starter] sites={'ALL' if sites is None else list(sites)}")
        print(f"[starter] clear_dest_site_first={clear_dest_site_first}")

    for year in years:
        yr = str(year)
        src_year_dir = src_base / yr
        dst_year_dir = dst_base / yr

        if not src_year_dir.exists():
            print(f"[starter][warn] missing year: {src_year_dir} — skipping")
            continue

        site_list = list(sites) if sites is not None else _list_sites(src_year_dir)
        if debug:
            print(f"[starter] Year {yr}: sites to copy = {site_list}")

        total_files = 0
        total_bytes = 0

        for site in site_list:
            src_site_dir = src_year_dir / site
            if not src_site_dir.exists():
                if debug:
                    print(f"[starter][warn] site not found in {yr}: {site} — skipping")
                continue

            dst_site_dir = dst_year_dir / site

            if clear_dest_site_first and dst_site_dir.exists():
                if debug:
                    print(f"[starter] clearing destination site dir → {dst_site_dir}")
                shutil.rmtree(dst_site_dir, ignore_errors=True)

            if debug:
                print(f"[starter] copying site {site} for year {yr} …")
            n_files, bytes_copied = _copy_site_tree(
                src_site_dir, dst_site_dir, skip_names_set, skip_suffixes_set, debug=debug
            )
            total_files += n_files
            total_bytes += bytes_copied

            if debug:
                print(f"[starter] site {site}: copied {n_files} files, {_fmt_bytes(bytes_copied)}")

        # Rebuild manifest/catalog for this year directory (overwrites)
        try:
            year_dir_str = str(dst_year_dir)
            if debug:
                print(f"[starter] rebuilding manifest/catalog in → {year_dir_str}")
            build_year_manifest_and_catalog(
                year_dir_str,
                update_schema_checksums=True,  # backfill applies_to_sha256 in context schemas if missing
                debug=debug
            )
        except NameError:
            raise RuntimeError(
                "build_year_manifest_and_catalog(...) is not defined in this notebook. "
                "Paste your function definition cell above and rerun."
            )

        year_summaries.append((yr, total_files, total_bytes))

    if debug:
        print("\n[starter] ===== SUMMARY =====")
        for yr, nf, nb in year_summaries:
            print(f"[starter] {yr}: files={nf}, bytes={_fmt_bytes(nb)}")
        print(f"[starter] OUTPUT ROOT: {dst_base}")
        print(f"[starter] TOTAL elapsed: {(time.perf_counter()-t0):.2f} s")


# -------------------------- run it (edit args if desired) --------------------------
build_dataset_starter(
    src_base="Datasets/training_datasets/level_two",       # where full dataset currently lives
    dst_base="Datasets/training_datasets/starter",         # where the starter subset should be written
    years=("2017",),                                       # which years to include
    sites=("KDLH", "KUEX"),                                # which radar sites to include;
                                                           # set to None to copy ALL sites found for each year

    clear_dest_site_first=True,                            # wipe existing starter before copying (should always be true)
    debug=True,
    skip_filenames=("manifest.csv", "catalog.csv"),        # filenames to skip (legacy or unwanted artifacts)
    skip_suffixes=(".sha256",),
)

