def _should_skip(name: str,
                 exclude_names: set[str],
                 exclude_suffixes: set[str]) -> bool:
    base = os.path.basename(name)
    if base in exclude_names:
        return True
    if any(base.endswith(s) for s in exclude_suffixes):
        return True
    return False



def _gather_files(root: Path,
                  exclude_names: set[str],
                  exclude_suffixes: set[str],
                  debug: bool = True) -> list[Path]:
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        # silently skip Jupyter checkpoint dirs without exposing a param
        if ".ipynb_checkpoints" in Path(dirpath).parts:
            if debug:
                print(f"[zen][skipdir] {dirpath} (.ipynb_checkpoints)")
            continue
        for fn in filenames:
            if _should_skip(fn, exclude_names, exclude_suffixes):
                if debug:
                    print(f"[zen][skip] {Path(dirpath)/fn}")
                continue
            if fn == ".DS_Store":
                if debug:
                    print(f"[zen][skip] {Path(dirpath)/fn} (.DS_Store)")
                continue
            p = Path(dirpath)/fn
            if p.is_file():
                files.append(p)
    files.sort()
    return files

def _is_subpath(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False

# -------------------------- main packer --------------------------
def build_zenodo_bundle(
    src_dir: str,
    out_dir: str,
    bundle_name: str,
    zip_mode: str = "zip",                    # "zip" or "tar.gz"
    arc_prefix: Optional[str] = None,         # folder prefix inside archive; default = bundle_name
    exclude_names: Iterable[str] = ("checksums.sha256",),  # don't re-embed old checksum files
    exclude_suffixes: Iterable[str] = (".sha256",),
    clear_out_dir_first: bool = True,         # nuke out_dir before building
    zip_store: bool = True,                   # NEW: True = no compression (fastest for .h5)
    zip_deflate_level: int = 6,               # used only when zip_store=False
    debug: bool = True,
):
    """
    Package a HRSS dataset subtree for Zenodo.

    zip_store:
        If True (default), use ZIP_STORED (no compression) — fastest and ideal for already-compressed HDF5.
        If False, use ZIP_DEFLATED with 'zip_deflate_level' (1=fastest, 9=smallest).
    """
    t0 = time.perf_counter()
    src_dir = Path(src_dir)
    out_dir = Path(out_dir)

    if not src_dir.exists():
        raise FileNotFoundError(f"src_dir does not exist: {src_dir}")

    # Safety: out_dir must NOT be inside src_dir (or equal)
    if _is_subpath(out_dir, src_dir) or src_dir.resolve() == out_dir.resolve():
        raise RuntimeError(
            f"out_dir must not be inside src_dir.\n  src_dir={src_dir}\n  out_dir={out_dir}\n"
            "Pick an out_dir that is outside the source tree."
        )

    if clear_out_dir_first and out_dir.exists():
        if debug:
            print(f"[zen] clearing out_dir → {out_dir}")
        shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    if arc_prefix is None:
        arc_prefix = bundle_name

    if debug:
        print(f"[zen] src_dir={src_dir}")
        print(f"[zen] out_dir={out_dir}")
        print(f"[zen] bundle_name={bundle_name}")
        print(f"[zen] zip_mode={zip_mode}")
        print(f"[zen] arc_prefix={arc_prefix}")
        print(f"[zen] exclude_names={list(exclude_names)}")
        print(f"[zen] exclude_suffixes={list(exclude_suffixes)}")
        print(f"[zen] clear_out_dir_first={clear_out_dir_first}")
        if zip_mode == "zip":
            print(f"[zen] zip_store={zip_store} (deflate_level={zip_deflate_level if not zip_store else 'N/A'})")

    exclude_names = set(exclude_names)
    exclude_suffixes = set(exclude_suffixes)

    # 1) Gather files
    files = _gather_files(src_dir, exclude_names, exclude_suffixes, debug=debug)
    if debug:
        print(f"[zen] files to archive: {len(files)}")
        for p in files[:8]:
            print(f"[zen]   + {p.relative_to(src_dir)}")
        if len(files) > 8:
            print(f"[zen]   ... (+{len(files)-8} more)")

    # 2) Create the archive
    if zip_mode not in {"zip","tar.gz"}:
        raise ValueError("zip_mode must be 'zip' or 'tar.gz'")

    if zip_mode == "zip":
        archive_path = out_dir / f"{bundle_name}.zip"
        if debug:
            print(f"[zen] creating ZIP → {archive_path}")
        compression = zipfile.ZIP_STORED if zip_store else zipfile.ZIP_DEFLATED
        zkwargs = {"compression": compression}
        if not zip_store:
            # compresslevel is honored only for ZIP_DEFLATED
            zkwargs["compresslevel"] = int(zip_deflate_level)

        with zipfile.ZipFile(archive_path, "w", **zkwargs) as zf:
            total_bytes = 0
            for fp in files:
                rel = fp.relative_to(src_dir)
                arcname = str(Path(arc_prefix) / rel)
                zf.write(fp, arcname)  # store/deflate decided by ZipFile(...)
                try:
                    total_bytes += fp.stat().st_size
                except Exception:
                    pass
        if debug:
            mode_str = "STORED(no-compress)" if zip_store else f"DEFLATED(level={zip_deflate_level})"
            print(f"[zen] ZIP mode={mode_str}; contents ≈ {_fmt_bytes(total_bytes)}; file size = {_fmt_bytes(archive_path.stat().st_size)}")

    else:
        # tar.gz path (still single-threaded in Python). If you need even faster,
        # consider: create 'w' (no-compress) tar then compress with 'pigz' externally.
        archive_path = out_dir / f"{bundle_name}.tar.gz"
        if debug:
            print(f"[zen] creating TAR.GZ → {archive_path}")
        with tarfile.open(archive_path, "w:gz") as tf:
            total_bytes = 0
            for fp in files:
                rel = fp.relative_to(src_dir)
                arcname = str(Path(arc_prefix) / rel)
                tf.add(fp, arcname=arcname)
                try:
                    total_bytes += fp.stat().st_size
                except Exception:
                    pass
        if debug:
            print(f"[zen] TAR.GZ contents ≈ {_fmt_bytes(total_bytes)}; file size = {_fmt_bytes(archive_path.stat().st_size)}")

    # 3) Write top-level checksums.sha256 for the ARCHIVE
    checksums_path = out_dir / "checksums.sha256"
    arch_hash = _sha256(str(archive_path))
    with open(checksums_path, "w", encoding="utf-8") as f:
        f.write(f"{arch_hash}  {archive_path.name}\n")

    if debug:
        print(f"[zen] wrote checksums.sha256 → {checksums_path}")
        print(f"[zen]   {arch_hash}  {archive_path.name}")
        print(f"[zen] DONE in {(time.perf_counter()-t0):.2f} s")
        print("\n[zen] Upload the following to Zenodo:")
        print(f"[zen]   - {archive_path}")
        print(f"[zen]   - {checksums_path}")
        print("[zen] After downloading, users can verify quickly with:")
        print(f"[zen]   sha256sum -c {checksums_path.name}  # (Linux)  OR")
        print(f"[zen]   shasum -a 256 -c {checksums_path.name}  # (macOS)")

# -------------------------- run it (edit args) --------------------------
build_zenodo_bundle(
    src_dir="Datasets/training_datasets/starter/2017",
    out_dir="Datasets/training_datasets/starter/zenodo_bundle",
    bundle_name="hrss-starter-2017",
    zip_mode="zip",                 # "zip" or "tar.gz"
    arc_prefix="hrss-starter-2017",
    exclude_names=("checksums.sha256",),
    exclude_suffixes=(".sha256",),
    clear_out_dir_first=True,
    zip_store=True,                 # <- FAST: no recompression of .h5
    zip_deflate_level=3,            # ignored when zip_store=True
    debug=True,
)

