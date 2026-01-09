import csv
import hashlib
import json
from pathlib import Path

import pytest

from storm250.provenance import build_year_manifest_and_catalog


################################################## HELPERS ##################################################


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _read_csv_dicts(path: str | Path) -> list[dict]:
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _mkfile(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _mkjson(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def _make_storm_tree(tmp_path: Path) -> tuple[Path, dict]:
    """
    Create a "good enough" year_dir structure that matches how we actually lay things out:
      year_dir/
        KHGX/
          storm_1/
            <SITE>_<STORM>_context_T..._<t0>_<t1>.h5
            <same>.schema.json
            <SITE>_<STORM>_reflectivity_T..._<H>x<W>x<C>ch_<t0>_<t1>.h5
            <SITE>_<STORM>_velocity_T..._<H>x<W>x<C>ch_<t0>_<t1>.h5
            notes.txt
            stray.sha256   (should be skipped)
        manifest.csv      (should be skipped as input + overwritten)
        catalog.csv       (should be skipped as input + overwritten)
    """
    year_dir = tmp_path / "2022"
    site = "KHGX"
    storm_id = 1
    t0 = "20220101T000000Z"
    t1 = "20220101T000100Z"
    T = 2

    storm_dir = year_dir / site / f"storm_{storm_id}"

    context_fn = f"{site}_{storm_id}_context_T{T}_{t0}_{t1}.h5"
    refl_fn = f"{site}_{storm_id}_reflectivity_T{T}_64x64x1ch_{t0}_{t1}.h5"
    vel_fn = f"{site}_{storm_id}_velocity_T{T}_64x64x1ch_{t0}_{t1}.h5"

    # content bytes (don’t need to be valid HDF5 for these tests; we only hash + size)
    context_bytes = b"ctx: pretend this is a context h5\n"
    refl_bytes = b"refl: pretend this is a product h5\n"
    vel_bytes = b"vel: pretend this is a product h5\n"
    notes_bytes = b"hello\n"

    _mkfile(storm_dir / context_fn, context_bytes)
    _mkfile(storm_dir / refl_fn, refl_bytes)
    _mkfile(storm_dir / vel_fn, vel_bytes)
    _mkfile(storm_dir / "notes.txt", notes_bytes)

    # should be ignored by the scan loop
    _mkfile(storm_dir / "stray.sha256", b"deadbeef\n")

    # placeholders that should be skipped during scan; will be overwritten by builder
    _mkfile(year_dir / "manifest.csv", b"old\n")
    _mkfile(year_dir / "catalog.csv", b"old\n")

    return year_dir, {
        "site": site,
        "storm_id": storm_id,
        "t0": t0,
        "t1": t1,
        "T": T,
        "storm_dir": storm_dir,
        "context_fn": context_fn,
        "refl_fn": refl_fn,
        "vel_fn": vel_fn,
        "context_bytes": context_bytes,
        "refl_bytes": refl_bytes,
        "vel_bytes": vel_bytes,
    }


################################################## TESTS ##################################################


def test_build_year_manifest_and_catalog_uses_schema_sha_for_context(tmp_path: Path):
    year_dir, meta = _make_storm_tree(tmp_path)

    # Put an intentionally "wrong" (but valid) SHA in the schema to ensure the builder prefers it.
    schema_sha = "a" * 64
    schema_path = meta["storm_dir"] / f"{Path(meta['context_fn']).stem}.schema.json"
    _mkjson(
        schema_path,
        {
            "applies_to": meta["context_fn"],
            "applies_to_sha256": schema_sha,
            "schema_version": "test",
        },
    )

    manifest_path, catalog_path = build_year_manifest_and_catalog(
        str(year_dir),
        manifest_name="manifest.csv",
        catalog_name="catalog.csv",
        update_schema_checksums=False,
        debug=False,
    )

    assert Path(manifest_path).exists()
    assert Path(catalog_path).exists()

    manifest = _read_csv_dicts(manifest_path)
    catalog = _read_csv_dicts(catalog_path)

    # --- manifest: sanity + skip rules ---
    relpaths = {r["relpath"] for r in manifest}
    assert "manifest.csv" not in relpaths
    assert "catalog.csv" not in relpaths
    assert any(rp.endswith("stray.sha256") for rp in relpaths) is False

    # Find the context_hdf row and ensure SHA comes from schema (not computed)
    ctx_rows = [r for r in manifest if r["file_type"] == "context_hdf" and r["relpath"].endswith(meta["context_fn"])]
    assert len(ctx_rows) == 1
    assert ctx_rows[0]["sha256"] == schema_sha
    assert ctx_rows[0]["site"] == meta["site"]
    assert int(ctx_rows[0]["storm_id"]) == meta["storm_id"]
    assert ctx_rows[0]["kind"] == "context"

    # Product HDFs should hash the actual bytes
    refl_rows = [r for r in manifest if r["file_type"] == "product_hdf" and r["relpath"].endswith(meta["refl_fn"])]
    vel_rows = [r for r in manifest if r["file_type"] == "product_hdf" and r["relpath"].endswith(meta["vel_fn"])]
    assert len(refl_rows) == 1
    assert len(vel_rows) == 1
    assert refl_rows[0]["sha256"] == _sha256_bytes(meta["refl_bytes"])
    assert vel_rows[0]["sha256"] == _sha256_bytes(meta["vel_bytes"])
    assert refl_rows[0]["kind"] == "reflectivity"
    assert vel_rows[0]["kind"] == "velocity"

    # --- catalog: event aggregation correctness ---
    assert len(catalog) == 1
    ev = catalog[0]
    assert ev["site"] == meta["site"]
    assert int(ev["storm_id"]) == meta["storm_id"]
    assert ev["t0_utc"] == meta["t0"]
    assert ev["t1_utc"] == meta["t1"]
    assert int(ev["T"]) == meta["T"]
    assert int(ev["n_products"]) == 2

    # products are written as ",".join(sorted(...))
    assert ev["products"] == "reflectivity,velocity"

    # dims are written as ";".join(sorted([...]))
    dims = ev["dims"].split(";") if ev["dims"] else []
    assert sorted(dims) == sorted(["reflectivity:64x64x1", "velocity:64x64x1"])


def test_build_year_manifest_and_catalog_backfills_schema_checksum_when_requested(tmp_path: Path):
    year_dir, meta = _make_storm_tree(tmp_path)

    # Schema exists but has no checksum; builder should compute + backfill when update_schema_checksums=True
    schema_path = meta["storm_dir"] / f"{Path(meta['context_fn']).stem}.schema.json"
    _mkjson(schema_path, {"applies_to": "WRONG_NAME.h5"})  # mismatched on purpose

    expected_ctx_sha = _sha256_bytes(meta["context_bytes"])

    manifest_path, catalog_path = build_year_manifest_and_catalog(
        str(year_dir),
        update_schema_checksums=True,
        debug=False,
    )

    # Schema should now be updated with correct applies_to + sha
    js = json.loads(schema_path.read_text(encoding="utf-8"))
    assert js.get("applies_to") == meta["context_fn"]
    assert js.get("applies_to_sha256") == expected_ctx_sha

    # Manifest should also carry the computed SHA (since schema didn't provide one at read time)
    manifest = _read_csv_dicts(manifest_path)
    ctx_rows = [r for r in manifest if r["file_type"] == "context_hdf" and r["relpath"].endswith(meta["context_fn"])]
    assert len(ctx_rows) == 1
    assert ctx_rows[0]["sha256"] == expected_ctx_sha
