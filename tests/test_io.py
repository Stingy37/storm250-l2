import logging
from pathlib import Path

import numpy as np
import numpy.ma as ma

from types import SimpleNamespace
from storm250.io import (
    _load_field_pack,
    _load_gz_pickle,
    _save_field_pack,
    _save_gz_pickle,
    _radar_base_from_skeleton_path,
    _load_radar_from_skeleton_and_field_packs,
)

def test_gz_pickle_roundtrip(tmp_path: Path):
    obj = {
        "a": 123,
        "b": "hello",
        "arr": np.arange(10, dtype=np.int64),
        "nested": {"x": [1, 2, 3]},
    }
    p = tmp_path / "cache" / "obj.pkl.gz"

    _save_gz_pickle(obj, p, debug=True)
    assert p.exists()

    out = _load_gz_pickle(p, debug=True)
    assert out is not None
    assert out["a"] == 123
    assert out["b"] == "hello"
    np.testing.assert_array_equal(out["arr"], obj["arr"])
    assert out["nested"] == obj["nested"]


def test_gz_pickle_corrupt_returns_none(tmp_path: Path, caplog):
    p = tmp_path / "cache" / "bad.pkl.gz"
    p.parent.mkdir(parents=True, exist_ok=True)

    # write garbage
    p.write_bytes(b"not a gz file")

    caplog.set_level(logging.ERROR)
    out = _load_gz_pickle(p, debug=True)
    assert out is None
    assert any("skeleton read failed" in rec.message for rec in caplog.records)


def test_gz_pickle_corrupt_quiet_when_debug_false(tmp_path: Path, caplog):
    p = tmp_path / "cache" / "bad.pkl.gz"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"not a gz file")

    caplog.set_level(logging.ERROR)
    out = _load_gz_pickle(p, debug=False)
    assert out is None
    # Should not log if debug=False
    assert not any("skeleton read failed" in rec.message for rec in caplog.records)


def test_field_pack_roundtrip_downcast_true(tmp_path: Path):
    bp = tmp_path / "fields" / "reflectivity"

    data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    mask = np.array([[0, 1], [0, 0]], dtype=bool)
    marr = ma.MaskedArray(data, mask=mask)

    field = {
        "data": marr,
        "units": "dBZ",
        "long_name": "equivalent_reflectivity_factor",
        "comment": "test",
    }

    _save_field_pack(bp, field, downcast=True, debug=True)

    assert Path(str(bp) + ".npz").exists()
    assert Path(str(bp) + ".json").exists()

    out = _load_field_pack(bp, debug=True)
    assert out is not None
    assert out["units"] == "dBZ"
    assert out["long_name"] == "equivalent_reflectivity_factor"
    assert out["comment"] == "test"

    out_data = out["data"]
    assert isinstance(out_data, ma.MaskedArray)
    assert out_data.dtype == np.float32  # downcast=True
    np.testing.assert_allclose(out_data.data, data.astype(np.float32))
    assert np.array_equal(out_data.mask, mask)


def test_field_pack_roundtrip_downcast_false_preserves_dtype(tmp_path: Path):
    bp = tmp_path / "fields" / "velocity"

    data = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float64)
    mask = np.array([[0, 0], [1, 0]], dtype=bool)
    marr = ma.MaskedArray(data, mask=mask)

    field = {
        "data": marr,
        "units": "m/s",
        "long_name": "radial_velocity_of_scatterers_away_from_instrument",
    }

    _save_field_pack(bp, field, downcast=False, debug=True)
    out = _load_field_pack(bp, debug=True)

    assert out is not None
    out_data = out["data"]
    assert out_data.dtype == np.float64  # preserved
    np.testing.assert_allclose(out_data.data, data)
    assert np.array_equal(out_data.mask, mask)


def test_field_pack_missing_returns_none(tmp_path: Path, caplog):
    bp = tmp_path / "fields" / "missing_field"
    caplog.set_level(logging.ERROR)

    out = _load_field_pack(bp, debug=True)
    assert out is None
    assert any("field load failed" in rec.message for rec in caplog.records)

def test_radar_base_from_skeleton_path_ok_and_raises(tmp_path: Path):
    # happy path
    sk = tmp_path / "cache" / "KHGX20220322_120125_V06.skeleton.pkl.gz"
    sk.parent.mkdir(parents=True, exist_ok=True)
    sk.write_bytes(b"")  # doesn't need to be a valid gz for this helper

    base = _radar_base_from_skeleton_path(sk)
    assert base == str(tmp_path / "cache" / "KHGX20220322_120125_V06")

    # wrong suffix -> ValueError
    bad = tmp_path / "cache" / "KHGX20220322_120125_V06.pkl.gz"
    try:
        _radar_base_from_skeleton_path(bad)
        assert False, "expected ValueError for non-skeleton path"
    except ValueError:
        pass

def test_load_radar_from_skeleton_and_field_packs_populates_fields(tmp_path: Path):
    # Create a skeleton cache file (gz pickle) with a picklable radar-like object
    sk = tmp_path / "vol" / "KHGX20220322_120125_V06.skeleton.pkl.gz"
    _save_gz_pickle(SimpleNamespace(fields={}), sk, debug=True)
    assert sk.exists()

    # Build two field packs at base + ".<key>"
    base = _radar_base_from_skeleton_path(sk)

    data_r = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    mask_r = np.array([[0, 1], [0, 0]], dtype=bool)
    field_ref = {
        "data": ma.MaskedArray(data_r, mask=mask_r),
        "units": "dBZ",
        "long_name": "equivalent_reflectivity_factor",
    }
    _save_field_pack(base + ".reflectivity", field_ref, downcast=True, debug=True)

    data_v = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float64)
    mask_v = np.array([[0, 0], [1, 0]], dtype=bool)
    field_vel = {
        "data": ma.MaskedArray(data_v, mask=mask_v),
        "units": "m/s",
        "long_name": "radial_velocity_of_scatterers_away_from_instrument",
    }
    _save_field_pack(base + ".velocity", field_vel, downcast=False, debug=True)

    # Rebuild: request 2 real fields + 1 missing field
    radar = _load_radar_from_skeleton_and_field_packs(
        sk,
        field_keys=["reflectivity", "velocity", "does_not_exist"],
        debug=True,
    )
    assert radar is not None
    assert hasattr(radar, "fields")
    assert isinstance(radar.fields, dict)

    # Confirm populated fields (missing pack should be skipped)
    assert set(radar.fields.keys()) == {"reflectivity", "velocity"}

    # reflectivity got downcast=True -> float32
    out_ref = radar.fields["reflectivity"]["data"]
    assert isinstance(out_ref, ma.MaskedArray)
    assert out_ref.dtype == np.float32
    np.testing.assert_allclose(out_ref.data, data_r.astype(np.float32))
    assert np.array_equal(out_ref.mask, mask_r)

    # velocity preserved dtype -> float64
    out_vel = radar.fields["velocity"]["data"]
    assert isinstance(out_vel, ma.MaskedArray)
    assert out_vel.dtype == np.float64
    np.testing.assert_allclose(out_vel.data, data_v)
    assert np.array_equal(out_vel.mask, mask_v)
