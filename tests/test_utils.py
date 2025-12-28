import math
import numpy as np
import pytest

from storm250.utils import (
    _compute_weighted_dimensions,
    _ensure_dir,
    _fmt_bytes,
    aggressive_memory_cleanup,
    haversine,
    trimmed_cluster_center,
    mad_scale,
    section_seperator,
)


def test_haversine_zero_distance():
    d = haversine(-97.0, 35.0, -97.0, 35.0)
    assert d == pytest.approx(0.0, abs=1e-9)


def test_haversine_known_distance_equator_one_degree_lon():
    # At the equator, 1 degree of longitude is about 111.319 km
    d = haversine(0.0, 0.0, 1.0, 0.0)
    assert d == pytest.approx(111.319, rel=5e-3)


def test_section_seperator_noop_without_logger():
    # Should not raise, and should not print (we don't assert prints here)
    section_seperator(3)


def test_ensure_dir_creates_directory(tmp_path):
    p = tmp_path / "a" / "b" / "c"
    assert not p.exists()
    _ensure_dir(p)
    assert p.exists()
    assert p.is_dir()


@pytest.mark.parametrize(
    "n,expected",
    [
        (0, "0 B"),
        (1, "1 B"),
        (1023, "1023 B"),
        (1024, "1.00 KB"),
        (1536, "1.50 KB"),
        (1024**2, "1.00 MB"),
        (1024**3, "1.00 GB"),
    ],
)
def test_fmt_bytes(n, expected):
    assert _fmt_bytes(n) == expected


def test_center_basic_cluster():
    vals = np.array([0, 0, 0, 1, 1, 10, 10, 10, 10], dtype=float)
    # should land near the dominant cluster around 0 / 1 (5 values of either 0/1 vs 4 of 10)
    #   \- cluster -> by count, not by magnitude (we don't want to weight larger width / heights more)
    m = trimmed_cluster_center(vals)
    assert math.isfinite(m)
    assert m < 5.0  # should be closer to 0/1 than to 10


def test_center_all_nan():
    vals = np.array([np.nan, np.nan], dtype=float)
    m = trimmed_cluster_center(vals)
    assert math.isnan(m)


def test_mad_scale_nonzero_floor():
    vals = np.array([5.0, 5.0, 5.0, 5.0])
    # MAD is 0 here, but function floors at 1.0
    assert mad_scale(vals) == pytest.approx(1.0)


def test_mad_scale_handles_nan():
    vals = np.array([1.0, 2.0, np.nan, 4.0])
    s = mad_scale(vals)
    assert s >= 1.0


def test_compute_weighted_dimensions_returns_correct_shapes():
    widths = np.array([1000, 1000, 1200, 1000, 1000], dtype=float)
    heights = np.array([800, 800, 900, 800, 800], dtype=float)

    avg_w, avg_h, weights = _compute_weighted_dimensions(widths, heights, debug=False)

    assert isinstance(avg_w, float)
    assert isinstance(avg_h, float)
    assert isinstance(weights, np.ndarray)
    assert weights.shape == widths.shape
    assert np.isfinite(avg_w)
    assert np.isfinite(avg_h)
    assert np.isclose(weights.sum(), 1.0, atol=1e-9)


def test_compute_weighted_dimensions_biases_toward_center():
    # One outlier; average should be closer to the center (1000/800) than naive mean
    widths = np.array([1000, 1000, 1000, 1000, 5000], dtype=float)
    heights = np.array([800, 800, 800, 800, 4000], dtype=float)

    naive_w = float(np.mean(widths))
    naive_h = float(np.mean(heights))

    avg_w, avg_h, _ = _compute_weighted_dimensions(widths, heights, debug=False)

    # Should be closer to the typical values than the naive mean is
    assert abs(avg_w - 1000.0) < abs(naive_w - 1000.0)
    assert abs(avg_h - 800.0) < abs(naive_h - 800.0)


def test_aggressive_memory_cleanup_does_not_crash():
    # This should be safe and should not depend on external libs being installed.
    locals_dict = {
        "lsr_df": object(),
        "spc_df": object(),
        "some_big_thing": np.zeros((10, 10)),
    }
    aggressive_memory_cleanup(locals_dict)
    
    # Known names should be nulled out if present
    assert locals_dict.get("lsr_df") is None
    assert locals_dict.get("spc_df") is None
