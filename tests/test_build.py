"""
Tests for storm250.build module.

These tests act as invariants to ensure future changes to build.py maintain:
- Proper multiprocessing behavior and isolation
- Correct pipeline execution flow
- Memory cleanup and resource management
- Error handling and graceful degradation
- Configuration-driven behavior
"""
from __future__ import annotations

import os
import pickle
import tempfile
from datetime import date, datetime
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml

from storm250.build import (
    _process_one_group,
    _run_group_in_child,
    main_pipeline,
)
from storm250.config import default_config, validate_config


####################################################### Fixtures #######################################################


@pytest.fixture(scope="function")
def minimal_config(tmp_path):
    """Minimal valid config for testing build pipeline."""
    cfg = default_config()
    cfg["root_dir"] = str(tmp_path / "Storm250")
    cfg["ensure_dirs"] = True
    cfg["execution"]["year"] = 2017
    cfg["execution"]["rewrite_existing"] = True
    cfg["execution"]["multiprocessing_method"] = "spawn"  # safer for tests
    cfg["build"]["n_workers"] = 1
    cfg["build"]["min_track_samples"] = 1
    cfg["build"]["max_range_km"] = 250.0
    cfg["build"]["grs_timeout_s"] = 30
    cfg["build"]["max_gap_hours"] = 2.0
    cfg["build"]["time_tolerance_s"] = 300
    cfg["build"]["reflectivity_dbz_thresh"] = 20
    cfg["build"]["min_blob_size"] = 2500
    cfg["build"]["buffer_km"] = 10.0
    cfg["build"]["grid_resolution_m"] = 1000
    cfg["build"]["include_nearby_km"] = 10.0
    cfg["build"]["debug_plot_limit"] = 0
    cfg["build"]["lsr_force_refresh"] = False
    cfg["build"]["nexrad_products"] = ["reflectivity"]
    validate_config(cfg)
    return cfg


@pytest.fixture(scope="function")
def radar_info_minimal():
    """Minimal radar_info for testing."""
    return {
        "KTLX": {
            "position": {"lat": 35.333, "lon": -97.278},
            "active_ranges": [{"start": None, "end": None}],
        },
        "KHGX": {
            "position": {"lat": 29.472, "lon": -95.079},
            "active_ranges": [{"start": None, "end": None}],
        },
    }


@pytest.fixture(scope="function")
def mock_grs_group():
    """Mock GR-S track group for a single storm."""
    return pd.DataFrame({
        "storm_id": [1, 1, 1],
        "radar_site": ["KTLX", "KTLX", "KTLX"],
        "time": pd.to_datetime([
            "2017-05-01 12:00:00",
            "2017-05-01 12:05:00",
            "2017-05-01 12:10:00",
        ], utc=True),
        "latitude": [35.5, 35.6, 35.7],
        "longitude": [-97.5, -97.6, -97.7],
        "distance_to_site": [50.0, 55.0, 60.0],
    })


####################################################### Unit Tests #######################################################
#  ( ensure pickles containing group + config info are passed correctly from _run_group_in_child to _process_one_group)


def test_process_one_group_requires_valid_config_pickle(tmp_path, mock_grs_group, radar_info_minimal):
    """
    Invariant: _process_one_group must fail gracefully if config pickle is missing or corrupt.
    """
    import multiprocessing as mp
    
    group_pkl = tmp_path / "group_1.pkl"
    cfg_pkl = tmp_path / "cfg_1.pkl"
    
    # Write valid group pickle
    mock_grs_group.to_pickle(group_pkl, protocol=5)
    
    # Write corrupt config pickle
    cfg_pkl.write_bytes(b"not a pickle")
    
    ctx = mp.get_context("spawn")
    errq = ctx.Queue(maxsize=100)
    
    # Should exit with code 1 due to config load failure
    p = ctx.Process(
        target=_process_one_group,
        args=(str(group_pkl), str(cfg_pkl), radar_info_minimal, False, 1, errq)
    )
    p.start()
    p.join(timeout=5)
    
    assert p.exitcode == 1, "Process should exit with code 1 on config load failure"


def test_process_one_group_handles_missing_group_pickle(tmp_path, minimal_config, radar_info_minimal):
    """
    Invariant: _process_one_group must handle missing group pickle gracefully.
    """
    import multiprocessing as mp
    
    group_pkl = tmp_path / "nonexistent_group.pkl"
    cfg_pkl = tmp_path / "cfg_1.pkl"
    
    # Write valid config
    with open(cfg_pkl, 'wb') as f:
        pickle.dump(minimal_config, f, protocol=5)
    
    ctx = mp.get_context("spawn")
    errq = ctx.Queue(maxsize=100)
    
    p = ctx.Process(
        target=_process_one_group,
        args=(str(group_pkl), str(cfg_pkl), radar_info_minimal, False, 1, errq)
    )
    p.start()
    p.join(timeout=5)
    
    # Should fail (non-zero exit) since group pickle doesn't exist
    assert p.exitcode != 0, "Process should fail when group pickle is missing"


def test_run_group_in_child_creates_temp_files(tmp_path, mock_grs_group, minimal_config, radar_info_minimal):
    """
    Invariant: _run_group_in_child must create temporary pickle files for IPC.
    """
    with patch("storm250.build.mp.Process") as mock_process_class:
        mock_proc = MagicMock()
        mock_proc.is_alive.side_effect = [True, False]  # alive once, then done
        mock_proc.exitcode = 0
        mock_process_class.return_value = mock_proc
        
        # Track what files are created during the call
        created_files = []
        original_to_pickle = pd.DataFrame.to_pickle
        
        def track_pickle(self, path, **kwargs):
            created_files.append(Path(path))
            return original_to_pickle(self, path, **kwargs)
        
        with patch.object(pd.DataFrame, 'to_pickle', track_pickle):
            try:
                _run_group_in_child(
                    sid=1,
                    group=mock_grs_group,
                    cfg=minimal_config,
                    radar_info=radar_info_minimal,
                    debug=False
                )
            except Exception:
                pass  # Expected to fail in mock environment
        
        # Verify temp files were attempted to be created
        assert len(created_files) > 0, "Should create temporary pickle files for IPC"


####################################################### Integration Tests #######################################################
#                                                ( testing main_pipeline behavior )

def test_main_pipeline_requires_valid_config():
    """
    Invariant: main_pipeline must validate config structure before execution.
    """
    invalid_cfg = {"execution": {}}  # Missing required keys
    
    with pytest.raises((KeyError, ValueError, TypeError)):
        main_pipeline(invalid_cfg, radar_info={}, debug_flag=False)


def test_main_pipeline_creates_output_directories(tmp_path, minimal_config, radar_info_minimal):
    """
    Invariant: main_pipeline must create necessary output directories.
    """
    # Mock all external data loading to avoid network calls
    with patch("storm250.build.load_raw_lsr") as mock_lsr, \
         patch("storm250.build.load_raw_spc") as mock_spc, \
         patch("storm250.build.load_grs_tracks") as mock_grs, \
         patch("storm250.build.build_year_manifest_and_catalog") as mock_manifest:
        
        # Return empty dataframes
        mock_lsr.return_value = pd.DataFrame(columns=["time", "lat", "lon", "gust", "type"])
        mock_spc.return_value = pd.DataFrame(columns=["time", "lat", "lon", "gust", "type"])
        mock_grs.return_value = pd.DataFrame(columns=["storm_id", "radar_site", "time"])
        
        main_pipeline(minimal_config, radar_info_minimal, debug_flag=False)
        
        # Verify output directory was created
        from storm250.config import get_path
        dataset_dir = get_path(minimal_config, "dataset_out_dir")
        assert dataset_dir.exists(), "Dataset output directory should be created"


def test_main_pipeline_skips_existing_storms_when_rewrite_false(tmp_path, minimal_config, radar_info_minimal):
    """
    Invariant: main_pipeline must skip already-processed storms when rewrite_existing=False.
    """
    minimal_config["execution"]["rewrite_existing"] = False
    
    # Create fake existing storm directory with .h5 file
    from storm250.config import get_path
    dataset_dir = get_path(minimal_config, "dataset_out_dir")
    year_dir = dataset_dir / "2017"
    storm_dir = year_dir / "KTLX" / "storm_1"
    storm_dir.mkdir(parents=True, exist_ok=True)
    (storm_dir / "fake_scan.h5").write_bytes(b"fake")
    
    mock_grs_df = pd.DataFrame({
        "storm_id": [1, 1],
        "radar_site": ["KTLX", "KTLX"],
        "time": pd.to_datetime(["2017-05-01 12:00:00", "2017-05-01 12:05:00"], utc=True),
    })
    
    with patch("storm250.build.load_raw_lsr") as mock_lsr, \
         patch("storm250.build.load_raw_spc") as mock_spc, \
         patch("storm250.build.load_grs_tracks") as mock_grs, \
         patch("storm250.build._run_group_in_child") as mock_run_child, \
         patch("storm250.build.build_year_manifest_and_catalog"):
        
        mock_lsr.return_value = pd.DataFrame(columns=["time", "lat", "lon", "gust", "type"])
        mock_spc.return_value = pd.DataFrame(columns=["time", "lat", "lon", "gust", "type"])
        mock_grs.return_value = mock_grs_df
        
        main_pipeline(minimal_config, radar_info_minimal, debug_flag=False)
        
        # _run_group_in_child should NOT be called since storm already exists
        mock_run_child.assert_not_called()


def test_main_pipeline_processes_storms_when_rewrite_true(tmp_path, minimal_config, radar_info_minimal):
    """
    Invariant: main_pipeline must process all storms when rewrite_existing=True, even if they exist.
    """
    minimal_config["execution"]["rewrite_existing"] = True
    
    # Create fake existing storm
    from storm250.config import get_path
    dataset_dir = get_path(minimal_config, "dataset_out_dir")
    year_dir = dataset_dir / "2017"
    storm_dir = year_dir / "KTLX" / "storm_1"
    storm_dir.mkdir(parents=True, exist_ok=True)
    (storm_dir / "fake_scan.h5").write_bytes(b"fake")
    
    mock_grs_df = pd.DataFrame({
        "storm_id": [1, 1],
        "radar_site": ["KTLX", "KTLX"],
        "time": pd.to_datetime(["2017-05-01 12:00:00", "2017-05-01 12:05:00"], utc=True),
    })
    
    with patch("storm250.build.load_raw_lsr") as mock_lsr, \
         patch("storm250.build.load_raw_spc") as mock_spc, \
         patch("storm250.build.load_grs_tracks") as mock_grs, \
         patch("storm250.build._run_group_in_child") as mock_run_child, \
         patch("storm250.build.build_year_manifest_and_catalog"):
        
        mock_lsr.return_value = pd.DataFrame(columns=["time", "lat", "lon", "gust", "type"])
        mock_spc.return_value = pd.DataFrame(columns=["time", "lat", "lon", "gust", "type"])
        mock_grs.return_value = mock_grs_df
        
        main_pipeline(minimal_config, radar_info_minimal, debug_flag=False)
        
        # _run_group_in_child SHOULD be called since rewrite=True
        assert mock_run_child.call_count == 1


def test_main_pipeline_continues_on_child_failure(tmp_path, minimal_config, radar_info_minimal):
    """
    Invariant: main_pipeline must continue processing other storms if one child fails.
    """
    mock_grs_df = pd.DataFrame({
        "storm_id": [1, 1, 2, 2],
        "radar_site": ["KTLX", "KTLX", "KHGX", "KHGX"],
        "time": pd.to_datetime([
            "2017-05-01 12:00:00", "2017-05-01 12:05:00",
            "2017-05-01 13:00:00", "2017-05-01 13:05:00"
        ], utc=True),
    })
    
    call_count = 0
    def mock_run_child_side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("Child failed")
        # Second call succeeds
    
    with patch("storm250.build.load_raw_lsr") as mock_lsr, \
         patch("storm250.build.load_raw_spc") as mock_spc, \
         patch("storm250.build.load_grs_tracks") as mock_grs, \
         patch("storm250.build._run_group_in_child", side_effect=mock_run_child_side_effect) as mock_run_child, \
         patch("storm250.build.build_year_manifest_and_catalog"):
        
        mock_lsr.return_value = pd.DataFrame(columns=["time", "lat", "lon", "gust", "type"])
        mock_spc.return_value = pd.DataFrame(columns=["time", "lat", "lon", "gust", "type"])
        mock_grs.return_value = mock_grs_df
        
        # Should not raise, should continue to second storm
        main_pipeline(minimal_config, radar_info_minimal, debug_flag=False)
        
        # Both storms should be attempted
        assert mock_run_child.call_count == 2


def test_main_pipeline_builds_manifest_at_end(tmp_path, minimal_config, radar_info_minimal):
    """
    Invariant: main_pipeline must build year manifest and catalog after processing all storms.
    """
    with patch("storm250.build.load_raw_lsr") as mock_lsr, \
         patch("storm250.build.load_raw_spc") as mock_spc, \
         patch("storm250.build.load_grs_tracks") as mock_grs, \
         patch("storm250.build.build_year_manifest_and_catalog") as mock_manifest:
        
        mock_lsr.return_value = pd.DataFrame(columns=["time", "lat", "lon", "gust", "type"])
        mock_spc.return_value = pd.DataFrame(columns=["time", "lat", "lon", "gust", "type"])
        mock_grs.return_value = pd.DataFrame(columns=["storm_id", "radar_site", "time"])
        
        main_pipeline(minimal_config, radar_info_minimal, debug_flag=False)
        
        # Manifest should be built
        mock_manifest.assert_called_once()
        
        # Verify it was called with correct year directory
        call_args = mock_manifest.call_args
        year_dir_arg = call_args[0][0]
        assert "2017" in str(year_dir_arg)