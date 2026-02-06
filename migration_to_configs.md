# Config-Driven Migration Plan

## Overview
Migrate `build.py` and `build_dataset.py` to use centralized config YAML (`configs/config_v1.0.0.yaml`) and path resolution utilities from `config.py`, eliminating all hardcoded paths and parameters.

---

## Current State Analysis

### What's Already Config-Ready ✅
- `config.py` has comprehensive path resolution (`resolve_path`, `get_path`)
- `configs/config_v1.0.0.yaml` has most build knobs defined
- Test suite already uses config loading in `test_config.py`

### What Needs Migration 🔧

#### 1. **Missing Config Knobs**
These parameters are currently hardcoded but should be in YAML:

**In `build.py` → `main_pipeline()`:**
- `train_rewrite` (bool) - currently function parameter
- `training_base` (str) - currently function parameter with default
- `year` (int) - currently function parameter

**In `build.py` → `_process_one_group()`:**
- `level2_base` = "unidata-nexrad-level2" (hardcoded)
- `cache_dir` = "Datasets/nexrad_datasets/level_two_raw" (hardcoded)
- `product_filter` = ["reflectivity"] (hardcoded)
- `time_tolerance_seconds` = 29 (hardcoded, but exists in config as `time_tolerance_s`)
- `threshold` = 20 (hardcoded, but exists in config as `reflectivity_dbz_thresh`)
- `min_size` = 6000 (hardcoded)
- `pad_km` = 5.0 (hardcoded, but exists in config as `buffer_km`)
- `grid_res_m` = 250.0 (hardcoded)
- `buffer_km` = 5.0 (hardcoded, exists in config)
- `include_nearby_km` = 3.0 (hardcoded)
- `debug_plot_dir` path construction (uses hardcoded "Logs/plots")
- `debug_plot_limit` = 0 (hardcoded)

**In `build.py` → `main_pipeline()` data loading:**
- LSR cache_dir = "Datasets/surface_obs_datasets/lsr_raw" (hardcoded)
- LSR force_refresh = False (hardcoded)
- SPC start/end dates (derived from year parameter)
- GRS base_url (hardcoded)
- GRS min_rows = 60 (hardcoded, but exists in config as `min_track_samples`)
- GRS max_distance_km = 250 (hardcoded, but exists in config as `max_range_km`)
- GRS timeout = 10 (hardcoded)
- GRS save_dir (hardcoded)
- GRS max_gap_hours = 6.0 (hardcoded, but exists in config as `max_gap_hours`)

**In `release/build_dataset.py`:**
- `debug_flag` = False (hardcoded)
- `year` = 2017 (hardcoded)
- `radar_info` (loaded separately, not from config)
- `train_rewrite` = False (hardcoded)
- `training_base` (hardcoded)
- Multiprocessing start method = "fork" (hardcoded)

#### 2. **Path Resolution Issues**
All path strings need to use `resolve_path()` or `get_path()`:
- Direct path strings like `"Datasets/temp"` → `get_path(cfg, "temp_dir")`
- Path construction like `os.path.join("Logs", "plots", ...)` → `get_path(cfg, "plots_dir")`

#### 3. **Function Signatures Need Config Parameter**
These functions need `cfg` parameter added:

**Core pipeline functions:**
- `main_pipeline()` - needs cfg instead of individual params
- `_process_one_group()` - needs cfg for paths and knobs
- `_run_group_in_child()` - needs cfg to pass to child

**Module functions (check if they use hardcoded paths):**
- `load_raw_lsr()` - already has cache_dir param ✅
- `load_raw_spc()` - need to check implementation
- `load_grs_tracks()` - already has path params ✅
- `find_radar_scans()` - already has path params ✅
- `build_bboxes_for_linked_df()` - need to check for hardcoded params
- `save_df_for_training()` - already has base_dir param ✅
- `build_year_manifest_and_catalog()` - already has year_dir param ✅
- `build_saved_storm_index()` - already has base_dir param ✅
- `should_skip_sid()` - no paths, just logic ✅

---

## Migration Strategy

### Phase 1: Extend Config YAML ⚙️

**Add missing knobs to `configs/config_v1.0.0.yaml`:**

```yaml
# Add to existing build section:
build:
  # ... existing knobs ...
  
  # Cropping/bbox parameters
  min_blob_size: 6000
  grid_resolution_m: 250.0
  include_nearby_km: 3.0
  debug_plot_limit: 0
  
  # Data loading parameters
  lsr_force_refresh: false
  grs_base_url: "https://data-osdf.rda.ucar.edu/ncar/rda/d841006/tracks"
  grs_timeout_s: 10
  
  # Product filtering
  nexrad_products: ["reflectivity"]

# Add execution section:
execution:
  year: 2017
  rewrite_existing: false
  multiprocessing_method: "fork"

# Add radar_info reference (or inline):
radar_info_yaml: "radar_info.yaml"
```

### Phase 2: Update build.py 🔨

**2.1 Add imports:**
```python
from storm250.config import load_config, get_path, resolve_path
```

**2.2 Refactor `main_pipeline()` signature:**
```python
# OLD:
def main_pipeline(debug_flag, year, radar_info, train_rewrite: bool = False, 
                  training_base: str = "Datasets/training_datasets/level_two"):

# NEW:
def main_pipeline(cfg: dict, radar_info: dict, debug_flag: bool = False):
    """
    Run the full pipeline using configuration from cfg dict.
    
    Parameters
    ----------
    cfg : dict
        Configuration loaded via load_config()
    radar_info : dict
        Radar metadata (loaded separately or from cfg reference)
    debug_flag : bool
        Enable debug logging
    """
    # Extract from config
    year = cfg["execution"]["year"]
    train_rewrite = cfg["execution"]["rewrite_existing"]
    training_base = get_path(cfg, "dataset_out_dir")
    
    # ... rest of implementation
```

**2.3 Refactor `_process_one_group()` signature:**
```python
# OLD:
def _process_one_group(group_pkl_path, radar_info, training_base, debug_flag, sid, errq):

# NEW:
def _process_one_group(group_pkl_path, cfg_pkl_path, radar_info, debug_flag, sid, errq):
    """
    cfg_pkl_path: Path to pickled config dict (passed to child process)
    """
    import pickle
    
    # Load config in child process
    with open(cfg_pkl_path, 'rb') as f:
        cfg = pickle.load(f)
    
    # Extract all parameters from config
    training_base = get_path(cfg, "dataset_out_dir")
    level2_base = cfg["aws"]["nexrad_bucket"]
    cache_dir = get_path(cfg, "nexrad_lv2_cache_dir")
    product_filter = cfg["build"]["nexrad_products"]
    time_tolerance_s = cfg["build"]["time_tolerance_s"]
    threshold = cfg["build"]["reflectivity_dbz_thresh"]
    min_size = cfg["build"]["min_blob_size"]
    buffer_km = cfg["build"]["buffer_km"]
    grid_res_m = cfg["build"]["grid_resolution_m"]
    include_nearby_km = cfg["build"]["include_nearby_km"]
    debug_plot_limit = cfg["build"]["debug_plot_limit"]
    
    # Construct plot dir using config
    plots_base = get_path(cfg, "plots_dir")
    plot_dir = plots_base / f"bbox_sid{sid}_pid{os.getpid()}"
    
    # ... rest of implementation using extracted values
```

**2.4 Update `_run_group_in_child()`:**
```python
# OLD:
def _run_group_in_child(sid, group, radar_info, training_base, debug):

# NEW:
def _run_group_in_child(sid, group, cfg, radar_info, debug):
    """
    cfg: Config dict to pickle and pass to child
    """
    with tempfile.TemporaryDirectory() as td:
        group_pkl = Path(td) / f"group_{sid}.pkl"
        cfg_pkl = Path(td) / f"cfg_{sid}.pkl"
        
        group.to_pickle(group_pkl, protocol=5)
        
        # Pickle config for child
        import pickle
        with open(cfg_pkl, 'wb') as f:
            pickle.dump(cfg, f, protocol=5)
        
        # ... rest with updated args
        p = ctx.Process(
            target=_process_one_group,
            args=(str(group_pkl), str(cfg_pkl), radar_info, debug, sid, errq)
        )
```

**2.5 Update data loading calls in `main_pipeline()`:**
```python
# LSR loading
lsr_df = load_raw_lsr(
    start=start,
    end=end,
    debug=False,
    cache_dir=get_path(cfg, "lsr_raw_dir"),
    force_refresh=cfg["build"]["lsr_force_refresh"],
)

# GRS loading
grs_df = load_grs_tracks(
    year=year,
    radar_info=radar_info,
    base_url=cfg["build"]["grs_base_url"],
    min_rows=cfg["build"]["min_track_samples"],
    max_distance_km=cfg["build"]["max_range_km"],
    debug=False,
    timeout=cfg["build"]["grs_timeout_s"],
    save_dir=get_path(cfg, "grs_raw_dir"),
    processed_cache_dir=get_path(cfg, "grs_processed_dir"),
    max_gap_hours=cfg["build"]["max_gap_hours"],
)

# Manifest building
year_dir = get_path(cfg, "dataset_out_dir") / str(year)
build_year_manifest_and_catalog(
    str(year_dir),
    update_schema_checksums=True,
    debug=True
)
```

### Phase 3: Update build_dataset.py 🚀

**3.1 Complete rewrite:**
```python
#!/usr/bin/env python3
"""
Dataset builder entry point - loads config and runs pipeline.
"""
import cProfile
import pstats
import multiprocessing as mp
import yaml
from pathlib import Path

from storm250.config import load_config
from storm250.build import main_pipeline


def load_radar_info(cfg: dict) -> dict:
    """Load radar_info from YAML referenced in config."""
    radar_yaml = cfg.get("radar_info_yaml", "radar_info.yaml")
    radar_path = Path("configs") / radar_yaml
    
    with open(radar_path, 'r') as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    # Load configuration
    config_path = "configs/config_v1.0.0.yaml"
    cfg = load_config(config_path)
    
    # Load radar info
    radar_info = load_radar_info(cfg)
    
    # Set multiprocessing method from config
    mp_method = cfg.get("execution", {}).get("multiprocessing_method", "fork")
    try:
        mp.set_start_method(mp_method)
    except RuntimeError:
        pass  # Already set
    
    # Run pipeline with profiling
    try:
        with cProfile.Profile() as pr:
            main_pipeline(
                cfg=cfg,
                radar_info=radar_info,
                debug_flag=cfg.get("logging", {}).get("level") == "DEBUG"
            )
    finally:
        # Print profiling stats
        stats = pstats.Stats(pr)
        stats.sort_stats('cumtime').print_stats(30)
        
        # Cleanup
        del stats, pr
        import gc, ctypes
        gc.collect()
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass
```

### Phase 4: Update Tests 🧪

**4.1 Tests that need config parameter:**

Most tests use fixtures and mock data, so they should continue working. However, any integration tests that call the main functions need updates:

**Files to check:**
- `tests/test_build.py` - if it exists and tests main_pipeline
- Any integration tests that use real paths

**Pattern for test updates:**
```python
def test_something(tmp_path):
    # Create minimal config
    cfg = {
        "root_dir": str(tmp_path),
        "paths": {
            "dataset_out_dir": "output",
            "nexrad_lv2_cache_dir": "cache",
            # ... other required paths
        },
        "build": {
            "max_range_km": 250.0,
            "time_tolerance_s": 29,
            # ... other required knobs
        },
        # ... other required sections
    }
    
    # Validate config
    from storm250.config import validate_config
    cfg = validate_config(cfg)
    
    # Use in test
    result = some_function(cfg, ...)
```

**4.2 Tests that should NOT change:**
- Unit tests that don't use paths (geometry, math, etc.)
- Tests that already use fixtures with explicit paths
- Tests using `tmp_path` fixture

---

## Implementation Checklist

### Config YAML Updates
- [ ] Add `build.min_blob_size`
- [ ] Add `build.grid_resolution_m`
- [ ] Add `build.include_nearby_km`
- [ ] Add `build.debug_plot_limit`
- [ ] Add `build.lsr_force_refresh`
- [ ] Add `build.grs_base_url`
- [ ] Add `build.grs_timeout_s`
- [ ] Add `build.nexrad_products` (list)
- [ ] Add `execution.year`
- [ ] Add `execution.rewrite_existing`
- [ ] Add `execution.multiprocessing_method`
- [ ] Add `radar_info_yaml` reference

### build.py Updates
- [ ] Add config imports
- [ ] Refactor `main_pipeline()` signature
- [ ] Extract all config values in `main_pipeline()`
- [ ] Update `_process_one_group()` to accept cfg_pkl_path
- [ ] Load config in child process
- [ ] Extract all config values in `_process_one_group()`
- [ ] Update `_run_group_in_child()` to pickle config
- [ ] Update all path constructions to use `get_path()`
- [ ] Update LSR loading call
- [ ] Update SPC loading call (if needed)
- [ ] Update GRS loading call
- [ ] Update manifest building call
- [ ] Update all hardcoded parameters to use config

### build_dataset.py Updates
- [ ] Complete rewrite with config loading
- [ ] Add radar_info loading function
- [ ] Update main block to use config
- [ ] Remove all hardcoded parameters

### Test Updates
- [ ] Review `tests/test_build.py` (if exists)
- [ ] Update any integration tests using main_pipeline
- [ ] Verify unit tests still pass
- [ ] Add test for config loading in build context

### Documentation
- [ ] Update README with config usage
- [ ] Document new config parameters
- [ ] Add migration notes for users

---

## Benefits After Migration

1. **Single Source of Truth**: All parameters in one YAML file
2. **Environment Portability**: Easy to switch between local/EC2/Colab
3. **Version Control**: Config changes tracked in git
4. **Reproducibility**: Exact parameters saved with dataset version
5. **Flexibility**: Easy to run multiple configurations
6. **Testing**: Easier to create test configs
7. **Maintenance**: No scattered hardcoded values

---

## Backward Compatibility

**Breaking Changes:**
- `main_pipeline()` signature changes (cfg instead of individual params)
- `build_dataset.py` requires config file

**Migration Path for Users:**
1. Update config YAML with their parameters
2. Update any custom scripts calling `main_pipeline()`
3. Use new `build_dataset.py` entry point

---

## Risk Assessment

**Low Risk:**
- Config loading (already tested)
- Path resolution (already tested)
- YAML parsing (standard library)

**Medium Risk:**
- Pickling config for child processes (test serialization)
- Missing config keys (add validation)

**Mitigation:**
- Comprehensive validation in `validate_config()`
- Clear error messages for missing keys
- Fallback to defaults where sensible
- Thorough testing before deployment

---

## Timeline Estimate

- **Phase 1** (Config YAML): 30 minutes
- **Phase 2** (build.py): 2-3 hours
- **Phase 3** (build_dataset.py): 30 minutes
- **Phase 4** (Tests): 1-2 hours
- **Testing & Validation**: 2-3 hours

**Total**: ~6-9 hours of focused work
