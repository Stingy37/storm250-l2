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


def load_radar_info(cfg):
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
