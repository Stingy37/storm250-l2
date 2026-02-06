"""
Dataset builder entry point - loads config and runs pipeline.
"""
import cProfile
import pstats
import multiprocessing as mp
import yaml
import argparse

from pathlib import Path

from storm250.config import load_config
from storm250.build import main_pipeline


def parse_args():
    """
    Here we add a method to parse arguments supplied by a CLI command entry point, such as 
    ' python build_dataset.py --config configs/config_v1.0.0.yaml '

    We define the possible arguments and add description / help values for the --help flag. 
    """
    # create the parser 
    p = argparse.ArgumentParser(description="Storm250 dataset builder")
    
    # define the accepted arguments (supplied through --arg_name arg_value)
    #                   /- argument for path to the used configs (MUST be supplied by the user)
    p.add_argument("--config", required=True, help="Path to config YAML, where a relative path like 'configs/config_v1.0.0.yaml' is expected")
    #                   /- argument for usage of profiler (not required, default -> not used)
    p.add_argument("--profile", action="store_true", help="Enable cProfile performance profiling")
    
    return p.parse_args()


def load_radar_info(cfg):
    """
    Helper that loads radar_info from YAML referenced in config.
    """
    radar_yaml = cfg.get("radar_info_yaml", "radar_info.yaml")
    radar_path = Path("configs") / radar_yaml
    
    with open(radar_path, 'r') as f:
        return yaml.safe_load(f)


def run_pipeline_helper(cfg, radar_info): 
    """
    Helper to centralize main_pipeline calls. 
    """
    main_pipeline(
        cfg=cfg,
        radar_info=radar_info,
        debug_flag=cfg.get("logging", {}).get("level") == "DEBUG"
    )


if __name__ == "__main__":
    # load supplied args from CLI entry point
    args = parse_args()

    # Load configuration
    config_path = args.config
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
        if args.profile is True: 
            with cProfile.Profile() as pr:
                run_pipeline_helper(cfg, radar_info)
        else:
            run_pipeline_helper(cfg, radar_info)
    finally:
        if args.profile:
            # Print profiling stats
            stats = pstats.Stats(pr)
            stats.sort_stats('cumtime').print_stats(30)
            del stats, pr

        # Cleanup (regardless of profiler)
        import gc, ctypes
        gc.collect()
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass
