from pathlib import Path
import yaml

from storm250.config import load_config, resolve_path, get_path, validate_config, default_config

# test if resolve_path works for old colab paths 
def test_resolve_path_relative(tmp_path):
    cfg = default_config()
    cfg["root_dir"] = str(tmp_path / "Storm250")
    cfg["ensure_dirs"] = True
    validate_config(cfg)

    p = resolve_path(cfg, "Datasets/temp")
    assert p == (tmp_path / "Storm250" / "Datasets" / "temp")

# test if resolve_path works for absolute paths 
def test_resolve_path_absolute(tmp_path):
    cfg = default_config()
    cfg["root_dir"] = str(tmp_path / "Storm250")
    validate_config(cfg)

    abs_p = tmp_path / "x" / "y"
    out = resolve_path(cfg, str(abs_p))
    assert out == abs_p

def test_get_path_creates_dirs(tmp_path):
    cfg = default_config()
    cfg["root_dir"] = str(tmp_path / "Storm250")
    cfg["ensure_dirs"] = True
    validate_config(cfg)

    out_dir = get_path(cfg, "dataset_out_dir")
    assert out_dir.exists()
    assert out_dir.is_dir()

# test that config loading / overwriting behavior is as expected 
def test_load_config_merges_and_validates(tmp_path):
    # Write a minimal YAML overriding just root_dir + one knob
    cfg_path = tmp_path / "v0.1.yaml"
    user_cfg = {
        "root_dir": str(tmp_path / "Storm250"), # -------------- overwritten stuff
        "build": {"n_workers": 3},# --------------------------/

        "ensure_dirs": True,
    }
    cfg_path.write_text(yaml.safe_dump(user_cfg), encoding="utf-8")

    # merges with default configs and returns it, ready to be used in dataset pipeline
    cfg = load_config(cfg_path)
    assert cfg["build"]["n_workers"] == 3 # overwritten value, so should be 3 (NOT 8, the default)

    #                  /---- (for the dataset with specific configs)
    # ensure a derived directory is created by checking one of the paths in DEFAULT_CONFIG that hasn't been overwritten (temp_dir)
    #                                 full path should be EBS-style path (tmp_path/Storm250/Datasets/temp) ------/
    #                                 but we just check if it exists 
    assert get_path(cfg = cfg, key = "temp_dir").exists()
