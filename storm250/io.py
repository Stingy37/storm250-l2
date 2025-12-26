def _save_gz_pickle(obj, path, debug=False):
    try:
        _ensure_dir(os.path.dirname(path))
        with gzip.open(path, "wb") as f:
            pickle.dump(obj, f, protocol=4)
    except Exception as e:
        if debug: print(f"[find_radar_scans] skeleton write failed {path}: {e}")


def _load_gz_pickle(path, debug=False):
    try:
        with gzip.open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        if debug: print(f"[find_radar_scans] skeleton read failed {path}: {e}")
        return None


def _save_field_pack(base_path, field_dict, downcast=True, debug=False):
    """
    Save a single field into two small files:
      - {base_path}.npz  : compressed data + mask
      - {base_path}.json : tiny metadata (everything except 'data')
    """
    try:
        _ensure_dir(os.path.dirname(base_path))
        arr = field_dict["data"]
        data = np.asarray(arr, dtype=np.float32 if downcast else arr.dtype)
        mask = np.asarray(_ma.getmaskarray(arr), dtype=np.uint8)
        np.savez_compressed(base_path + ".npz", data=data, mask=mask)
        meta = {k: v for k, v in field_dict.items() if k != "data"}
        with open(base_path + ".json", "w") as f:
            json.dump(meta, f)
    except Exception as e:
        if debug: print(f"[find_radar_scans] field save failed {base_path}: {e}")


def _load_field_pack(base_path, debug=False):
    """
    Load a single field pack and return a pyart-compatible field dict.
    """
    try:
        z = np.load(base_path + ".npz", allow_pickle=False)
        with open(base_path + ".json", "r") as f:
            meta = json.load(f)
        data = _ma.MaskedArray(z["data"], mask=z["mask"].astype(bool))
        meta["data"] = data
        return meta
    except Exception as e:
        if debug: print(f"[find_radar_scans] field load failed {base_path}: {e}")
        return None