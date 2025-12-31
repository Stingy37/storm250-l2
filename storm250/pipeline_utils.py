
class _QueueStream(io.TextIOBase):
    def __init__(self, q, kind, sid):
        self.q = q        # where q is the queue that we put the transformed stdout / stderr into 
        self.kind = kind  # "log" | "err" | "warn"
        self.sid = sid
        self._buf = []

    def write(self, s):
        if not s:
            return 0
        if not isinstance(s, str):
            s = str(s)
        self._buf.append(s)
        if "\n" in s:
            text = "".join(self._buf)
            self._buf.clear()
            for line in text.splitlines():
                try:
                    #  /- add created object to the queue (usually a errq), which then sends to parent
                    #                                                          |- which then unpacks -> prints
                    self.q.put_nowait((self.kind, f"[child sid={self.sid}] {line}"))
                except Exception:
                    # drop if queue is full to avoid deadlock
                    pass
        return len(s)

    def flush(self):
        if self._buf:
            text = "".join(self._buf)
            self._buf.clear()
            try:
                self.q.put_nowait((self.kind, f"[child sid={self.sid}] {text}"))
            except Exception:
                pass

    def isatty(self): return False
    @property
    def encoding(self): return "utf-8"


def build_saved_storm_index(base_dir: str, debug: bool = False) -> dict[tuple[int, int], list[str]]:
    """
    Fast one-pass index: {(year, sid) -> [storm_dir_paths]} for storms that already
    have at least one .h5 in base_dir/{year}/{site}/storm_{sid}/.

    Notes:
      - Robust to extra/non-numeric year folders and non-storm dirs.
      - Aggregates across sites (multiple paths per (year, sid)).
    """
    import os, re

    saved: dict[tuple[int, int], list[str]] = {}
    try:
        if not os.path.isdir(base_dir):
            if debug:
                print(f"[build_saved_storm_index] base_dir '{base_dir}' does not exist yet.")
            return saved

        with os.scandir(base_dir) as years:
            for y in years:
                if not y.is_dir():
                    continue
                # Parse year folder name (must start with digits)
                ym = re.match(r"^\s*(\d{4})\b", y.name)
                if not ym:
                    continue
                try:
                    year = int(ym.group(1))
                except Exception:
                    continue

                with os.scandir(y.path) as sites:
                    for s in sites:
                        if not s.is_dir():
                            continue
                        with os.scandir(s.path) as storms:
                            for st in storms:
                                if not st.is_dir():
                                    continue
                                name = st.name
                                if not name.startswith("storm_"):
                                    continue
                                m = re.match(r"^storm_(\d+)", name)
                                if not m:
                                    continue
                                sid = int(m.group(1))

                                # Consider it "present" only if there's at least one .h5 inside
                                has_h5 = False
                                try:
                                    with os.scandir(st.path) as files:
                                        for f in files:
                                            if f.is_file() and f.name.endswith(".h5"):
                                                has_h5 = True
                                                break
                                except Exception:
                                    pass
                                if not has_h5:
                                    continue

                                saved.setdefault((year, sid), []).append(st.path)

        if debug:
            n_keys = len(saved)
            n_dirs = sum(len(v) for v in saved.values())
            print(f"[build_saved_storm_index] indexed {n_keys} (year,sid) keys across {n_dirs} folder(s).")
        return saved

    except Exception as e:
        if debug:
            print(f"[build_saved_storm_index] error while scanning '{base_dir}': {e}")
        return saved


def should_skip_sid(year: int, sid: int, existing_index: dict[tuple[int, int], list[str]] | None,
                    rewrite: bool, debug: bool = False) -> bool:
    """
    Return True if we should skip processing this (year, sid):
      - skip if rewrite=False and (year, sid) is already in existing_index
      - process if rewrite=True (always)
    """
    if rewrite:
        if debug:
            print(f"[should_skip_sid] rewrite=True → will process ({year}, SID {sid}) regardless of existing files.")
        return False

    present = (existing_index is not None) and ((year, sid) in existing_index)
    if debug:
        if present:
            paths = existing_index.get((year, sid), [])
            path_hint = paths[0] if paths else "<unknown>"
            print(f"[should_skip_sid] ({year}, SID {sid}) already present at {path_hint} → skipping.")
        else:
            print(f"[should_skip_sid] ({year}, SID {sid}) not found in index → will process.")
    return present
