# tests/test_pipeline_utils.py

from __future__ import annotations
from queue import Queue
from storm250.pipeline_utils import _QueueStream, build_saved_storm_index, should_skip_sid

import pytest

# test _QueueStream newline behavior
def test_queue_stream_emits_lines_on_newline():
    """
    _QueueStream buffers until it sees a newline, then emits each line into the queue.
    """
    q = Queue()
    sid = 123
    qs = _QueueStream(q, kind="log", sid=sid)

    # Nothing should be emitted until a newline appears
    qs.write("hello")
    assert q.empty()

    # Writing a newline flushes the buffer and emits lines
    qs.write("\n")
    kind, msg = q.get_nowait()

    # check the items are as expected, and queue is flushed 
    assert kind == "log"
    assert msg == f"[child sid={sid}] hello"
    assert q.empty()

    # Multiple lines in one write should emit multiple queue messages
    qs.write("a\nb\nc\n")
    got = [q.get_nowait() for _ in range(3)]
    assert got == [
        ("log", f"[child sid={sid}] a"),
        ("log", f"[child sid={sid}] b"),
        ("log", f"[child sid={sid}] c"),
    ]

# test _QueueStream flush behavior
def test_queue_stream_flush_emits_partial_buffer():
    """
    flush() should emit buffered text even if no newline was written.
    """
    q = Queue()
    sid = 7
    qs = _QueueStream(q, kind="err", sid=sid)

    qs.write("partial-without-newline")
    assert q.empty()

    qs.flush()
    kind, msg = q.get_nowait()
    assert kind == "err"
    assert msg == f"[child sid={sid}] partial-without-newline"
    assert q.empty()

# helper function for testing build_saved_storm_index 
def _make_storm_dir(base_dir, year: str, site: str, sid: int, *, with_h5: bool):
    """
    Create base_dir/{year}/{site}/storm_{sid}/ and optionally drop a .h5 inside.
    Returns storm_dir (Path).
    """
    year_dir = base_dir / year
    site_dir = year_dir / site
    storm_dir = site_dir / f"storm_{sid}"
    storm_dir.mkdir(parents=True, exist_ok=True)

    if with_h5:
        (storm_dir / "scan_0001.h5").write_bytes(b"not really hdf5, but extension is what matters")
    else:
        (storm_dir / "note.txt").write_text("no h5 here")

    return storm_dir


def test_build_saved_storm_index_detects_only_storms_with_h5(tmp_path):
    """
    Tests build_saved_storm_index behavior. 

    build_saved_storm_index should:
      - find storms only if storm_{sid}/ contains at least one .h5 file
      - ignore storms without .h5
      - aggregate across sites for the same (year, sid)
      - ignore non-year folders
    """
    base = tmp_path / "training_base"
    base.mkdir(parents=True, exist_ok=True)

    # Valid year folder
    s1 = _make_storm_dir(base, "2017", "KAAA", 10, with_h5=True)
    s2 = _make_storm_dir(base, "2017", "KBBB", 10, with_h5=True)  # same sid, different site -> (year, sid) 
                                                                  #             \- key should have multiple paths now
    _make_storm_dir(base, "2017", "KAAA", 11, with_h5=False)      # should be ignored (no h5)
    _make_storm_dir(base, "2017", "KCCC", 12, with_h5=True)

    # Non-year folder should be ignored
    _make_storm_dir(base, "not_a_year", "KAAA", 99, with_h5=True)

    idx = build_saved_storm_index(str(base), debug=True)

    assert (2017, 10) in idx
    assert (2017, 12) in idx
    assert (2017, 11) not in idx
    assert (2017, 99) not in idx

    # order isn't guaranteed -> compare sets
    assert set(idx[(2017, 10)]) == {str(s1), str(s2)}
    assert set(idx[(2017, 12)]) == {str(base / "2017" / "KCCC" / "storm_12")}


def test_build_saved_storm_index_returns_empty_when_base_dir_missing(tmp_path):
    missing = tmp_path / "does_not_exist"
    idx = build_saved_storm_index(str(missing), debug=True)
    assert idx == {}


def test_should_skip_sid_respects_rewrite_and_existing_index():
    """
    tests should_skip_sid behavior
    """
    # mocked existing dictionary, containing two (year, SID) keys
    existing = {
        (2017, 10): ["/fake/path/a"],
        (2017, 12): ["/fake/path/b"],
    }

    # rewrite = True -> never skip
    assert should_skip_sid(2017, 10, existing, rewrite=True, debug=True) is False
    assert should_skip_sid(2017, 999, existing, rewrite=True, debug=True) is False

    # rewrite = False -> skip only if present in index
    assert should_skip_sid(2017, 10, existing, rewrite=False, debug=True) is True
    assert should_skip_sid(2017, 999, existing, rewrite=False, debug=True) is False
    #                             |- not a valid SID

    # if index dict is None -> never skip 
    assert should_skip_sid(2017, 10, None, rewrite=False, debug=True) is False
