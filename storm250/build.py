import s3fs
import cProfile
import pstats
import threading
import os, gc, ctypes, tempfile
import pandas as pd
import multiprocessing as mp
import sys, traceback, faulthandler, warnings
import queue as _q
import signal

from threading import Thread
from IPython.display import display
from datetime import date, datetime, timedelta


faulthandler.enable()


def _showwarn():
    pass

def log():
    pass

# Process one SID on a child process after GR-S is linked, to keep RAM usage down
def _process_one_group(group_pkl_path, radar_info, training_base, debug_flag, sid, errq):
    """
    Code for what happens in one child process, where each process is basically a separate program instance. 
        - Allows for program isolation (child process for one SID can fail gracefully).
        - important: allows for memory cleanup through exiting process when finished, which pythonic for-loop doesn't allow for. 
    """
    import pandas as pd
    import s3fs, gc, ctypes, os, sys, traceback, warnings

    # point the process's standard output and standard error to queue for deterministic prints
    #    \- deterministic -> same order -> process + parent CAN'T concurrently "write" (i.e. point stdout) to terminal
    #           (solution: use _QueueStream  + errq to "transport" process stdout to parent, and parent writes)
    #           ___________        _________        __________        ____________
    #           | process |--------| queue |--------| parent |--------| terminal |
    #           -----------        ----|----        ----------        ------------
    #               /------------------/
    #           ____|____     ___________________     ________________      ________      __________
    #           | queue |   = | stdout / stderr |-----| _QueueStream |------| errq |------| parent |
    #           ---------     -------------------     ----------------      --------      ----------
    #                                                       |- transforms raw text (stdout/stderr) into object (tuple) expected by errq, 
    #                                                          then errq passes to parent
    sys.stdout = _QueueStream(errq, "log", sid)
    sys.stderr = _QueueStream(errq, "err", sid)

    def _showwarn(message, category, filename, lineno, file=None, line=None):
        try:
            errq.put_nowait(("warn", f"[child sid={sid}] {category.__name__}: {message} ({filename}:{lineno})"))
        except Exception:
            pass
    warnings.showwarning = _showwarn

    # (optionally) make Python unbuffered semantics
    try:
        import os
        os.environ.setdefault("PYTHONUNBUFFERED", "1")
    except Exception:
        pass

    def log(msg):
        try:
            #                       /- sys.stderr = _QueueStream(errq, "err", sid)
            # this is passed to the queue and eventually flushed onto the terminal by parent
            errq.put(("log", f"[child sid={sid} pid={os.getpid()}] {msg}"))
        except Exception:
            pass

    # NOW, we start the per-SID operations (memory intensive)
    try:
        fs = s3fs.S3FileSystem(
            anon=True, skip_instance_cache=True,
            use_listings_cache=False, default_cache_type=None
        )

        if debug_flag: log("[_process_one_group] starting")
        group = pd.read_pickle(group_pkl_path)

        if debug_flag: log(f"[_process_one_group] loaded group shape={group.shape}")


        ############################################################ LINK GRS TO RADAR SCANS ############################################################


        if debug_flag: log("[_process_one_group] calling find_radar_scans()")
        linked_radar_df = find_radar_scans(
            group, site_column="radar_site",
            time_column="time",
            level2_base="unidata-nexrad-level2",
            cache_dir="Datasets/nexrad_datasets/level_two_raw",
            product_filter=["reflectivity"],
            time_tolerance_seconds=29,
            keep_in_memory=True,
            debug=False
        )
        if debug_flag: log(f"[_process_one_group] linked_radar_df shape={linked_radar_df.shape}")


        ################################################################## BUILD BBOX ############################################################


        if debug_flag: log("[_process_one_group] calling build_bboxes_for_linked_df()")
        plot_dir = os.path.join("Logs", "plots", f"bbox_sid{sid}_pid{os.getpid()}")
        bboxed_df = build_bboxes_for_linked_df(
            linked_radar_df,
            class_field='reflectivity',
            threshold=20,
            min_size=6000,
            pad_km=5.0,
            grid_res_m=250.0,
            buffer_km=5.0,
            include_nearby_km=3.0,
            debug=False,
            debug_plot_dir=plot_dir,
            debug_plot_limit=0       # Default is two plots per storm, set to zero when building dataset

        )
        if debug_flag: log(f"[_process_one_group] bboxed_df shape={bboxed_df.shape}")


        ################################################################ SAVE FOR TRAINING ############################################################


        if debug_flag: log("[_process_one_group] calling save_df_for_training()")
        save_df_for_training(
            df=bboxed_df,
            base_dir=training_base,    # Now a top-level parameter when main_pipeline is called
            year_override=None,
            radar_site_col="radar_site",
            storm_id_col="storm_id",
            time_col="time",
            debug=False,
            drop_scans_after_save=True # Should ALWAYS be true (free up memory)
        )
        if debug_flag: log("[_process_one_group] done")


        ###############################################################################################################################################


    except Exception:
        tb = traceback.format_exc()
        try:
            errq.put(("exc", f"[child sid={sid}] EXCEPTION\n{tb}"))
        finally:
            sys.exit(1)

    # shut down the child process to clear memory (we do further cleanup in _run_group_in_child)
    finally:
        try: fs.invalidate_cache()
        except Exception: pass
        try: fs.close()
        except Exception: pass
        try: s3fs.S3FileSystem.clear_instance_cache()
        except Exception: pass
        gc.collect()
        try: ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception: pass



def _run_group_in_child(sid, group, radar_info, training_base, debug):
    """
    creates and manages child process for a certain SID, given the supplied arguments 
    """
    p = None
    errq = None
    try:
        with tempfile.TemporaryDirectory() as td:
            pkl_path = os.path.join(td, f"group_{sid}.pkl")
            group.to_pickle(pkl_path, protocol=5)

            # errq used to pass process's stdout and stderr to parent for deterministic writing
            #   - queue specifically designed for communication between processes
            #   - works with python objects, NOT raw text (i.e. NOT stdout or stderr)
            #           \- QueueStream "translates" between the two 
            ctx = mp.get_context("fork")
            errq = ctx.Queue(maxsize=1000)

            # create child process with the given arguments 
            p = ctx.Process(
                target=_process_one_group,
                args=(pkl_path, radar_info, training_base, debug, sid, errq)
            )
            p.start()

            # stream process logs while child runs 
            # here, this is where the parent flushes stdout and stderr recieved from process via errq + QueueStream
            while True:
                try:
                    # we only care about the msg from errq / child process, not the kind (for now)
                    kind, msg = errq.get(timeout=0.5)
                    print(msg, flush=True) # ONE write to terminal from parent
                except _q.Empty:
                    pass
                if not p.is_alive():
                    break

            p.join()

            # drain anything left
            try:
                while True:
                    kind, msg = errq.get_nowait()
                    print(msg, flush=True)
            except _q.Empty:
                pass

            if p.exitcode != 0:
                print(f"[main_pipeline] SID {sid} child exitcode={p.exitcode}", flush=True)
                raise RuntimeError(f"Child for SID {sid} failed with exit code {p.exitcode}")

    except KeyboardInterrupt:
        # Explicit, deterministic cleanup on Ctrl-C
        if p and p.is_alive():
            try:
                p.terminate()             # SIGTERM
                p.join(timeout=5)
                if p.is_alive():
                    os.kill(p.pid, signal.SIGKILL)  # hard kill if needed
                    p.join()
            except Exception:
                pass
        raise  # re-propagate so outer finally runs

    # always free process memory
    finally:
        # tidy queue resources even if interrupted
        if errq is not None:
            try:
                # best effort drain to avoid join_thread hang
                try:
                    while True:
                        kind, msg = errq.get_nowait()
                        print(msg, flush=True)
                except _q.Empty:
                    pass
                errq.close()
                errq.join_thread()
            except Exception:
                pass


##########################################################################################################################################################################
#                                                                        MAIN PIPELINE
##########################################################################################################################################################################


def main_pipeline(debug_flag, year, radar_info, train_rewrite: bool = False, training_base: str = "Datasets/training_datasets/level_two"):
    # Unpack year into start and end datetime objects in the format YYYY, MM, DD
    start = date(year, 1, 1)
    end   = date(year, 12, 31)

    # Pre-index already-finished storms (fast, one-time)
    existing_index = None
    if not train_rewrite:
        existing_index = build_saved_storm_index(training_base, debug=debug_flag)

    # Start plot thread for concurrent plotting
    plot_thread = Thread(target=_plotter_loop, daemon=True)
    plot_thread.start()

    # Wrap everything in a try-finally block to close plotting queuing / threads when finished
    try:
        ################################################################### LOAD DATASETS ###############################################################################
        #                                    |-- NOTE: We don't actually USE the lsr and spc datasets that we load in here.
        #                                    \-- For consistency with the preprint, lets leave them out. Should be easy to hook up later, if needed

        # Load lsr-iasate reports
        lsr_df = load_raw_lsr(
            start=start,
            end  =end,
            debug=False,
            cache_dir="Datasets/surface_obs_datasets/lsr_raw",
            force_refresh=False,  # If true, then reload cached files (set to false / remove param entirely later)
        )
        if debug_flag:
            print(f"\n lsr_df shape: {lsr_df.shape} \n")
            display(lsr_df.head())


        # Load spc reports     NOTE -> remember to actually get the 2017_wind.csv after block comes off
        spc_df = load_raw_spc(
            start=start,
            end  =end,
            debug=False
        )
        if debug_flag:
            print(f"\n spc_df shape: {spc_df.shape} \n")
            display(spc_df.head())

        ################################################################ LOAD GR-S TRACKS  ###############################################################################

        # Load gr-s tracks
        grs_df = load_grs_tracks(
            year=year,
            radar_info=radar_info,
            base_url="https://data-osdf.rda.ucar.edu/ncar/rda/d841006/tracks",
            min_rows=60, # Storm must spend [min_rows] minutes within max_distance_km of a given radar
            max_distance_km=250,
            debug=False,
            timeout=10, # How much seconds before a request is skipped
            save_dir="Datasets/cell_tracks/raw_grs",
            max_gap_hours=6.0 # Highest allowable time-discontinuity for rows in storm_id
        )
        if debug_flag:
            print(f"\n grs_df shape: {grs_df.shape} \n")
            display(grs_df.head(2000))

        # Link gr-s tracks to radar scan
        grouped = grs_df.groupby("storm_id")


        ############################################################ START SID-SPECIFIC OPERATIONS ###############################################################################
        #                                         (now, our "global" processes are done and we can operate on each SID individually)
        #                                                            \- we use processes to avoid memory buildup & practice program isolation
        # Run the child subprocesses
        for sid, group in grouped:
            if debug_flag:
                site_col = "radar_site"
                print(f"[main_pipeline] storm_id={sid} with {len(group)} rows; site(s): {group[site_col].unique().tolist()}")

            # Compute the year for this group (robust)
            try:
                row_year = int(pd.to_datetime(group["time"].iloc[0]).year)
            except Exception:
                row_year = int(year)  # fallback to pipeline's year arg

            # Fast skip if (year, SID) already saved (unless rewrite=True)
            if should_skip_sid(row_year, int(sid), existing_index, rewrite=train_rewrite, debug=False):
                continue

            # If not saved, then start the radar scan and bbox process
            try:
                _run_group_in_child(
                    sid=sid,
                    group=group,
                    radar_info=radar_info,
                    debug=False,
                    training_base=training_base)
            except Exception as e:
                if debug_flag:
                    print(f"[main_pipeline] SID {sid} child failed: {e}")
                continue # Skip and move on



        ########################################################## BUILD DATASET-WIDE PROVENANCE ############################################################################


        # Build manifest and catalog csvs
        year_dir = os.path.join(training_base, str(year))
        build_year_manifest_and_catalog(
            year_dir,
            update_schema_checksums=True,  # set True once to backfill (not used anymore)
            debug=True
        )

    finally:
        # Always close the concurrent plotting thread
        raw_queue.put(None)   # signal “done”
        plot_thread.join()

        # Aggressive cleanup: drop big refs, GC, trim arenas
        aggressive_memory_cleanup(locals())

