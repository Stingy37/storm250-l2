# Handle environment for running pipeline
if __name__ == "__main__":
    try:
        # Set spawn method for child processes 
        try:
            mp.set_start_method("fork")
        except RuntimeError:
            pass

        # Run main pipeline
        with cProfile.Profile() as pr:
            main_pipeline(
                debug_flag=False,        # NOTE -> Individual components have separate debug flags
                year=2017,               # Unpack into start and end times (datetime objects) within main_pipeline
                radar_info=radar_info,
                train_rewrite=False,     # If rewrite is true, then ignore what is currently in training dataset and rewrite entire dataset
                training_base="Datasets/training_datasets/level_two"
                )
    finally:
        # Print time spent in slowest functions
        stats = pstats.Stats(pr)
        stats.sort_stats('cumtime').print_stats(30)

        del stats, pr
        import gc, ctypes
        gc.collect()
        ctypes.CDLL("libc.so.6").malloc_trim(0)