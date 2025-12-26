def _save_plot(fig, plot_dir, func_name, stub, debug):
    try:
        import matplotlib.pyplot as plt
        if plot_dir:
            subdir = os.path.join(plot_dir, func_name)
            os.makedirs(subdir, exist_ok=True)
            name = stub or func_name
            out_path = os.path.join(subdir, f"{name}.png")
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            if debug:
                print(f"[{func_name}] saved debug plot -> {out_path}")
            plt.close(fig)
    except Exception as e:
        if debug:
            print(f"[{func_name}] saving/plotting failed: {e}")


def _plotter_loop():
    """
    NOTE -> DEPRECATED, but keep for backwards compat until refactoring is done.
    Decouple figure construction from main thread to eliminate blocking locks while plotting figures.
    """
    print("[_plotter_loop] Queue monitoring started")
    while True:
        item = raw_queue.get()
        if item is None:                # sentinel to shut down
            break

        # Unpack the item in the queue
        *payload, plot_type = item
        print(f"[_plotter_loop] Plotting figure for plot_type: {plot_type}")

        # Have the last "argument" in a queue's item determine the plot type
        if plot_type == 'level_three':
            T, R, echo_clean = payload
            fig = plt.figure()
            ax  = fig.add_subplot(projection='polar')
            ax.set_theta_zero_location('N')
            ax.set_theta_direction(-1)
            pcm = ax.pcolormesh(T, R, echo_clean, shading='flat')
            plt.colorbar(pcm, label='Echo Classification Code', ax=ax)
            ax.set_title('Radar Echo Classification (Polar View)')

            # Add processed fig to fig_queue
            fig_queue.put(fig)

        # EXAMPLE (add more elif blocks for new plot types)
        elif plot_type == 'contour':
            x, y, z = payload
            fig, ax = plt.subplots()
            cs = ax.contour(x, y, z)
            ax.clabel(cs)
            # And again, fig_queue.put(fig)