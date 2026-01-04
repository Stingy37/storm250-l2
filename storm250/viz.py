import os 

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

