import os
from pathlib import Path

import pandas as pd
import pytest

#                         |- so when pytest "calls" the test, n = 3
#                         |                                   \- where n is the number of figures saved 
@pytest.mark.parametrize("n", [3])
def test_plot_observations_on_fig_writes_pngs(tmp_path, n: int):
    # Force a headless backend BEFORE importing pyplot (needed since EC2 has no GUI)
    os.environ.setdefault("MPLBACKEND", "Agg")

    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    plt = pytest.importorskip("matplotlib.pyplot")

    from storm250.linking import _plot_observations_on_fig

    plot_dir = tmp_path / "overlay_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    expected_files = []

    for i in range(n):
        fig = plt.figure()
        ax = fig.add_subplot(111)

        # give the axis some initial bounds so autoscale has something sensible
        ax.set_xlim(-130, -60)
        ax.set_ylim(20, 55)

        obs_df = pd.DataFrame(
            {
                "station_lon": [-100.0 + 0.1 * i, -99.98 + 0.1 * i],
                "station_lat": [35.0 + 0.1 * i, 35.02 + 0.1 * i],
            }
        )

        # include a slightly "unsafe" stem to exercise sanitization
        file_stem = f"scan{i}/cell:{i}"

        _plot_observations_on_fig(
            fig,
            obs_df,
            lon_col="station_lon",
            lat_col="station_lat",
            save_dir=plot_dir,
            file_stem=file_stem,
        )

        # linking.py sanitizes stem -> underscores
        safe_stem = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in file_stem)
        expected_files.append(plot_dir / f"{safe_stem}.png")

        plt.close(fig)

    # Assert all expected PNGs exist and are non-empty
    for p in expected_files:
        assert p.exists(), f"Expected plot file not found: {p}"
        assert p.stat().st_size > 0, f"Plot file is empty: {p}"

    # Also assert the directory contains exactly n pngs
    pngs = list(plot_dir.glob("*.png"))
    assert len(pngs) == n
