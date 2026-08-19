"""Plot the ((6,2,3)) C4 manifold-optimization benchmark.

This script only visualizes data already stored in ``623_group_range01.pkl``.
It does not rerun the optimization.  The two panels show

1. representative selected-run optimization trajectories; and
2. the optimized total loss returned after up to 100 random initializations
   at each of 101 target squared signature norms.  The restart search stops
   early when its current best preliminary loss is at most 1e-5, after which
   that successful candidate is refined.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

# Keep Matplotlib's cache outside the source tree when this script is run by
# an automated agent or in another read-only-home environment.
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/codex-matplotlib-cache")

import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter
from zzz233 import from_pickle_wrapper


HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "623_group_range01.pkl"
OUTPUT_STEM = HERE / "c4_benchmark"

LOWER_BOUNDARY = 2 / 3
UPPER_BOUNDARY = 1.0
SUCCESS_THRESHOLD = 1e-7


def load_data() -> tuple[list[float], list[np.ndarray], np.ndarray, np.ndarray]:
    """Load the stored convergence traces and C4 target scan."""
    from_pickle = from_pickle_wrapper(str(DATA_PATH))

    convergence = from_pickle("convergence")
    trace_targets = list(convergence["lambda2_list"])
    traces = [np.asarray(values, dtype=float) for values in convergence["z0"]]

    scan = from_pickle("c4_range")
    scan_targets = np.asarray(scan["lambda2_list"], dtype=float)
    scan_losses = np.asarray([result.fun for result in scan["z0"]], dtype=float)

    if len(trace_targets) != len(traces):
        raise ValueError("The number of trace targets does not match the traces.")
    if scan_targets.shape != scan_losses.shape:
        raise ValueError("The C4 scan targets and optimized losses do not match.")
    if scan_targets.size != 101:
        raise ValueError(f"Expected 101 C4 scan targets, found {scan_targets.size}.")
    if np.any(scan_losses <= 0) or not np.all(np.isfinite(scan_losses)):
        raise ValueError("A logarithmic loss axis requires finite positive losses.")

    return trace_targets, traces, scan_targets, scan_losses


def make_figure() -> None:
    """Create PDF, SVG, and high-resolution PNG versions of the benchmark."""
    trace_targets, traces, scan_targets, scan_losses = load_data()

    # 183 mm is the Nature Portfolio/npj double-column width.  Explicit font
    # and line settings make the exports independent of the user's rc file.
    figure_style = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "axes.linewidth": 0.8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,
        "legend.fontsize": 6.8,
        "lines.linewidth": 1.2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "mathtext.fontset": "dejavusans",
    }

    # Restrained blue/orange palette plus neutrals.  Dash patterns and marker
    # shapes provide redundant distinctions for grayscale reproduction.
    trace_styles = [
        dict(color="#767676", linestyle=(0, (1.2, 1.4)), marker="s"),
        dict(color="#3F78A8", linestyle=(0, (4.0, 1.8)), marker="^"),
        dict(color="#183B56", linestyle="-", marker="o", linewidth=1.55),
        dict(color="#B85C24", linestyle=(0, (5.0, 1.5, 1.2, 1.5)), marker="D"),
    ]
    target_labels = ["0.65", "0.66", r"$2/3$", "0.67"]

    with plt.rc_context(figure_style):
        fig, (ax_trace, ax_scan) = plt.subplots(
            1,
            2,
            figsize=(183 / 25.4, 82 / 25.4),
            sharey=True,
            gridspec_kw={"width_ratios": [1.04, 1.0]},
        )

        for values, label, style in zip(
            traces, target_labels, trace_styles, strict=True
        ):
            iterations = np.arange(1, len(values) + 1)
            # Sparse markers make overlapping line styles legible in print.
            markevery = np.unique(
                np.geomspace(1, len(values), num=10).astype(int) - 1
            )
            ax_trace.plot(
                iterations,
                values,
                label=label,
                markersize=2.5,
                markerfacecolor="white",
                markeredgewidth=0.65,
                markevery=markevery,
                **style,
            )

        ax_trace.set_xscale("log")
        ax_trace.set_yscale("log")
        ax_trace.set_xlim(1, 3000)
        ax_trace.set_ylim(3e-17, 3e1)
        ax_trace.set_xlabel("Optimization iteration")
        ax_trace.set_ylabel(r"Total loss $\mathcal{L}_{\mathrm{total}}$")
        # ax_trace.set_title(
        #     "Representative selected-run optimization trajectories",
        #     loc="left",
        #     pad=7,
        # )
        ax_trace.legend(
            title=r"Target $[\lambda^*_{\mathrm{target}}]^2$",
            loc="lower left",
            ncol=2,
            frameon=False,
            handlelength=2.7,
            handletextpad=0.55,
            columnspacing=0.9,
            labelspacing=0.35,
            borderaxespad=0.35,
            title_fontsize=6.8,
        )

        # The shaded region is the analytically realized C4 interval.  Its
        # edges are also drawn explicitly so both boundaries survive grayscale.
        ax_scan.axvspan(
            LOWER_BOUNDARY,
            UPPER_BOUNDARY,
            color="#DCE7EF",
            alpha=0.65,
            linewidth=0,
            zorder=0,
        )
        for boundary in (LOWER_BOUNDARY, UPPER_BOUNDARY):
            ax_scan.axvline(
                boundary,
                color="#4D4D4D",
                linestyle=(0, (2.0, 1.5)),
                linewidth=1.0,
                zorder=2,
            )

        ax_scan.plot(
            scan_targets,
            scan_losses,
            color="#183B56",
            marker="o",
            markersize=2.25,
            markerfacecolor="white",
            markeredgewidth=0.65,
            linewidth=1.05,
            zorder=3,
        )
        ax_scan.axhline(
            SUCCESS_THRESHOLD,
            color="#B85C24",
            linestyle=(0, (5.0, 2.0)),
            linewidth=1.15,
            zorder=2,
        )
        ax_scan.text(
            0.938,
            SUCCESS_THRESHOLD * 0.07,
            r"Success threshold $10^{-7}$",
            color="#7A3A17",
            fontsize=6.8,
            ha="right",
            va="bottom",
        )
        ax_scan.text(
            (LOWER_BOUNDARY + UPPER_BOUNDARY) / 2,
            8e-2,
            r"Analytical interval $[2/3,\,1]$",
            color="#333333",
            fontsize=6.8,
            ha="center",
            va="top",
        )
        # ax_scan.text(
        #     LOWER_BOUNDARY,
        #     3.5e-2,
        #     r"$2/3$",
        #     color="#333333",
        #     fontsize=6.8,
        #     ha="center",
        #     va="top",
        # )
        # ax_scan.text(
        #     UPPER_BOUNDARY,
        #     3.5e-2,
        #     r"$1$",
        #     color="#333333",
        #     fontsize=6.8,
        #     ha="center",
        #     va="top",
        # )
        ax_scan.set_xlim(0.5, 1.1)
        ax_scan.set_xlabel(r"Target squared signature norm $[\lambda^*_{\mathrm{target}}]^2$")
        # ax_scan.set_title(
        #     "Target scan (up to 100 random initializations)", loc="left", pad=7
        # )

        for panel_label, ax in zip(("a", "b"), (ax_trace, ax_scan), strict=True):
            ax.text(
                -0.13,
                1.08,
                panel_label,
                transform=ax.transAxes,
                fontsize=8.5,
                fontweight="bold",
                ha="left",
                va="top",
            )
            ax.grid(
                axis="y",
                which="major",
                color="#D9D9D9",
                linewidth=0.65,
                zorder=-1,
            )
            ax.tick_params(
                which="both", direction="out", top=False, right=False, length=3
            )
            ax.tick_params(which="minor", length=1.7)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.yaxis.set_major_locator(LogLocator(base=10, numticks=10))
            ax.yaxis.set_minor_formatter(NullFormatter())

        fig.subplots_adjust(
            left=0.090,
            right=0.985,
            bottom=0.175,
            top=0.865,
            wspace=0.17,
        )
        fig.savefig(OUTPUT_STEM.with_suffix(".pdf"), facecolor="white")
        fig.savefig(OUTPUT_STEM.with_suffix(".svg"), facecolor="white")
        fig.savefig(
            OUTPUT_STEM.with_suffix(".png"), dpi=600, facecolor="white"
        )
        plt.close(fig)


if __name__ == "__main__":
    make_figure()
