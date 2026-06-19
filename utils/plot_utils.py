import os
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

def save_stitched_pvalue_histograms(
    worlds,
    p_by_world,
    basepath,
    filename,
    bins=30,
    density=False,
):
    """Create and save stitched p-value histograms across worlds.

    Generates a 1xN panel figure (one histogram per world), applies consistent
    styling, and saves the figure as both PDF and PNG.

    Parameters
    ----------
    worlds : dict
        Mapping from world name to world metadata. Each world is expected to
        contain an ``"a"`` entry used in the panel title.
    p_by_world : dict
        Mapping from world name to iterable of p-values.
    basepath : str
        Output directory where figures are saved.
    filename : str, optional
        Output filename stem (without extension), by default
        ``"p_values_stitched"``.
    bins : int, optional
        Number of histogram bins, by default 30.
    density : bool, optional
        If ``True``, plot densities instead of counts, by default ``False``.
    
    Returns
    -------
    tuple[str, str]
        Paths to the saved ``.pdf`` and ``.png`` files, respectively.
    """

    os.makedirs(basepath, exist_ok=True)

    world_names = list(worlds.keys())
    n = len(world_names)

    # Scalable width; height optimized for strip layout.
    fig, axs = plt.subplots(1, n, figsize=(2.6 * n, 2.6), dpi=400)

    if n == 1:
        axs = [axs]

    for i, w in enumerate(world_names):
        ax = axs[i]
        p_scores = p_by_world[w]

        ax.hist(
            p_scores,
            bins=bins,
            color="black",
            edgecolor="black",
            linewidth=0.4,
            density=density,
        )

        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_xlabel("p", fontsize=20)

        if i == 0:
            ax.set_ylabel("Density" if density else "Count", fontsize=20)
        else:
            ax.set_ylabel("")

        ax.tick_params(axis="both", labelsize=16, width=1.5, length=5)

        # Clean styling for small panels.
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.5)
        ax.spines["bottom"].set_linewidth(1.5)
        ax.grid(False)

        # Panel label: "W ($a$=..)" where W is world name.
        a_value = worlds[w]["a"]
        text_str = f"{w} ($a$={a_value:.1f})"

        ax.text(
            0.5,
            1.0,  # slightly above axes
            text_str,
            transform=ax.transAxes,
            fontsize=18,
            fontweight="bold",
            va="bottom",
            ha="center",
        )

    fig.subplots_adjust(left=0.06, right=0.995, bottom=0.22, top=0.98, wspace=0.35)

    png_path = os.path.join(basepath, f"{filename}.png")
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)


def plot_benchmark_results(
    worlds: dict,
    all_series: dict,
    n_raters_list: tuple[int],
    save_path: str,
):  
    worlds_in_order = list(worlds.keys())
    x = np.arange(len(worlds_in_order))

    plt.rcParams.update({
      "figure.dpi": 200,
      "savefig.dpi": 600,
      "figure.facecolor": "white",
      "axes.facecolor": "white",
      "font.family": "DejaVu Sans",
      "font.size": 12,
      "axes.titlesize": 11,
      "axes.titleweight": "bold",
      "axes.labelsize": 12,
      "xtick.labelsize": 12,
      "ytick.labelsize": 12,
      "axes.linewidth": 0.8,
      "axes.grid": True,
      "grid.alpha": 0.18,
      "grid.linewidth": 0.6,
      "legend.frameon": False,
    })

    fig, (ax_abs, ax_case, ax_rater, ax_adj) = plt.subplots(
      1, 4, figsize=(7.2 * 2.0, 2.8), sharex=True
    )

    prop_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if len(prop_cycle) == 0:
      prop_cycle = ["C0", "C1", "C2", "C3", "C4", "C5"]
    colors_for_nr = {nr: prop_cycle[i % len(prop_cycle)] for i, nr in enumerate(n_raters_list)}

    linestyle_for_metric = {
      "case_ece": "-",
      "rater_ece": "-",
      "adjusted_ece": "-",
    }

    marker_cycle = ["o", "s", "D", "^", "v", "P", "X", "*", "<", ">"]
    markers_for_nr = {nr: marker_cycle[i % len(marker_cycle)] for i, nr in enumerate(n_raters_list)}

    # (a) TVD trend 
    for nr in n_raters_list:
      ax_abs.plot(
        x, all_series[nr]["tvd"],
        linewidth=2.2,
        color=colors_for_nr[nr],
        alpha=0.95,
        zorder=3,
        label=f"n={nr}",
        marker=markers_for_nr[nr],
        markersize=4.0
      )

    ax_abs.set_ylabel("TVD")
    ax_abs.set_xlabel("World")
    ax_abs.set_xticks(x)
    ax_abs.set_xticklabels(worlds_in_order)
    ax_abs.set_ylim(0.0, 0.25)

    # Right panels: raw metric values
    metric_axes = {
      "case_ece": ax_case,
      "rater_ece": ax_rater,
      "adjusted_ece": ax_adj,
    }
    metric_ylabel = {
      "case_ece": "CaseECE",
      "rater_ece": "MR-ECE",
      "adjusted_ece": "AdjustedECE",
    }

    for metric, axm in metric_axes.items():
      for nr in n_raters_list:
        axm.plot(
            x,
            all_series[nr][metric],
            linestyle=linestyle_for_metric[metric],
            linewidth=1.9,
            color=colors_for_nr[nr],
            alpha=0.95,
            zorder=3,
            marker=markers_for_nr[nr],
            markersize=4.0
        )

      axm.set_ylabel(metric_ylabel[metric])
      axm.set_xlabel("World")
      axm.set_xticks(x)
      axm.set_xticklabels(worlds_in_order)
      axm.set_ylim(0.0, 0.25)

      lg = axm.get_legend()
      if lg is not None:
        lg.remove()

    for ax in (ax_abs, ax_case, ax_rater, ax_adj):
      ax.spines["top"].set_visible(False)
      ax.spines["right"].set_visible(False)
      ax.tick_params(axis="both", which="major", length=4, width=0.8)
      ax.margins(x=0.03)

    ax_abs.text(0.5, 0.95, "(a)", transform=ax_abs.transAxes,
            fontsize=12, fontweight="bold", va="bottom")
    ax_case.text(0.5, 0.95, "(b)", transform=ax_case.transAxes,
             fontsize=12, fontweight="bold", va="bottom")
    ax_rater.text(0.5, 0.95, "(c)", transform=ax_rater.transAxes,
            fontsize=12, fontweight="bold", va="bottom")
    ax_adj.text(0.5, 0.95, "(d)", transform=ax_adj.transAxes,
            fontsize=12, fontweight="bold", va="bottom")

    legend_handles = [
      mlines.Line2D([], [], linestyle="none", label="Number of Raters:")
    ]
    legend_handles += [
      mlines.Line2D(
        [], [],
        color=colors_for_nr[nr],
        marker=markers_for_nr[nr],
        linestyle='-',
        linewidth=2.2,
        markersize=5,
        label=f"n={nr}"
      )
      for nr in n_raters_list
    ]

    fig.legend(
      handles=legend_handles,
      loc="upper center",
      bbox_to_anchor=(0.5, 1.12),
      ncol=len(n_raters_list) + 1,
      frameon=True,
      fancybox=False,
      edgecolor="black",
      fontsize=12,
    )

    fig.tight_layout(w_pad=1.6)

    if save_path is not None:
      fig.savefig(save_path, bbox_inches="tight")
      
    return fig, (ax_abs, ax_case, ax_rater, ax_adj)