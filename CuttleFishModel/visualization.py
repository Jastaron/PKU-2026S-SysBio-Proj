from __future__ import annotations

import matplotlib.pyplot as plt

from .model import build_render_colormap, compose_ca_render_grid


def apply_report_style() -> None:
    """Apply a single plotting style across report figures."""
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 11,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
        }
    )


def render_ca_on_axis(ax: plt.Axes, frame: dict, params: dict, side_label: str | None = None) -> None:
    """Render one CA state using the same lattice style as the GIF."""
    state_grid, _ = compose_ca_render_grid(frame, params)
    cmap, norm = build_render_colormap()
    ax.imshow(state_grid, cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.5, state_grid.shape[1] - 0.5)
    ax.set_ylim(state_grid.shape[0] - 0.5, -0.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if side_label:
        ax.text(
            -0.10,
            0.5,
            side_label,
            transform=ax.transAxes,
            rotation=90,
            va="center",
            ha="center",
            fontsize=13,
            fontweight="bold",
        )


def add_bar_value_labels(ax: plt.Axes, values, *, fmt: str = "{:.3f}", fontsize: int = 11) -> None:
    """Place bar labels with data-dependent offsets so they do not collide with bars or frame."""
    values = list(values)
    if not values:
        return
    ymax = max(values)
    ymin, ymax_lim = ax.get_ylim()
    span = max(ymax_lim - ymin, 1e-6)
    ax.set_ylim(ymin, max(ymax_lim, ymax + 0.12 * span))
    for idx, value in enumerate(values):
        offset = 0.02 * max(ax.get_ylim()[1] - ax.get_ylim()[0], 1.0)
        ax.text(idx, value + offset, fmt.format(value), ha="center", va="bottom", fontsize=fontsize)


def place_legend_above(ax: plt.Axes, *, ncol: int = 3) -> None:
    """Place the legend above the plotting area to avoid line overlap."""
    ax.legend(
        frameon=False,
        ncol=ncol,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.02),
        borderaxespad=0.0,
    )
