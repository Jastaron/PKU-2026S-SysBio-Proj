from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

from .core import BLACK, EMPTY, Frame, RED, SKIN, YELLOW, pigment_display_size


def apply_plot_style() -> None:
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


def draw_frame(ax: plt.Axes, frame: Frame, params: dict, title: str | None = None, side_label: str | None = None) -> None:
    state_grid = compose_state_grid(frame, params)
    cmap, norm = build_colormap()
    ax.imshow(state_grid, cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.5, state_grid.shape[1] - 0.5)
    ax.set_ylim(state_grid.shape[0] - 0.5, -0.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title:
        ax.set_title(title, pad=7)
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


def draw_timeline_slices(axes, timeline: list[Frame], steps: list[int], params: dict, row_label: str | None = None) -> None:
    axes = list(np.ravel(axes))
    for idx, (ax, step) in enumerate(zip(axes, steps)):
        frame = timeline[step]
        title = (
            f"step {step}\n"
            f"N={frame.pigment_count} | Y={frame.yellow_count} R={frame.red_count} B={frame.black_count}"
        )
        draw_frame(ax, frame, params, title=title, side_label=row_label if idx == 0 else None)


def draw_nnd_distribution(
    ax: plt.Axes,
    nnd_map: dict[str, np.ndarray],
    labels: dict[str, str],
    colors: dict[str, str],
    *,
    bins: np.ndarray | None = None,
    linewidth: float = 2.2,
    title: str = "NND Distribution",
    legend_loc: str = "upper left",
) -> None:
    all_vals = np.concatenate([vals for vals in nnd_map.values() if len(vals) > 0])
    if bins is None:
        bins = np.linspace(0.0, max(8.0, np.percentile(all_vals, 99)), 26)
    for model, values in nnd_map.items():
        ax.hist(
            values,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=linewidth,
            color=colors[model],
            label=labels[model],
        )
    ax.set_xlabel("Nearest-neighbour distance")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend(frameon=False, loc=legend_loc)


def draw_cv_bar(
    ax: plt.Axes,
    summary_df,
    order: list[str],
    labels: list[str],
    colors: list[str],
    *,
    value_col: str = "CV_NND",
    ylabel: str = "CV of nearest-neighbour distance",
    title: str = "CV_NND Comparison",
    value_label_fmt: str | None = None,
) -> None:
    values = summary_df.set_index("model").loc[order, value_col].to_numpy()
    ax.bar(labels, values, color=colors, width=0.65)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fmt = value_label_fmt or ("{:.3f}" if value_col != "N" else "{:.0f}")
    add_bar_value_labels(ax, values, fmt=fmt)


def draw_color_composition(
    ax: plt.Axes,
    metrics_df,
    *,
    fraction: bool = False,
    colors: dict[str, str] | None = None,
    linewidths: dict[str, float] | None = None,
    title: str | None = None,
) -> None:
    if colors is None:
        colors = {"yellow": "#ffcc33", "red": "#c94b23", "black": "#171411"}
    if linewidths is None:
        linewidths = {"yellow": 2.5, "red": 2.5, "black": 2.8}
    if fraction:
        data = metrics_df[["yellow_count", "red_count", "black_count"]].div(
            metrics_df["pigment_count"].replace(0, np.nan),
            axis=0,
        ).fillna(0.0)
        ylabel = "Fraction"
        default_title = "Colour Fractions Over Time"
        ax.set_ylim(0.0, 1.0)
    else:
        data = metrics_df[["yellow_count", "red_count", "black_count"]]
        ylabel = "Chromatophore count"
        default_title = "Colour Composition Over Time"
    ax.plot(metrics_df["step"], data["yellow_count"], color=colors["yellow"], linewidth=linewidths["yellow"], label="yellow")
    ax.plot(metrics_df["step"], data["red_count"], color=colors["red"], linewidth=linewidths["red"], label="red")
    ax.plot(metrics_df["step"], data["black_count"], color=colors["black"], linewidth=linewidths["black"], label="black")
    ax.set_xlabel("Step")
    ax.set_ylabel(ylabel)
    ax.set_title(title or default_title, pad=18)
    ax.margins(y=0.08 if not fraction else 0.05)


def draw_parameter_heatmap(ax: plt.Axes, scan_df, value_col: str, cmap: str = "viridis", title: str | None = None) -> None:
    x_vals = sorted(scan_df["absolute_min_distance"].unique())
    y_vals = sorted(scan_df["target_gap_to_black"].unique())
    grid = (
        scan_df.groupby(["target_gap_to_black", "absolute_min_distance"])[value_col]
        .mean()
        .unstack()
        .reindex(index=y_vals, columns=x_vals)
    )
    image = ax.imshow(grid.to_numpy(), origin="lower", aspect="auto", cmap=cmap)
    ax.set_xlabel("absolute_min_distance")
    ax.set_ylabel("target_gap_to_black")
    ax.set_xticks(range(len(x_vals)), [f"{value:.1f}" for value in x_vals])
    ax.set_yticks(range(len(y_vals)), [f"{value:.1f}" for value in y_vals])
    if title:
        ax.set_title(title)
    return image


def draw_pair_correlation(
    ax: plt.Axes,
    curve_map: dict[str, tuple[np.ndarray, np.ndarray]],
    labels: dict[str, str],
    colors: dict[str, str],
    *,
    linewidth: float = 2.4,
    title: str = "Pair-Correlation-Like Curve",
    shade_until: float = 3.0,
) -> None:
    for model, (centers, density) in curve_map.items():
        ax.plot(centers, density, color=colors[model], linewidth=linewidth, label=labels[model])
    ax.axvspan(0.0, shade_until, color="#d9d9d9", alpha=0.25)
    ax.set_xlabel("Pair distance")
    ax.set_ylabel("Pair-count density")
    ax.set_title(title)
    ax.legend(frameon=False)


def draw_fraction_stack(ax: plt.Axes, summary_df, order: list[str], labels: list[str]) -> None:
    summary = summary_df.set_index("model").loc[order]
    bottoms = np.zeros(len(order))
    fractions = [
        ("yellow_fraction", "#ffcc33", "yellow"),
        ("red_fraction", "#c94b23", "red"),
        ("black_fraction", "#171411", "black"),
    ]
    for col, color, label in fractions:
        ax.bar(labels, summary[col], bottom=bottoms, color=color, width=0.68, label=label)
        bottoms += summary[col].to_numpy()
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Colour fractions")
    ax.set_ylabel("Fraction")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(frameon=False, loc="upper left")


def add_bar_value_labels(ax: plt.Axes, values, fmt: str = "{:.3f}", fontsize: int = 11) -> None:
    values = list(values)
    if not values:
        return
    current_top = ax.get_ylim()[1]
    ymax = max(values)
    min_y = min(0.0, ax.get_ylim()[0])
    span = max(current_top - min_y, 1e-6)
    ax.set_ylim(min_y, max(current_top, ymax + 0.16 * span))
    low_threshold = ax.get_ylim()[1] * 0.18
    for idx, value in enumerate(values):
        if value > low_threshold:
            y = value + 0.03 * ax.get_ylim()[1]
            va = "bottom"
            color = "black"
        else:
            y = value + 0.04 * ax.get_ylim()[1]
            va = "bottom"
            color = "black"
        ax.text(idx, y, fmt.format(value), ha="center", va=va, fontsize=fontsize, color=color)


def build_colormap() -> tuple[ListedColormap, BoundaryNorm]:
    cmap = ListedColormap(
        [
            "#f7f4ee",
            "#e7e3cc",
            "#ffcc33",
            "#c94b23",
            "#171411",
        ]
    )
    norm = BoundaryNorm(np.arange(-0.5, 5.5, 1), cmap.N)
    return cmap, norm


def compose_state_grid(frame: Frame, params: dict) -> np.ndarray:
    skin = frame.skin
    state_grid = skin.copy()
    score_grid = np.full(skin.shape, -np.inf, dtype=float)
    priority = {YELLOW: 1.0, RED: 2.0, BLACK: 3.0}

    for pigment in frame.pigments:
        stage = pigment.stage(params)
        rows, cols = rasterize_pigment_cells(pigment, params, skin.shape[0])
        dist2 = (rows - pigment.pos[0]) ** 2 + (cols - pigment.pos[1]) ** 2
        local_score = priority[stage] * 100.0 - dist2 - 0.02 * pigment.age
        for rr, cc, score in zip(rows, cols, local_score):
            if skin[rr, cc] != SKIN:
                continue
            if score > score_grid[rr, cc]:
                state_grid[rr, cc] = stage
                score_grid[rr, cc] = score
    return state_grid


def rasterize_pigment_cells(pigment, params: dict, grid_size: int) -> tuple[np.ndarray, np.ndarray]:
    major, minor = pigment_display_size(pigment, params)
    half_h = max(1, int(math.ceil(major * 0.75)))
    half_w = max(1, int(math.ceil(major * 0.75)))
    rr_center = pigment.pos[0]
    cc_center = pigment.pos[1]
    angle = math.radians(pigment.angle)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    rows = []
    cols = []
    for rr in range(int(math.floor(rr_center - half_h)), int(math.ceil(rr_center + half_h)) + 1):
        if rr < 0 or rr >= grid_size:
            continue
        for cc in range(int(math.floor(cc_center - half_w)), int(math.ceil(cc_center + half_w)) + 1):
            if cc < 0 or cc >= grid_size:
                continue
            dy = rr - rr_center
            dx = cc - cc_center
            x_rot = cos_a * dx + sin_a * dy
            y_rot = -sin_a * dx + cos_a * dy
            value = (x_rot / max(major * 0.5, 1e-6)) ** 2 + (y_rot / max(minor * 0.5, 1e-6)) ** 2
            if value <= 1.0:
                rows.append(rr)
                cols.append(cc)
    if not rows:
        rows = [int(np.clip(round(rr_center), 0, grid_size - 1))]
        cols = [int(np.clip(round(cc_center), 0, grid_size - 1))]
    return np.asarray(rows, dtype=int), np.asarray(cols, dtype=int)
