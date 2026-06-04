from __future__ import annotations

from pathlib import Path
import math
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from SysBio_color_growth_CA_intercalated import (
    BLACK,
    PARAMS,
    RED,
    SKIN,
    YELLOW,
    build_render_colormap,
    center_birth_gate,
    choose_new_births,
    choose_final_step,
    compose_ca_render_grid,
    compute_boundary_positions,
    compute_counts,
    grow_skin,
    initialize_pigments,
    initialize_skin,
    make_pigment,
    move_with_skin_growth,
    nearest_neighbor_distances,
    pairwise_distances,
    pigment_stage,
    project_inside_skin,
    relax_overlaps,
    simulate,
    summarize_nnd,
)


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


def choose_random_births(
    skin: np.ndarray,
    skin_age: np.ndarray,
    pigments: list[dict],
    params: dict,
    rng: np.random.Generator,
) -> list[dict]:
    """Random-development ablation: keep tissue growth and boundary bias, remove local spacing logic."""
    eligible = (skin == SKIN) & (skin_age >= params["min_skin_age_for_diff"])
    candidate_positions = np.argwhere(eligible)
    if len(candidate_positions) == 0:
        return []

    sample_size = min(params["candidate_sample_size"], len(candidate_positions))
    sampled_idx = rng.choice(len(candidate_positions), size=sample_size, replace=False)
    sampled = candidate_positions[sampled_idx].astype(float)

    boundary_positions = compute_boundary_positions(skin)
    if len(boundary_positions) > 0:
        d_boundary = pairwise_distances(sampled, boundary_positions).min(axis=1)
    else:
        d_boundary = np.full(len(sampled), np.inf)

    boundary_gate = 1.0 / (
        1.0 + np.exp(-(d_boundary - params["min_boundary_distance"]) / params["boundary_softness"])
    )
    center_gate = center_birth_gate(sampled, skin, params)
    birth_prob = np.clip(
        params["base_birth_rate"] * boundary_gate * center_gate,
        0.0,
        params["birth_prob_cap"],
    )
    birth_prob[d_boundary < params["min_boundary_distance"] * 0.55] = 0.0

    target_pigment_count = np.count_nonzero(eligible) / params["target_area_per_pigment"]
    pigment_deficit = max(0.0, target_pigment_count - len(pigments))
    dynamic_quota = int(
        np.clip(
            np.ceil(
                params["birth_quota_base"]
                + params["birth_quota_skin_scale"] * np.count_nonzero(eligible)
                + params["birth_quota_deficit_scale"] * pigment_deficit
            ),
            1,
            params["max_new_pigments_per_step"],
        )
    )

    order = rng.permutation(len(sampled))
    chosen: list[dict] = []
    for idx in order:
        if len(chosen) >= dynamic_quota:
            break
        if rng.random() >= birth_prob[idx]:
            continue
        chosen.append(make_pigment(sampled[idx].copy(), params, rng, age=0))
    return chosen


def simulate_random_development(params: dict) -> tuple[list[dict], int]:
    """Run the random-development ablation with the same tissue growth and colour maturation."""
    rng_init = np.random.default_rng(params["seed"] + 101)
    rng_growth = np.random.default_rng(params["seed"] + 102)
    rng_birth = np.random.default_rng(params["seed"] + 103)

    skin, skin_age = initialize_skin(params)
    pigments = initialize_pigments(skin, params, rng_init)
    center = np.array([(params["grid_size"] - 1) / 2.0, (params["grid_size"] - 1) / 2.0], dtype=float)
    previous_radius = 0.5 * (params["initial_radius_x"] + params["initial_radius_y"])
    timeline: list[dict] = []

    for step in range(params["n_steps"]):
        if step > 0:
            occupied = skin == SKIN
            skin_age[occupied] += 1

            _, current_radius = grow_skin(skin, skin_age, params, rng_growth, step)
            move_with_skin_growth(pigments, previous_radius, current_radius, params, center)
            project_inside_skin(pigments, skin, center)
            pigments.extend(choose_random_births(skin, skin_age, pigments, params, rng_birth))
            relax_overlaps(pigments, params, center)
            project_inside_skin(pigments, skin, center)
            for pigment in pigments:
                pigment["age"] += 1
            previous_radius = current_radius

        points = get_positions(pigments)
        nnd_mean, nnd_std, nnd_cv = summarize_nnd(points)
        counts = compute_counts(pigments, params)
        timeline.append(
            {
                "step": step,
                "skin": skin.copy(),
                "pigments": [
                    {
                        "pos": pigment["pos"].copy(),
                        "age": pigment["age"],
                        "base_major": pigment["base_major"],
                        "base_minor": pigment["base_minor"],
                        "angle": pigment["angle"],
                        "tone": pigment["tone"],
                    }
                    for pigment in pigments
                ],
                "skin_area": int(np.count_nonzero(skin == SKIN)),
                "pigment_count": int(len(pigments)),
                "yellow_count": counts["yellow"],
                "red_count": counts["red"],
                "black_count": counts["black"],
                "nnd_mean": nnd_mean,
                "nnd_std": nnd_std,
                "nnd_cv": nnd_cv,
            }
        )

    final_step = choose_final_step(timeline, params)
    return timeline, final_step


def simulate_final_self_frame(params: dict) -> dict:
    """Fast end-state simulation for parameter scans without storing a full timeline."""
    rng_init = np.random.default_rng(params["seed"] + 1)
    rng_growth = np.random.default_rng(params["seed"] + 2)
    rng_birth = np.random.default_rng(params["seed"] + 3)

    skin, skin_age = initialize_skin(params)
    pigments = initialize_pigments(skin, params, rng_init)
    center = np.array([(params["grid_size"] - 1) / 2.0, (params["grid_size"] - 1) / 2.0], dtype=float)
    previous_radius = 0.5 * (params["initial_radius_x"] + params["initial_radius_y"])

    for step in range(params["n_steps"]):
        if step > 0:
            occupied = skin == SKIN
            skin_age[occupied] += 1
            _, current_radius = grow_skin(skin, skin_age, params, rng_growth, step)
            move_with_skin_growth(pigments, previous_radius, current_radius, params, center)
            project_inside_skin(pigments, skin, center)
            pigments.extend(choose_new_births(skin, skin_age, pigments, params, rng_birth))
            relax_overlaps(pigments, params, center)
            project_inside_skin(pigments, skin, center)
            for pigment in pigments:
                pigment["age"] += 1
            previous_radius = current_radius

    points = get_positions(pigments)
    nnd_mean, nnd_std, nnd_cv = summarize_nnd(points)
    counts = compute_counts(pigments, params)
    return {
        "step": params["n_steps"] - 1,
        "skin_area": int(np.count_nonzero(skin == SKIN)),
        "pigment_count": int(len(pigments)),
        "yellow_count": counts["yellow"],
        "red_count": counts["red"],
        "black_count": counts["black"],
        "nnd_mean": nnd_mean,
        "nnd_std": nnd_std,
        "nnd_cv": nnd_cv,
    }


def get_positions(pigments: list[dict]) -> np.ndarray:
    """Return a point cloud from pigment records."""
    if not pigments:
        return np.empty((0, 2), dtype=float)
    return np.asarray([pigment["pos"] for pigment in pigments], dtype=float)


def frame_positions(frame: dict) -> np.ndarray:
    """Return chromatophore centres from one timeline frame."""
    return get_positions(frame["pigments"])


def frame_counts(frame: dict) -> tuple[int, int, int]:
    """Return Y/R/B counts from one frame."""
    return frame["yellow_count"], frame["red_count"], frame["black_count"]


def random_mask_points(skin: np.ndarray, n_points: int, seed: int) -> np.ndarray:
    """Sample a random control inside the same final skin mask."""
    rng = np.random.default_rng(seed)
    skin_sites = np.argwhere(skin == SKIN)
    if len(skin_sites) == 0 or n_points <= 0:
        return np.empty((0, 2), dtype=float)
    chosen_idx = rng.choice(len(skin_sites), size=min(n_points, len(skin_sites)), replace=False)
    chosen = skin_sites[chosen_idx].astype(float)
    jitter = rng.uniform(-0.45, 0.45, size=chosen.shape)
    return np.clip(chosen + jitter, 0.0, skin.shape[0] - 1.0)


def metrics_dataframe(timeline: list[dict]) -> pd.DataFrame:
    """Convert the timeline into a report-ready metrics table."""
    return pd.DataFrame(
        [
            {
                "step": frame["step"],
                "skin_area": frame["skin_area"],
                "pigment_count": frame["pigment_count"],
                "yellow_count": frame["yellow_count"],
                "red_count": frame["red_count"],
                "black_count": frame["black_count"],
                "nnd_mean": frame["nnd_mean"],
                "nnd_std": frame["nnd_std"],
                "nnd_cv": frame["nnd_cv"],
            }
            for frame in timeline
        ]
    )


def nnd_summary_row(model: str, points: np.ndarray) -> dict:
    """Summarize nearest-neighbour statistics for one model."""
    nnd = nearest_neighbor_distances(points)
    if len(nnd) == 0:
        return {"model": model, "N": len(points), "mean_NND": math.nan, "std_NND": math.nan, "CV_NND": math.nan}
    mean_val = float(np.mean(nnd))
    std_val = float(np.std(nnd))
    return {
        "model": model,
        "N": int(len(points)),
        "mean_NND": mean_val,
        "std_NND": std_val,
        "CV_NND": float(std_val / (mean_val + 1e-12)),
    }


def pair_density_curve(points: np.ndarray, bins: np.ndarray) -> np.ndarray:
    """Compute a pair-correlation-like radial pair density."""
    if len(points) < 2:
        return np.zeros(len(bins) - 1)
    dist = pairwise_distances(points, points)
    iu = np.triu_indices(len(points), k=1)
    values = dist[iu]
    counts, _ = np.histogram(values, bins=bins)
    annulus_area = math.pi * (bins[1:] ** 2 - bins[:-1] ** 2)
    density = counts / (len(points) * annulus_area + 1e-12)
    return density


def color_fractions(frame: dict) -> tuple[float, float, float]:
    """Return Y/R/B fractions for one frame."""
    total = max(frame["pigment_count"], 1)
    return (
        frame["yellow_count"] / total,
        frame["red_count"] / total,
        frame["black_count"] / total,
    )


def order_score(frame: dict, params: dict) -> float:
    """Score patterns that are uniform, black-dominant, yellow-present, and red-sparse."""
    y_frac, r_frac, b_frac = color_fractions(frame)
    cv = frame["nnd_cv"]
    if not np.isfinite(cv) or cv <= 0:
        return 0.0

    expected_count = frame["skin_area"] / params["target_area_per_pigment"]
    count_ratio = frame["pigment_count"] / max(expected_count, 1e-6)
    count_score = float(np.exp(-((count_ratio - 1.0) ** 2) / 0.18))

    dominance = max(0.0, b_frac - y_frac * 0.15)
    yellow_presence = float(np.clip(y_frac / 0.22, 0.0, 1.2))
    red_penalty = float(np.exp(-10.0 * r_frac))
    ordering_bonus = 1.0 if (b_frac > y_frac > r_frac) else 0.6
    color_score = dominance * yellow_presence * red_penalty * ordering_bonus
    return float((1.0 / cv) * color_score * count_score)


def render_ca_on_axis(ax: plt.Axes, frame: dict, params: dict, row_label: str | None = None) -> None:
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
    if row_label:
        ax.text(
            -0.10,
            0.5,
            row_label,
            transform=ax.transAxes,
            rotation=90,
            va="center",
            ha="center",
            fontsize=13,
            fontweight="bold",
        )


def save_fig1_development_slices(
    self_timeline: list[dict],
    self_final_step: int,
    random_timeline: list[dict],
    out_path: Path,
    params: dict,
) -> None:
    """Figure 1: side-by-side development slices for self-organized vs random-development."""
    steps = np.linspace(0, self_final_step, 5)
    steps = sorted({int(round(x)) for x in steps})
    while len(steps) < 5:
        steps.append(min(self_final_step, steps[-1] + 1))
    steps = steps[:5]

    fig, axes = plt.subplots(2, len(steps), figsize=(3.8 * len(steps), 7.2), constrained_layout=True)
    row_info = [
        ("Self-organized", self_timeline),
        ("Random-development", random_timeline),
    ]
    for row_idx, (label, timeline) in enumerate(row_info):
        for col_idx, step in enumerate(steps):
            frame = timeline[step]
            ax = axes[row_idx, col_idx]
            render_ca_on_axis(ax, frame, params, row_label=label if col_idx == 0 else None)
            y_count, r_count, b_count = frame_counts(frame)
            ax.set_title(
                f"step {step}\nN={frame['pigment_count']} | Y={y_count} R={r_count} B={b_count}",
                pad=7,
            )
    fig.suptitle("Development Slices: Self-Organized vs Random-Development", y=1.02, fontsize=17, fontweight="bold")
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_fig2_nnd_distribution(
    self_points: np.ndarray,
    random_dev_points: np.ndarray,
    random_mask_points_arr: np.ndarray,
    out_path: Path,
    summary_csv_path: Path,
) -> pd.DataFrame:
    """Figure 2: NND distributions and CV comparison."""
    rows = [
        nnd_summary_row("self_organized", self_points),
        nnd_summary_row("random_development", random_dev_points),
        nnd_summary_row("random_mask", random_mask_points_arr),
    ]
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(summary_csv_path, index=False)

    colors = {
        "self_organized": "#1b1b1b",
        "random_development": "#d07b1a",
        "random_mask": "#7c94b6",
    }
    nnd_series = {
        "self_organized": nearest_neighbor_distances(self_points),
        "random_development": nearest_neighbor_distances(random_dev_points),
        "random_mask": nearest_neighbor_distances(random_mask_points_arr),
    }
    all_vals = np.concatenate([vals for vals in nnd_series.values() if len(vals) > 0])
    bins = np.linspace(0.0, max(8.0, np.percentile(all_vals, 99)), 26)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), constrained_layout=True)
    for label, values in nnd_series.items():
        axes[0].hist(
            values,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2.2,
            color=colors[label],
            label=label.replace("_", "-"),
        )
    axes[0].set_xlabel("Nearest-neighbour distance")
    axes[0].set_ylabel("Density")
    axes[0].set_title("NND Distribution")
    axes[0].legend(frameon=False)

    bar_labels = ["self", "random-dev", "random-mask"]
    bar_values = summary_df["CV_NND"].to_numpy()
    bar_colors = [colors["self_organized"], colors["random_development"], colors["random_mask"]]
    axes[1].bar(bar_labels, bar_values, color=bar_colors, width=0.65)
    axes[1].set_ylabel("CV of nearest-neighbour distance")
    axes[1].set_title("CV_NND Comparison")
    for idx, value in enumerate(bar_values):
        axes[1].text(idx, value + 0.01, f"{value:.3f}", ha="center", va="bottom", fontsize=11)

    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return summary_df


def save_fig3_color_composition(metrics_df: pd.DataFrame, out_path: Path) -> None:
    """Figure 3: developmental colour composition over time."""
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2), constrained_layout=True)

    axes[0].plot(metrics_df["step"], metrics_df["yellow_count"], color="#ffcc33", linewidth=2.5, label="yellow")
    axes[0].plot(metrics_df["step"], metrics_df["red_count"], color="#c94b23", linewidth=2.5, label="red")
    axes[0].plot(metrics_df["step"], metrics_df["black_count"], color="#171411", linewidth=2.8, label="black")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Chromatophore count")
    axes[0].set_title("Colour Composition Over Time")
    axes[0].legend(frameon=False, ncol=3, loc="upper left")

    fractions = metrics_df[["yellow_count", "red_count", "black_count"]].div(
        metrics_df["pigment_count"].replace(0, np.nan),
        axis=0,
    ).fillna(0.0)
    axes[1].plot(metrics_df["step"], fractions["yellow_count"], color="#ffcc33", linewidth=2.5, label="yellow")
    axes[1].plot(metrics_df["step"], fractions["red_count"], color="#c94b23", linewidth=2.5, label="red")
    axes[1].plot(metrics_df["step"], fractions["black_count"], color="#171411", linewidth=2.8, label="black")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Fraction")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title("Colour Fractions Over Time")
    axes[1].legend(frameon=False, ncol=3, loc="upper left")

    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_fig4_parameter_heatmap(scan_df: pd.DataFrame, out_path: Path, params: dict) -> None:
    """Figure 4: 2D scan of spacing parameters."""
    min_vals = sorted(scan_df["absolute_min_distance"].unique())
    gap_vals = sorted(scan_df["target_gap_to_black"].unique())
    cv_grid = (
        scan_df.groupby(["target_gap_to_black", "absolute_min_distance"])["final_cv_nnd"]
        .mean()
        .unstack()
        .reindex(index=gap_vals, columns=min_vals)
    )
    order_grid = (
        scan_df.groupby(["target_gap_to_black", "absolute_min_distance"])["order_score"]
        .mean()
        .unstack()
        .reindex(index=gap_vals, columns=min_vals)
    )

    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.6), constrained_layout=True)
    im1 = axes[0].imshow(cv_grid.to_numpy(), origin="lower", aspect="auto", cmap="magma_r")
    axes[0].set_title("Mean Final CV_NND")
    axes[0].set_xlabel("absolute_min_distance")
    axes[0].set_ylabel("target_gap_to_black")
    axes[0].set_xticks(range(len(min_vals)), [f"{x:.1f}" for x in min_vals])
    axes[0].set_yticks(range(len(gap_vals)), [f"{x:.1f}" for x in gap_vals])
    cbar1 = fig.colorbar(im1, ax=axes[0], shrink=0.92)
    cbar1.set_label("CV_NND")

    im2 = axes[1].imshow(order_grid.to_numpy(), origin="lower", aspect="auto", cmap="viridis")
    axes[1].set_title("Mean Order Score")
    axes[1].set_xlabel("absolute_min_distance")
    axes[1].set_ylabel("target_gap_to_black")
    axes[1].set_xticks(range(len(min_vals)), [f"{x:.1f}" for x in min_vals])
    axes[1].set_yticks(range(len(gap_vals)), [f"{x:.1f}" for x in gap_vals])
    cbar2 = fig.colorbar(im2, ax=axes[1], shrink=0.92)
    cbar2.set_label("Order score")

    x_default = np.interp(params["absolute_min_distance"], min_vals, np.arange(len(min_vals)))
    y_default = np.interp(params["target_gap_to_black"], gap_vals, np.arange(len(gap_vals)))
    for ax in axes:
        ax.scatter(x_default, y_default, s=90, facecolors="none", edgecolors="white", linewidths=1.8)
        ax.scatter(x_default, y_default, s=22, color="white")

    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_fig5_pair_correlation(self_points: np.ndarray, random_points: np.ndarray, out_path: Path) -> None:
    """Figure 5: pair-correlation-like comparison."""
    bins = np.linspace(0.0, 18.0, 31)
    self_curve = pair_density_curve(self_points, bins)
    random_curve = pair_density_curve(random_points, bins)
    centers = 0.5 * (bins[:-1] + bins[1:])

    fig, ax = plt.subplots(figsize=(8.0, 5.2), constrained_layout=True)
    ax.plot(centers, self_curve, color="#171411", linewidth=2.7, label="self-organized")
    ax.plot(centers, random_curve, color="#7c94b6", linewidth=2.4, label="random-mask")
    ax.set_xlabel("Pair distance")
    ax.set_ylabel("Pair-count density")
    ax.set_title("Pair-Correlation-Like Curve")
    ax.legend(frameon=False)
    ax.axvspan(0.0, 3.0, color="#d9d9d9", alpha=0.25)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_model_math_summary(out_path: Path, params: dict, final_frame: dict, summary_df: pd.DataFrame) -> None:
    """Write a short Chinese mathematical summary for the report."""
    text = f"""# 模型数学摘要

## 1. 状态变量

记二维格点元胞自动机状态为 $S_{{ij}}(t) \\in \\{{0,1,2,3,4\\}}$：

- 0：空白区域，尚未长出的皮肤
- 1：普通皮肤细胞
- 2：黄色新生色素细胞
- 3：红色/橙红色过渡态色素细胞
- 4：黑色成熟色素细胞

## 2. 皮肤生长

皮肤只在已有皮肤边界附近向外扩张。对空白格点，若其邻域中已有皮肤，则其转化概率可写为：

$$
P_{{grow}}(i,j,t) = g \\cdot \\left(\\frac{{n_{{skin}}}}{{8}}\\right)^\\alpha \\cdot R_{{radial}}(i,j,t)
$$

其中 $g$ 对应 `growth_rate`，$\\alpha$ 对应 `growth_power`，$R_{{radial}}$ 是随目标半径减速推进的径向门控函数，因此皮肤从中央近圆形扩张并在后期放缓。

## 3. 色素细胞出生

自组织模型中，普通皮肤细胞分化为新生黄色色素细胞的概率可概括为：

$$
P_{{birth}} = \\beta \\cdot I(d_{{all}}) \\cdot G(d_{{black}}) \\cdot B(d_{{boundary}})
$$

- $\\beta$：基础出生率，对应 `base_birth_rate`
- $I(d_{{all}})$：局部抑制项，反映与所有已有色素细胞的抑制场关系
- $G(d_{{black}})$：黑色成熟阵列的间隙偏好项，鼓励新生黄色细胞插入成熟黑色网络空隙
- $B(d_{{boundary}})$：边界惩罚项，避免色素细胞贴皮肤边界生成

random-development 消融则保留皮肤生长、年龄依赖成熟和边界惩罚，但去掉局部抑制与黑色间隙偏好，因此普通皮肤细胞更接近随机分化。

## 4. 颜色成熟

颜色严格由年龄决定，遵循：

$$
Y \\rightarrow R \\rightarrow B
$$

本版默认参数为：

- `yellow_duration = {params["yellow_duration"]}`
- `red_duration = {params["red_duration"]}`

因此黄色阶段较长，红色仅为短暂过渡态，黑色不断累积为成熟主色。

## 5. 自组织与涌现

### 自组织判据

模型中没有全局坐标蓝图，也没有预设晶格模板。每个格点只依据局部皮肤环境、边界距离、局部抑制场和成熟黑色阵列间隙来决定是否分化，但整体上会产生较均匀的色素细胞间距。

### 涌现性判据

单个格点的状态转移规则很简单，但群体层面会形成可作为神经快速变色基础的“皮肤像素阵列”。这说明复杂空间网络可以由局部规则在发育过程中涌现出来。

## 6. 主要图的解释

- `Fig1_development_slices_self_vs_random.png`：显示自组织模型在成熟黑色阵列中持续插入新生黄色细胞，而随机对照更容易出现无结构填充。
- `Fig2_nnd_distribution_and_cv.png`：展示 self-organized、random-development、random-mask 的最近邻距离分布与 CV_NND，对比空间均匀性。
- `Fig3_color_composition_over_time.png`：显示 black 持续累积、yellow 持续补充、red 始终较少，符合短过渡态设想。
- `Fig4_parameter_phase_heatmap.png`：展示从较随机/较拥挤到较均匀插空网络之间的参数转变边界。
- `Fig5_pair_correlation.png`：显示 self-organized 在极短距离处的明显 dip，说明存在短距离排斥。

## 7. 本次代表性结果

- final step = {final_frame["step"]}
- Y/R/B = {final_frame["yellow_count"]}/{final_frame["red_count"]}/{final_frame["black_count"]}
- self-organized CV_NND = {summary_df.loc[summary_df["model"] == "self_organized", "CV_NND"].iloc[0]:.4f}
- random-development CV_NND = {summary_df.loc[summary_df["model"] == "random_development", "CV_NND"].iloc[0]:.4f}
- random-mask CV_NND = {summary_df.loc[summary_df["model"] == "random_mask", "CV_NND"].iloc[0]:.4f}
"""
    out_path.write_text(text, encoding="utf-8")


def run_parameter_scan(params: dict) -> pd.DataFrame:
    """Run a 2D scan to map the transition from random/clustered to ordered gap-filling."""
    records: list[dict] = []
    min_vals = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    gap_vals = [3.0, 4.0, 5.0, 6.0, 7.0]
    seeds = [7, 17, 27]

    for min_dist in min_vals:
        for gap in gap_vals:
            for seed in seeds:
                scan_params = dict(params)
                scan_params["absolute_min_distance"] = min_dist
                scan_params["target_gap_to_black"] = gap
                scan_params["seed"] = seed
                # Keep the same model, but use a lighter report-only scan setting.
                scan_params["n_steps"] = 120
                scan_params["candidate_sample_size"] = min(scan_params["candidate_sample_size"], 900)
                scan_params["relax_iterations"] = 1
                final_frame = simulate_final_self_frame(scan_params)
                y_frac, r_frac, b_frac = color_fractions(final_frame)
                records.append(
                    {
                        "seed": seed,
                        "absolute_min_distance": min_dist,
                        "target_gap_to_black": gap,
                        "final_step": final_frame["step"],
                        "final_cv_nnd": final_frame["nnd_cv"],
                        "pigment_count": final_frame["pigment_count"],
                        "yellow_fraction": y_frac,
                        "red_fraction": r_frac,
                        "black_fraction": b_frac,
                        "order_score": order_score(final_frame, scan_params),
                    }
                )
    return pd.DataFrame.from_records(records)


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    results_dir = base_dir / "results_report"
    if results_dir.exists():
        shutil.rmtree(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    params = dict(PARAMS)
    self_timeline, self_final_step = simulate(params)
    random_timeline, random_final_step = simulate_random_development(params)

    self_final_frame = self_timeline[self_final_step]
    random_final_frame = random_timeline[random_final_step]
    self_points = frame_positions(self_final_frame)
    random_dev_points = frame_positions(random_final_frame)
    random_points = random_mask_points(self_final_frame["skin"], len(self_points), seed=params["seed"] + 400)

    metrics_df = metrics_dataframe(self_timeline)
    metrics_df.to_csv(results_dir / "metrics_over_time.csv", index=False)

    save_fig1_development_slices(
        self_timeline,
        self_final_step,
        random_timeline,
        results_dir / "Fig1_development_slices_self_vs_random.png",
        params,
    )
    summary_df = save_fig2_nnd_distribution(
        self_points,
        random_dev_points,
        random_points,
        results_dir / "Fig2_nnd_distribution_and_cv.png",
        results_dir / "random_vs_self_summary.csv",
    )
    save_fig3_color_composition(metrics_df, results_dir / "Fig3_color_composition_over_time.png")

    scan_df = run_parameter_scan(params)
    scan_df.to_csv(results_dir / "parameter_scan.csv", index=False)
    save_fig4_parameter_heatmap(scan_df, results_dir / "Fig4_parameter_phase_heatmap.png", params)
    save_fig5_pair_correlation(self_points, random_points, results_dir / "Fig5_pair_correlation.png")
    write_model_math_summary(results_dir / "model_math_summary.md", params, self_final_frame, summary_df)

    best_cv_row = scan_df.loc[scan_df["final_cv_nnd"].idxmin()]
    best_order_row = scan_df.loc[scan_df["order_score"].idxmax()]

    print(f"final_step={self_final_step}")
    print(f"self CV_NND={summary_df.loc[summary_df['model'] == 'self_organized', 'CV_NND'].iloc[0]:.6f}")
    print(f"random-development CV_NND={summary_df.loc[summary_df['model'] == 'random_development', 'CV_NND'].iloc[0]:.6f}")
    print(f"random-mask CV_NND={summary_df.loc[summary_df['model'] == 'random_mask', 'CV_NND'].iloc[0]:.6f}")
    print(
        f"final Y/R/B counts={self_final_frame['yellow_count']}/{self_final_frame['red_count']}/{self_final_frame['black_count']}"
    )
    print(
        "lowest CV_NND scan="
        f"absolute_min_distance={best_cv_row['absolute_min_distance']:.1f}, "
        f"target_gap_to_black={best_cv_row['target_gap_to_black']:.1f}, "
        f"CV_NND={best_cv_row['final_cv_nnd']:.6f}"
    )
    print(
        "highest order_score scan="
        f"absolute_min_distance={best_order_row['absolute_min_distance']:.1f}, "
        f"target_gap_to_black={best_order_row['target_gap_to_black']:.1f}, "
        f"order_score={best_order_row['order_score']:.6f}"
    )


if __name__ == "__main__":
    main()
