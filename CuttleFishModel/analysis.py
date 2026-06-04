from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .model import (
    PARAMS,
    SKIN,
    age_inhibition_field,
    center_birth_gate,
    choose_final_step,
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
from .visualization import add_bar_value_labels, apply_report_style, render_ca_on_axis as viz_render_ca_on_axis


apply_report_style()


@dataclass(frozen=True)
class MechanismConfig:
    """Explicit switches for the three local developmental mechanisms."""

    name: str
    display_name: str
    use_repulsion: bool
    use_gap_birth: bool
    use_growth_displacement: bool
    use_boundary_penalty: bool = True
    matched_birth_control: bool = False
    random_mask: bool = False


MODE_CONFIGS = {
    "full_self_organized": MechanismConfig(
        name="full_self_organized",
        display_name="Full self-organized",
        use_repulsion=True,
        use_gap_birth=True,
        use_growth_displacement=True,
    ),
    "no_repulsion": MechanismConfig(
        name="no_repulsion",
        display_name="No repulsion",
        use_repulsion=False,
        use_gap_birth=True,
        use_growth_displacement=True,
        matched_birth_control=True,
    ),
    "no_gap_birth": MechanismConfig(
        name="no_gap_birth",
        display_name="No gap-biased birth",
        use_repulsion=True,
        use_gap_birth=False,
        use_growth_displacement=True,
        matched_birth_control=True,
    ),
    "no_growth_displacement": MechanismConfig(
        name="no_growth_displacement",
        display_name="No growth displacement",
        use_repulsion=True,
        use_gap_birth=True,
        use_growth_displacement=False,
    ),
    "random_development_matched": MechanismConfig(
        name="random_development_matched",
        display_name="Random-development matched",
        use_repulsion=False,
        use_gap_birth=False,
        use_growth_displacement=True,
        matched_birth_control=True,
    ),
    "random_mask_matched": MechanismConfig(
        name="random_mask_matched",
        display_name="Random-mask matched",
        use_repulsion=False,
        use_gap_birth=False,
        use_growth_displacement=False,
        random_mask=True,
    ),
}

MODE_ORDER = [
    "full_self_organized",
    "no_repulsion",
    "no_gap_birth",
    "no_growth_displacement",
    "random_development_matched",
    "random_mask_matched",
]

MODE_COLORS = {
    "full_self_organized": "#171411",
    "no_repulsion": "#355c7d",
    "no_gap_birth": "#f08a24",
    "no_growth_displacement": "#7b3294",
    "random_development_matched": "#c94b23",
    "random_mask_matched": "#7c94b6",
}


def deep_copy_pigments(pigments: list[dict]) -> list[dict]:
    """Copy pigment records without sharing array references."""
    return [
        {
            "pos": pigment["pos"].copy(),
            "age": int(pigment["age"]),
            "base_major": float(pigment["base_major"]),
            "base_minor": float(pigment["base_minor"]),
            "angle": float(pigment["angle"]),
            "tone": float(pigment["tone"]),
        }
        for pigment in pigments
    ]


def get_positions(pigments: list[dict]) -> np.ndarray:
    """Return chromatophore centres as a point cloud."""
    if not pigments:
        return np.empty((0, 2), dtype=float)
    return np.asarray([pigment["pos"] for pigment in pigments], dtype=float)


def frame_positions(frame: dict) -> np.ndarray:
    """Return chromatophore centres from one frame."""
    return get_positions(frame["pigments"])


def frame_counts(frame: dict) -> tuple[int, int, int]:
    """Return Y/R/B counts from one frame."""
    return frame["yellow_count"], frame["red_count"], frame["black_count"]


def build_frame(step: int, skin: np.ndarray, pigments: list[dict], params: dict) -> dict:
    """Assemble one timeline frame."""
    points = get_positions(pigments)
    nnd_mean, nnd_std, nnd_cv = summarize_nnd(points)
    counts = compute_counts(pigments, params)
    return {
        "step": int(step),
        "skin": skin.copy(),
        "pigments": deep_copy_pigments(pigments),
        "skin_area": int(np.count_nonzero(skin == SKIN)),
        "pigment_count": int(len(pigments)),
        "yellow_count": counts["yellow"],
        "red_count": counts["red"],
        "black_count": counts["black"],
        "nnd_mean": nnd_mean,
        "nnd_std": nnd_std,
        "nnd_cv": nnd_cv,
    }


def compute_distance_to_all(sampled: np.ndarray, pigments: list[dict]) -> np.ndarray:
    """Distance to the nearest existing pigment."""
    if not pigments:
        return np.full(len(sampled), np.inf)
    positions = get_positions(pigments)
    return pairwise_distances(sampled, positions).min(axis=1)


def choose_births_by_mechanism(
    skin: np.ndarray,
    skin_age: np.ndarray,
    pigments: list[dict],
    params: dict,
    rng: np.random.Generator,
    mechanism: MechanismConfig,
    birth_rate_scale: float,
) -> list[dict]:
    """Generalized birth rule with switchable local mechanisms."""
    eligible = (skin == SKIN) & (skin_age >= params["min_skin_age_for_diff"])
    candidate_positions = np.argwhere(eligible)
    if len(candidate_positions) == 0:
        return []

    sample_size = min(params["candidate_sample_size"], len(candidate_positions))
    sampled_idx = rng.choice(len(candidate_positions), size=sample_size, replace=False)
    sampled = candidate_positions[sampled_idx].astype(float)

    d_all = compute_distance_to_all(sampled, pigments)
    if mechanism.use_repulsion:
        inhibition_field, d_all = age_inhibition_field(sampled, pigments, params)
        inhibition_allow = 1.0 / (
            1.0 + np.exp((inhibition_field - params["field_threshold"]) / params["field_softness"])
        )
    else:
        inhibition_allow = np.ones(len(sampled), dtype=float)

    black_positions = np.asarray(
        [pigment["pos"] for pigment in pigments if pigment_stage(pigment["age"], params) == 4],
        dtype=float,
    )
    if len(black_positions) > 0:
        d_black = pairwise_distances(sampled, black_positions).min(axis=1)
    else:
        d_black = np.full(len(sampled), np.inf)

    if mechanism.use_boundary_penalty:
        boundary_positions = compute_boundary_positions(skin)
        if len(boundary_positions) > 0:
            d_boundary = pairwise_distances(sampled, boundary_positions).min(axis=1)
        else:
            d_boundary = np.full(len(sampled), np.inf)
        boundary_gate = 1.0 / (
            1.0 + np.exp(-(d_boundary - params["min_boundary_distance"]) / params["boundary_softness"])
        )
    else:
        d_boundary = np.full(len(sampled), np.inf)
        boundary_gate = np.ones(len(sampled), dtype=float)

    center_gate = center_birth_gate(sampled, skin, params)

    if mechanism.use_gap_birth:
        if len(black_positions) >= params["bootstrap_black_count"]:
            gap_pref = np.exp(
                -((d_black - params["target_gap_to_black"]) ** 2) / (2.0 * params["target_gap_sigma"] ** 2)
            )
            if mechanism.use_repulsion:
                gap_pref *= 1.0 / (1.0 + np.exp(-(d_black - params["absolute_min_distance"]) / 0.7))
            gap_pref *= params["black_gap_weight"]
        else:
            gap_pref = np.exp(
                -((d_all - params["bootstrap_gap_to_all"]) ** 2)
                / (2.0 * (params["target_gap_sigma"] * 1.25) ** 2)
            )
            if mechanism.use_repulsion:
                gap_pref *= 1.0 / (1.0 + np.exp(-(d_all - params["absolute_min_distance"]) / 0.8))
            gap_pref *= params["fallback_gap_weight"]
    else:
        gap_pref = np.ones(len(sampled), dtype=float)

    birth_prob = np.clip(
        params["base_birth_rate"] * birth_rate_scale * inhibition_allow * gap_pref * boundary_gate * center_gate,
        0.0,
        params["birth_prob_cap"],
    )
    if mechanism.use_repulsion:
        birth_prob[d_all < params["absolute_min_distance"]] = 0.0
    if mechanism.use_boundary_penalty:
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

    order = np.argsort(birth_prob + rng.uniform(0.0, 1e-6, size=len(sampled)))[::-1]
    chosen: list[dict] = []
    existing_positions = [pigment["pos"] for pigment in pigments]
    for idx in order:
        if len(chosen) >= dynamic_quota:
            break
        if rng.random() >= birth_prob[idx]:
            continue
        pos = sampled[idx]
        if mechanism.use_repulsion and existing_positions:
            dist_existing = np.sqrt(np.sum((np.asarray(existing_positions) - pos) ** 2, axis=1))
            if np.any(dist_existing < params["absolute_min_distance"]):
                continue
        if mechanism.use_repulsion and chosen:
            chosen_positions = np.asarray([pigment["pos"] for pigment in chosen], dtype=float)
            dist_chosen = np.sqrt(np.sum((chosen_positions - pos) ** 2, axis=1))
            if np.any(dist_chosen < params["absolute_min_distance"]):
                continue
        chosen.append(make_pigment(pos.copy(), params, rng, age=0))
    return chosen


def simulate_mode(
    params: dict,
    mechanism: MechanismConfig,
    *,
    seed: int,
    birth_rate_scale: float = 1.0,
    capture_timeline: bool = True,
) -> tuple[list[dict] | None, dict, int]:
    """Run one mode with explicit mechanism switches."""
    if mechanism.name == "full_self_organized" and birth_rate_scale == 1.0 and capture_timeline:
        full_params = dict(params)
        full_params["seed"] = seed
        timeline, final_step = simulate(full_params)
        return timeline, timeline[final_step], final_step

    run_params = dict(params)
    run_params["seed"] = seed
    rng_init = np.random.default_rng(seed + 1)
    rng_growth = np.random.default_rng(seed + 2)
    rng_birth = np.random.default_rng(seed + 3)

    skin, skin_age = initialize_skin(run_params)
    pigments = initialize_pigments(skin, run_params, rng_init)
    center = np.array([(run_params["grid_size"] - 1) / 2.0, (run_params["grid_size"] - 1) / 2.0], dtype=float)
    previous_radius = 0.5 * (run_params["initial_radius_x"] + run_params["initial_radius_y"])
    timeline: list[dict] = []

    for step in range(run_params["n_steps"]):
        if step > 0:
            occupied = skin == SKIN
            skin_age[occupied] += 1
            _, current_radius = grow_skin(skin, skin_age, run_params, rng_growth, step)
            if mechanism.use_growth_displacement:
                move_with_skin_growth(pigments, previous_radius, current_radius, run_params, center)
            project_inside_skin(pigments, skin, center)
            newborn = choose_births_by_mechanism(
                skin,
                skin_age,
                pigments,
                run_params,
                rng_birth,
                mechanism,
                birth_rate_scale,
            )
            pigments.extend(newborn)
            if mechanism.use_growth_displacement and mechanism.use_repulsion:
                relax_overlaps(pigments, run_params, center)
            project_inside_skin(pigments, skin, center)
            for pigment in pigments:
                pigment["age"] += 1
            previous_radius = current_radius

        if capture_timeline:
            timeline.append(build_frame(step, skin, pigments, run_params))

    if capture_timeline:
        final_step = choose_final_step(timeline, run_params)
        return timeline, timeline[final_step], final_step

    final_frame = build_frame(run_params["n_steps"] - 1, skin, pigments, run_params)
    return None, final_frame, final_frame["step"]


def estimate_final_count(params: dict, mechanism: MechanismConfig, scale: float, seed: int) -> int:
    """Fast final-count estimate for matched-count calibration."""
    _, final_frame, _ = simulate_mode(
        params,
        mechanism,
        seed=seed,
        birth_rate_scale=scale,
        capture_timeline=False,
    )
    return int(final_frame["pigment_count"])


def calibrate_birth_scale(
    params: dict,
    mechanism: MechanismConfig,
    target_n: int,
    *,
    seed: int,
) -> float:
    """Tune birth rate so the final pigment count matches the self-organized reference."""
    low = 0.01
    high = 1.0
    low_n = estimate_final_count(params, mechanism, low, seed)
    while low_n > target_n and low > 0.001:
        low *= 0.5
        low_n = estimate_final_count(params, mechanism, low, seed)

    high_n = estimate_final_count(params, mechanism, high, seed)
    while high_n < target_n and high < 3.0:
        high *= 1.4
        high_n = estimate_final_count(params, mechanism, high, seed)

    best_scale = high
    best_error = abs(high_n - target_n)
    candidates = [(low, low_n), (high, high_n)]
    for scale, n_val in candidates:
        err = abs(n_val - target_n)
        if err < best_error:
            best_scale = scale
            best_error = err

    for _ in range(8):
        mid = 0.5 * (low + high)
        mid_n = estimate_final_count(params, mechanism, mid, seed)
        err = abs(mid_n - target_n)
        if err < best_error:
            best_scale = mid
            best_error = err
        if mid_n > target_n:
            high = mid
        else:
            low = mid
    return float(best_scale)


def refine_birth_scale_with_timeline(
    params: dict,
    mechanism: MechanismConfig,
    target_n: int,
    *,
    seed: int,
    initial_scale: float,
) -> float:
    """Refine matched-count calibration using the representative final frame, not only the last step."""
    scale = max(initial_scale, 0.0005)
    best_scale = scale
    best_error = float("inf")

    for _ in range(3):
        multipliers = [0.40, 0.60, 0.80, 1.00, 1.25, 1.60, 2.00, 2.60, 3.20]
        candidate_scales = sorted({max(0.0002, scale * m) for m in multipliers})
        candidate_results = []
        for candidate_scale in candidate_scales:
            _, final_frame, _ = simulate_mode(
                params,
                mechanism,
                seed=seed,
                birth_rate_scale=candidate_scale,
                capture_timeline=True,
            )
            error = abs(final_frame["pigment_count"] - target_n)
            candidate_results.append((candidate_scale, final_frame["pigment_count"], error))
            if error < best_error:
                best_scale = candidate_scale
                best_error = error
        candidate_results.sort(key=lambda item: item[2])
        scale = candidate_results[0][0]
        if best_error / max(target_n, 1) <= 0.10:
            break
    return float(best_scale)


def metrics_dataframe(timeline: list[dict]) -> pd.DataFrame:
    """Convert one timeline into a metrics table."""
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


def color_fractions(frame: dict) -> tuple[float, float, float]:
    """Return Y/R/B fractions for one frame."""
    total = max(frame["pigment_count"], 1)
    return (
        frame["yellow_count"] / total,
        frame["red_count"] / total,
        frame["black_count"] / total,
    )


def build_summary_row(
    model: str,
    frame: dict,
    mechanism: MechanismConfig,
    *,
    seed: int,
    calibrated_birth_rate: float,
) -> dict:
    """Create one summary row with counts, fractions, and switch states."""
    y_frac, r_frac, b_frac = color_fractions(frame)
    return {
        "model": model,
        "seed": seed,
        "N": frame["pigment_count"],
        "mean_NND": frame["nnd_mean"],
        "std_NND": frame["nnd_std"],
        "CV_NND": frame["nnd_cv"],
        "yellow_count": frame["yellow_count"],
        "red_count": frame["red_count"],
        "black_count": frame["black_count"],
        "yellow_fraction": y_frac,
        "red_fraction": r_frac,
        "black_fraction": b_frac,
        "use_repulsion": mechanism.use_repulsion,
        "use_gap_birth": mechanism.use_gap_birth,
        "use_growth_displacement": mechanism.use_growth_displacement,
        "calibrated_birth_rate": calibrated_birth_rate,
    }


def random_mask_frame(self_final_frame: dict, seed: int) -> dict:
    """Build a random-mask matched control with the same N and same Y/R/B composition as self."""
    rng = np.random.default_rng(seed)
    skin = self_final_frame["skin"].copy()
    skin_sites = np.argwhere(skin == SKIN)
    n_points = self_final_frame["pigment_count"]
    chosen_idx = rng.choice(len(skin_sites), size=min(n_points, len(skin_sites)), replace=False)
    chosen = skin_sites[chosen_idx].astype(float)
    jitter = rng.uniform(-0.45, 0.45, size=chosen.shape)
    positions = np.clip(chosen + jitter, 0.0, skin.shape[0] - 1.0)

    pigments = deep_copy_pigments(self_final_frame["pigments"])
    perm = rng.permutation(len(positions))
    for pigment, pos in zip(pigments, positions[perm]):
        pigment["pos"] = pos.copy()
    frame = build_frame(self_final_frame["step"], skin, pigments, PARAMS)
    return frame


def pair_density_curve(points: np.ndarray, bins: np.ndarray) -> np.ndarray:
    """Pair-correlation-like radial density."""
    if len(points) < 2:
        return np.zeros(len(bins) - 1)
    dist = pairwise_distances(points, points)
    iu = np.triu_indices(len(points), k=1)
    values = dist[iu]
    counts, _ = np.histogram(values, bins=bins)
    annulus_area = math.pi * (bins[1:] ** 2 - bins[:-1] ** 2)
    return counts / (len(points) * annulus_area + 1e-12)


def order_score(frame: dict, params: dict) -> float:
    """Score patterns that are uniform, black-dominant, yellow-present, and red-sparse."""
    y_frac, r_frac, b_frac = color_fractions(frame)
    cv = frame["nnd_cv"]
    if not np.isfinite(cv) or cv <= 0:
        return 0.0
    expected_count = frame["skin_area"] / params["target_area_per_pigment"]
    count_ratio = frame["pigment_count"] / max(expected_count, 1e-6)
    count_score = float(np.exp(-((count_ratio - 1.0) ** 2) / 0.18))
    color_score = max(0.0, b_frac - 0.2 * y_frac) * np.clip(y_frac / 0.22, 0.0, 1.2) * np.exp(-10.0 * r_frac)
    if not (b_frac > y_frac > r_frac):
        color_score *= 0.6
    return float((1.0 / cv) * color_score * count_score)


def render_ca_on_axis(ax: plt.Axes, frame: dict, params: dict, side_label: str | None = None) -> None:
    """Render one CA state using the same lattice style as the GIF."""
    viz_render_ca_on_axis(ax, frame, params, side_label=side_label)


def save_fig1_development_slices(
    self_timeline: list[dict],
    self_final_step: int,
    random_timeline: list[dict],
    out_path: Path,
    params: dict,
) -> None:
    """Figure 1: development slices for self-organized vs matched random-development."""
    steps = np.linspace(0, self_final_step, 5)
    steps = sorted({int(round(x)) for x in steps})
    while len(steps) < 5:
        steps.append(min(self_final_step, steps[-1] + 1))
    steps = steps[:5]

    fig, axes = plt.subplots(2, len(steps), figsize=(3.8 * len(steps), 7.2), constrained_layout=True)
    rows = [("Self-organized", self_timeline), ("Random-development matched", random_timeline)]
    for row_idx, (label, timeline) in enumerate(rows):
        for col_idx, step in enumerate(steps):
            frame = timeline[step]
            ax = axes[row_idx, col_idx]
            render_ca_on_axis(ax, frame, params, side_label=label if col_idx == 0 else None)
            ax.set_title(
                f"step {step}\nN={frame['pigment_count']} | Y={frame['yellow_count']} R={frame['red_count']} B={frame['black_count']}",
                pad=7,
            )
    fig.suptitle("Development Slices: Self-Organized vs Random-Development Matched", y=1.02, fontsize=17, fontweight="bold")
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_fig2_nnd_distribution_and_cv(summary_df: pd.DataFrame, nnd_map: dict[str, np.ndarray], out_path: Path) -> None:
    """Figure 2: NND distributions and CV comparison for matched controls."""
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), constrained_layout=True)
    all_vals = np.concatenate([vals for vals in nnd_map.values() if len(vals) > 0])
    bins = np.linspace(0.0, max(8.0, np.percentile(all_vals, 99)), 26)
    for model in ["full_self_organized", "random_development_matched", "random_mask_matched"]:
        axes[0].hist(
            nnd_map[model],
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2.2,
            color=MODE_COLORS[model],
            label=MODE_CONFIGS[model].display_name,
        )
    axes[0].set_xlabel("Nearest-neighbour distance")
    axes[0].set_ylabel("Density")
    axes[0].set_title("NND Distribution")
    axes[0].legend(frameon=False, loc="upper left")

    order = ["full_self_organized", "random_development_matched", "random_mask_matched"]
    bars = summary_df.set_index("model").loc[order, "CV_NND"]
    axes[1].bar(
        ["self", "random-dev", "random-mask"],
        bars.to_numpy(),
        color=[MODE_COLORS[key] for key in order],
        width=0.65,
    )
    axes[1].set_ylabel("CV of nearest-neighbour distance")
    axes[1].set_title("CV_NND Comparison")
    add_bar_value_labels(axes[1], bars.to_numpy(), fmt="{:.3f}", fontsize=11)

    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_fig3_color_composition(metrics_df: pd.DataFrame, out_path: Path) -> None:
    """Figure 3: colour composition over time for the full self-organized model."""
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2), constrained_layout=True)
    axes[0].plot(metrics_df["step"], metrics_df["yellow_count"], color="#ffcc33", linewidth=2.5, label="yellow")
    axes[0].plot(metrics_df["step"], metrics_df["red_count"], color="#c94b23", linewidth=2.5, label="red")
    axes[0].plot(metrics_df["step"], metrics_df["black_count"], color="#171411", linewidth=2.8, label="black")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Chromatophore count")
    axes[0].set_title("Colour Composition Over Time", pad=20)
    axes[0].margins(y=0.08)

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
    axes[1].set_title("Colour Fractions Over Time", pad=20)
    axes[1].margins(y=0.05)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.03))

    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_fig4_parameter_heatmap(scan_df: pd.DataFrame, out_path: Path, params: dict) -> None:
    """Figure 4: local-rule parameter landscape of spacing order."""
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

    fig.suptitle("Local-rule Parameter Landscape of Spacing Order", y=1.02, fontsize=17, fontweight="bold")
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_fig5_pair_correlation(self_points: np.ndarray, random_points: np.ndarray, out_path: Path) -> None:
    """Figure 5: pair-correlation-like comparison."""
    bins = np.linspace(0.0, 18.0, 31)
    self_curve = pair_density_curve(self_points, bins)
    random_curve = pair_density_curve(random_points, bins)
    centers = 0.5 * (bins[:-1] + bins[1:])

    fig, ax = plt.subplots(figsize=(8.0, 5.2), constrained_layout=True)
    ax.plot(centers, self_curve, color=MODE_COLORS["full_self_organized"], linewidth=2.7, label="full self-organized")
    ax.plot(centers, random_curve, color=MODE_COLORS["random_mask_matched"], linewidth=2.4, label="random-mask matched")
    ax.set_xlabel("Pair distance")
    ax.set_ylabel("Pair-count density")
    ax.set_title("Pair-Correlation-Like Curve")
    ax.legend(frameon=False)
    ax.axvspan(0.0, 3.0, color="#d9d9d9", alpha=0.25)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_fig6_ablation_summary(ablation_df: pd.DataFrame, out_path: Path) -> None:
    """Figure 6: ablation summary bars."""
    order = MODE_ORDER
    summary = ablation_df.set_index("model").loc[order]
    labels = [
        "full",
        "no-rep",
        "no-gap",
        "no-move",
        "rand-dev",
        "rand-mask",
    ]
    colors = [MODE_COLORS[key] for key in order]

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.6), constrained_layout=True)
    axes[0].bar(labels, summary["CV_NND"], color=colors, width=0.68)
    axes[0].set_title("CV_NND")
    axes[0].set_ylabel("CV_NND")
    axes[0].tick_params(axis="x", rotation=25)
    add_bar_value_labels(axes[0], summary["CV_NND"].to_numpy(), fmt="{:.3f}", fontsize=10)

    axes[1].bar(labels, summary["N"], color=colors, width=0.68)
    axes[1].set_title("Pigment count")
    axes[1].set_ylabel("N")
    axes[1].tick_params(axis="x", rotation=25)
    add_bar_value_labels(axes[1], summary["N"].to_numpy(), fmt="{:.0f}", fontsize=10)

    bottoms = np.zeros(len(order))
    frac_cols = [
        ("yellow_fraction", "#ffcc33", "yellow"),
        ("red_fraction", "#c94b23", "red"),
        ("black_fraction", "#171411", "black"),
    ]
    for col, color, label in frac_cols:
        axes[2].bar(labels, summary[col], bottom=bottoms, color=color, width=0.68, label=label)
        bottoms += summary[col].to_numpy()
    axes[2].set_ylim(0.0, 1.0)
    axes[2].set_title("Colour fractions")
    axes[2].set_ylabel("Fraction")
    axes[2].tick_params(axis="x", rotation=25)
    axes[2].legend(frameon=False, loc="upper left")

    fig.suptitle("Ablation Summary", y=1.02, fontsize=17, fontweight="bold")
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_fig7_ablation_final_patterns(frames: dict[str, dict], out_path: Path, params: dict) -> None:
    """Figure 7: final pattern array for all ablation modes."""
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 8.0), constrained_layout=True)
    for ax, mode in zip(axes.flat, MODE_ORDER):
        frame = frames[mode]
        render_ca_on_axis(ax, frame, params)
        ax.set_title(
            f"{MODE_CONFIGS[mode].display_name}\nN={frame['pigment_count']} | CV={frame['nnd_cv']:.3f}",
            pad=7,
        )
    fig.suptitle("Final Patterns Across Mechanism Ablations", y=1.02, fontsize=17, fontweight="bold")
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def run_parameter_scan(params: dict) -> pd.DataFrame:
    """Scan the full self-organized local-rule landscape."""
    records: list[dict] = []
    min_vals = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    gap_vals = [3.0, 4.0, 5.0, 6.0, 7.0]
    seeds = [7, 17, 27]
    mechanism = MODE_CONFIGS["full_self_organized"]

    for min_dist in min_vals:
        for gap in gap_vals:
            for seed in seeds:
                scan_params = dict(params)
                scan_params["absolute_min_distance"] = min_dist
                scan_params["target_gap_to_black"] = gap
                scan_params["candidate_sample_size"] = min(scan_params["candidate_sample_size"], 900)
                scan_params["n_steps"] = 120
                scan_params["relax_iterations"] = 1
                _, final_frame, _ = simulate_mode(
                    scan_params,
                    mechanism,
                    seed=seed,
                    birth_rate_scale=1.0,
                    capture_timeline=False,
                )
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


def write_model_math_summary(
    out_path: Path,
    params: dict,
    summary_df: pd.DataFrame,
    ablation_df: pd.DataFrame,
    calibration_map: dict[str, float],
) -> None:
    """Write a concise Chinese report summary with the new matched controls and ablations."""
    self_row = summary_df.set_index("model").loc["full_self_organized"]
    rand_row = summary_df.set_index("model").loc["random_development_matched"]
    mask_row = summary_df.set_index("model").loc["random_mask_matched"]
    no_rep = ablation_df.set_index("model").loc["no_repulsion"]
    no_gap = ablation_df.set_index("model").loc["no_gap_birth"]
    no_move = ablation_df.set_index("model").loc["no_growth_displacement"]
    text = f"""# 模型数学摘要

## 1. 状态变量

记二维格点元胞自动机状态为 $S_{{ij}}(t) \\in \\{{0,1,2,3,4\\}}$：

- 0：空白区域，尚未长出的皮肤
- 1：普通皮肤细胞
- 2：黄色新生色素细胞
- 3：红色/橙红色过渡态色素细胞
- 4：黑色成熟色素细胞

## 2. 皮肤生长概率

皮肤只在已有皮肤边界附近扩张，概率可概括为：

$$
P_{{grow}}(i,j,t)=g\\left(\\frac{{n_{{skin}}}}{{8}}\\right)^\\alpha R_{{radial}}(i,j,t)
$$

其中 $g$ 对应 `growth_rate`，$\\alpha$ 对应 `growth_power`，$R_{{radial}}$ 为径向门控函数，因此皮肤从中心近圆形扩张并在后期减速。

## 3. 色素细胞出生概率

full self-organized 模型中，普通皮肤细胞分化为新生黄色色素细胞的概率写成：

$$
P_{{birth}} = \\beta \\cdot I(d_{{all}}) \\cdot G(d_{{black}}) \\cdot B(d_{{boundary}})
$$

- $I(d_{{all}})$：局部抑制/短程排斥项，避免新生细胞离已有色素细胞过近
- $G(d_{{black}})$：成熟黑色阵列空隙偏好项，鼓励黄色细胞插入黑色网络 gap
- $B(d_{{boundary}})$：边界惩罚项，避免新生色素细胞总贴边出现

## 4. 颜色成熟

颜色严格由年龄决定：

$$
Y \\rightarrow R \\rightarrow B
$$

当前参数为：

- `yellow_duration = {params["yellow_duration"]}`
- `red_duration = {params["red_duration"]}`

因此黄色持续较久，红色只是短暂过渡态，黑色在发育过程中不断累积。

## 5. 为什么要做 matched random-development

原始 random-development 会产生远多于 self-organized 的 pigment，因此最近邻距离和 CV_NND 会受到密度差异污染，无法构成公平对照。

因此本次对 random-development 使用了 birth-rate calibration，使其最终 pigment 数量接近 self-organized。当前：

- full self-organized：N = {int(self_row["N"])}，CV_NND = {self_row["CV_NND"]:.4f}
- random-development matched：N = {int(rand_row["N"])}，CV_NND = {rand_row["CV_NND"]:.4f}
- random-mask matched：N = {int(mask_row["N"])}，CV_NND = {mask_row["CV_NND"]:.4f}

random-development matched 使用的校准 birth rate scale 为 {calibration_map["random_development_matched"]:.4f}。

## 6. 三项机制与 ablation

full self-organized 同时包含三项局部机制：

1. pigment-pigment repulsion  
   控制短距离排斥和最小间距约束。

2. gap-biased birth / intercalation  
   控制新生黄色细胞优先插入成熟黑色阵列空隙。

3. growth-associated displacement  
   控制皮肤生长与内部插入导致的新旧 pigment 局部位移与重排。

对应 ablation：

- no_repulsion：去掉短程排斥，只保留空隙偏好与组织位移。
- no_gap_birth：保留排斥，但去掉“向黑色阵列 gap 插空”的出生偏好。
- no_growth_displacement：保留排斥与 gap birth，但 pigment 出生后位置不再随组织生长重排。

本次结果：

- no_repulsion：N = {int(no_rep["N"])}，CV_NND = {no_rep["CV_NND"]:.4f}
- no_gap_birth：N = {int(no_gap["N"])}，CV_NND = {no_gap["CV_NND"]:.4f}
- no_growth_displacement：N = {int(no_move["N"])}，CV_NND = {no_move["CV_NND"]:.4f}

如果去掉某机制后 CV_NND 上升，或者图像变得更随机、更拥挤或更空洞，就说明该局部规则对全局均匀网络形成有贡献。

## 7. 自组织与涌现

### 自组织判据

模型没有全局坐标蓝图；每个格点只依据局部皮肤环境、边界距离、局部抑制场和成熟黑色阵列 gap 来决定是否分化，但整体仍能形成较均匀的色素细胞间距。

### 涌现性判据

单格点规则很简单，但群体层面会形成可作为成年神经快速变色基础的“皮肤像素阵列”。这说明全局空间秩序可以由多个局部规则叠加涌现。

## 8. 主要图的解释

- `Fig1_development_slices_self_vs_random.png`：比较 full self-organized 与 matched random-development 的发育切片。
- `Fig2_nnd_distribution_and_cv.png`：比较 self、matched random-development、matched random-mask 的 NND 分布与 CV_NND。
- `Fig3_color_composition_over_time.png`：展示 black 累积、yellow 持续补充、red 始终较少。
- `Fig4_parameter_phase_heatmap.png`：展示 full self-organized 在局部规则参数上的 spacing order landscape。
- `Fig5_pair_correlation.png`：显示 self-organized 在短距离处的 pair-density dip。
- `Fig6_ablation_summary.png` 与 `Fig7_ablation_final_patterns.png`：总结去掉不同机制后，空间均匀性和最终图景如何变化。
"""
    out_path.write_text(text, encoding="utf-8")


def main() -> None:
    base_dir = Path(__file__).resolve().parent.parent
    results_dir = base_dir / "results_report"
    if results_dir.exists():
        shutil.rmtree(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    params = dict(PARAMS)
    display_seed = params["seed"]

    full_timeline, full_final_frame, full_final_step = simulate_mode(
        params,
        MODE_CONFIGS["full_self_organized"],
        seed=display_seed,
        birth_rate_scale=1.0,
        capture_timeline=True,
    )
    assert full_timeline is not None
    target_n = int(full_final_frame["pigment_count"])

    calibration_targets = {}
    for mode in ["no_repulsion", "no_gap_birth", "random_development_matched"]:
        rough_scale = calibrate_birth_scale(
            params,
            MODE_CONFIGS[mode],
            target_n,
            seed=display_seed,
        )
        calibration_targets[mode] = refine_birth_scale_with_timeline(
            params,
            MODE_CONFIGS[mode],
            target_n,
            seed=display_seed,
            initial_scale=rough_scale,
        )

    timelines: dict[str, list[dict]] = {"full_self_organized": full_timeline}
    final_frames: dict[str, dict] = {"full_self_organized": full_final_frame}
    final_steps: dict[str, int] = {"full_self_organized": full_final_step}
    birth_scales = {"full_self_organized": 1.0}
    birth_scales.update(calibration_targets)
    birth_scales["no_growth_displacement"] = 1.0

    for mode in ["no_repulsion", "no_gap_birth", "no_growth_displacement", "random_development_matched"]:
        timeline, final_frame, final_step = simulate_mode(
            params,
            MODE_CONFIGS[mode],
            seed=display_seed,
            birth_rate_scale=birth_scales[mode],
            capture_timeline=True,
        )
        assert timeline is not None
        timelines[mode] = timeline
        final_frames[mode] = final_frame
        final_steps[mode] = final_step

    final_frames["random_mask_matched"] = random_mask_frame(full_final_frame, seed=display_seed + 400)
    final_steps["random_mask_matched"] = full_final_step

    metrics_df = metrics_dataframe(full_timeline)
    metrics_df.to_csv(results_dir / "metrics_over_time.csv", index=False)

    summary_rows = []
    for mode in ["full_self_organized", "random_development_matched", "random_mask_matched"]:
        scale = birth_scales.get(mode, 1.0)
        summary_rows.append(
            build_summary_row(
                mode,
                final_frames[mode],
                MODE_CONFIGS[mode],
                seed=display_seed,
                calibrated_birth_rate=scale,
            )
        )
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(results_dir / "random_vs_self_summary.csv", index=False)

    ablation_rows = []
    for mode in MODE_ORDER:
        ablation_rows.append(
            build_summary_row(
                mode,
                final_frames[mode],
                MODE_CONFIGS[mode],
                seed=display_seed,
                calibrated_birth_rate=birth_scales.get(mode, 1.0),
            )
        )
    ablation_df = pd.DataFrame(ablation_rows)
    ablation_df.to_csv(results_dir / "ablation_summary.csv", index=False)

    save_fig1_development_slices(
        full_timeline,
        full_final_step,
        timelines["random_development_matched"],
        results_dir / "Fig1_development_slices_self_vs_random.png",
        params,
    )

    nnd_map = {
        mode: nearest_neighbor_distances(frame_positions(final_frames[mode]))
        for mode in ["full_self_organized", "random_development_matched", "random_mask_matched"]
    }
    save_fig2_nnd_distribution_and_cv(summary_df, nnd_map, results_dir / "Fig2_nnd_distribution_and_cv.png")
    save_fig3_color_composition(metrics_df, results_dir / "Fig3_color_composition_over_time.png")

    scan_df = run_parameter_scan(params)
    scan_df.to_csv(results_dir / "parameter_scan.csv", index=False)
    save_fig4_parameter_heatmap(scan_df, results_dir / "Fig4_parameter_phase_heatmap.png", params)
    save_fig5_pair_correlation(
        frame_positions(final_frames["full_self_organized"]),
        frame_positions(final_frames["random_mask_matched"]),
        results_dir / "Fig5_pair_correlation.png",
    )
    save_fig6_ablation_summary(ablation_df, results_dir / "Fig6_ablation_summary.png")
    save_fig7_ablation_final_patterns(final_frames, results_dir / "Fig7_ablation_final_patterns.png", params)
    write_model_math_summary(results_dir / "model_math_summary.md", params, summary_df, ablation_df, birth_scales)

    best_cv_row = scan_df.loc[scan_df["final_cv_nnd"].idxmin()]
    best_order_row = scan_df.loc[scan_df["order_score"].idxmax()]

    print(
        f"full_self_organized: N={full_final_frame['pigment_count']}, CV_NND={full_final_frame['nnd_cv']:.6f}, "
        f"Y/R/B={full_final_frame['yellow_count']}/{full_final_frame['red_count']}/{full_final_frame['black_count']}"
    )
    rand_frame = final_frames["random_development_matched"]
    print(
        f"random_development_matched: N={rand_frame['pigment_count']}, CV_NND={rand_frame['nnd_cv']:.6f}, "
        f"Y/R/B={rand_frame['yellow_count']}/{rand_frame['red_count']}/{rand_frame['black_count']}"
    )
    rand_mask_frame = final_frames["random_mask_matched"]
    print(
        f"random_mask_matched: N={rand_mask_frame['pigment_count']}, CV_NND={rand_mask_frame['nnd_cv']:.6f}"
    )
    for mode in ["no_repulsion", "no_gap_birth", "no_growth_displacement"]:
        frame = final_frames[mode]
        print(f"{mode}: N={frame['pigment_count']}, CV_NND={frame['nnd_cv']:.6f}")
    print(f"random-development calibrated birth_rate scale={birth_scales['random_development_matched']:.6f}")
    print(f"no_repulsion calibrated birth_rate scale={birth_scales['no_repulsion']:.6f}")
    print(f"no_gap_birth calibrated birth_rate scale={birth_scales['no_gap_birth']:.6f}")
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
