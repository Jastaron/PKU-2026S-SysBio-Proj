# %% Imports and config
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import shutil
import tempfile

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from CuttleFishModel.core import CuttlefishCA, DEFAULT_PARAMS, choose_final_step
from CuttleFishModel.metrics import (
    nearest_neighbor_distances,
    pair_correlation_like,
    pigment_points,
    summarize_nnd,
    timeline_to_dataframe,
)
from CuttleFishModel.controls import (
    MODE_COLORS,
    MODE_LABELS,
    run_ablation,
    run_random_development_matched,
    run_random_mask,
    run_self,
    scan_parameter_landscape,
    summarize_result,
)
from CuttleFishModel.render import (
    add_bar_value_labels,
    apply_plot_style,
    draw_color_composition,
    draw_cv_bar,
    draw_frame,
    draw_fraction_stack,
    draw_nnd_distribution,
    draw_pair_correlation,
    draw_parameter_heatmap,
    draw_timeline_slices,
)


apply_plot_style()

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

params = deepcopy(DEFAULT_PARAMS)


# %% Small helper functions
def frame_summary_text(frame) -> str:
    return (
        f"N={frame.pigment_count}, CV_NND={frame.nnd_cv:.6f}, "
        f"Y/R/B={frame.yellow_count}/{frame.red_count}/{frame.black_count}"
    )


def save_current_figure(output_path: Path, dpi: int = 220) -> None:
    plt.gcf().savefig(output_path, dpi=dpi, bbox_inches="tight")
    print(f"Saved: {output_path}")


def save_gif(timeline, final_step: int, params: dict, output_path: Path, frame_stride: int | None = None, duration_ms: int | None = None) -> None:
    stride = params["frame_stride"] if frame_stride is None else frame_stride
    duration = params["gif_duration_ms"] if duration_ms is None else duration_ms
    frame_steps = list(range(0, final_step + 1, stride))
    if frame_steps[-1] != final_step:
        frame_steps.append(final_step)

    with tempfile.TemporaryDirectory() as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        frame_paths = []
        for step in frame_steps:
            fig, ax = plt.subplots(figsize=(6.2, 6.2), constrained_layout=True)
            frame = timeline[step]
            title = (
                f"step {frame.step} | skin={frame.skin_area} | pigments={frame.pigment_count} | "
                f"Y={frame.yellow_count} R={frame.red_count} B={frame.black_count}"
            )
            draw_frame(ax, frame, params, title=title)
            frame_path = tmp_dir / f"frame_{step:04d}.png"
            fig.savefig(frame_path, dpi=180)
            plt.close(fig)
            frame_paths.append(frame_path)

        images = []
        for path in frame_paths:
            with Image.open(path) as img:
                images.append(img.convert("P", palette=Image.ADAPTIVE))
        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=0,
            disposal=2,
        )
    print(f"Saved GIF: {output_path}")


def build_summary_dataframe(results: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([summarize_result(result) for result in results])


# %% Run self-organized model
def run_self_model(local_params: dict | None = None):
    current_params = deepcopy(params if local_params is None else local_params)
    result = run_self(current_params, seed=current_params["seed"])
    frame = result["final_frame"]
    print(f"self final_step={result['final_step']}")
    print(f"self {frame_summary_text(frame)}")
    return result


# Example interactive use:
# self_result = run_self_model()


# %% Preview final frame
def preview_final_frame(result: dict, output_path: Path | None = None):
    fig, ax = plt.subplots(figsize=(6.2, 6.2), constrained_layout=True)
    frame = result["final_frame"]
    draw_frame(ax, frame, result["params"], title=f"step {result['final_step']} | {frame_summary_text(frame)}")
    if output_path is not None:
        fig.savefig(output_path, dpi=220, bbox_inches="tight")
        print(f"Saved: {output_path}")
    return fig, ax


# Example interactive use:
# fig, ax = preview_final_frame(self_result)


# %% Save GIF if needed
def generate_main_gif(result: dict, output_path: Path | None = None):
    if output_path is None:
        output_path = RESULTS_DIR / "01_intercalated_development.gif"
    save_gif(result["timeline"], result["final_step"], result["params"], output_path)


# Example interactive use:
# generate_main_gif(self_result)


# %% Run matched controls
def run_matched_controls(self_result: dict, local_params: dict | None = None):
    current_params = deepcopy(params if local_params is None else local_params)
    target_N = self_result["final_frame"].pigment_count
    seed = current_params["seed"]

    random_dev = run_random_development_matched(current_params, target_N=target_N, seed=seed)
    random_mask = run_random_mask(self_result["final_frame"], target_N=target_N, seed=seed + 400)

    print(f"random-development matched {frame_summary_text(random_dev['final_frame'])}")
    print(f"random-mask matched N={random_mask['final_frame'].pigment_count}, CV_NND={random_mask['final_frame'].nnd_cv:.6f}")
    print(f"random-development birth_rate_scale={random_dev['birth_rate_scale']:.6f}")
    return random_dev, random_mask


# Example interactive use:
# random_dev_result, random_mask_result = run_matched_controls(self_result)


# %% Run ablations
def run_ablation_set(self_result: dict, local_params: dict | None = None):
    current_params = deepcopy(params if local_params is None else local_params)
    target_N = self_result["final_frame"].pigment_count
    seed = current_params["seed"]

    no_repulsion = run_ablation(current_params, "no_repulsion", target_N=target_N, seed=seed)
    no_gap_birth = run_ablation(current_params, "no_gap_birth", target_N=target_N, seed=seed)
    no_growth_displacement = run_ablation(current_params, "no_growth_displacement", target_N=target_N, seed=seed)

    print(f"no_repulsion {frame_summary_text(no_repulsion['final_frame'])}")
    print(f"no_gap_birth {frame_summary_text(no_gap_birth['final_frame'])}")
    print(f"no_growth_displacement {frame_summary_text(no_growth_displacement['final_frame'])}")
    return no_repulsion, no_gap_birth, no_growth_displacement


# Example interactive use:
# no_repulsion_result, no_gap_birth_result, no_growth_displacement_result = run_ablation_set(self_result)


# %% Fig1 development slices
def make_fig1(self_result: dict, random_dev_result: dict, output_path: Path | None = None):
    if output_path is None:
        output_path = RESULTS_DIR / "Fig1_development_slices_self_vs_random.png"

    self_final = self_result["final_step"]
    shared_steps = sorted({int(round(x)) for x in pd.Series([0, self_final / 3, 2 * self_final / 3, self_final * 0.82])})
    while len(shared_steps) < 4:
        shared_steps.append(min(self_final, shared_steps[-1] + 1))
    self_steps = shared_steps[:4] + [self_final]
    random_steps = shared_steps[:4] + [random_dev_result["final_step"]]

    fig, axes = plt.subplots(2, 5, figsize=(19.0, 7.2), constrained_layout=True)
    draw_timeline_slices(axes[0], self_result["timeline"], self_steps, self_result["params"], row_label="Self-organized")
    draw_timeline_slices(
        axes[1],
        random_dev_result["timeline"],
        random_steps,
        random_dev_result["params"],
        row_label="Random-development matched",
    )
    fig.suptitle("Development Slices: Self-Organized vs Random-Development Matched", y=1.02, fontsize=17, fontweight="bold")
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    print(f"Saved: {output_path}")


# %% Fig2 NND distribution and CV
def make_fig2(self_result: dict, random_dev_result: dict, random_mask_result: dict, output_dir: Path | None = None):
    if output_dir is None:
        output_dir = RESULTS_DIR

    summary_df = build_summary_dataframe([self_result, random_dev_result, random_mask_result])
    summary_df.to_csv(output_dir / "random_vs_self_summary.csv", index=False)

    nnd_map = {
        "self": nearest_neighbor_distances(pigment_points(self_result["final_frame"])),
        "random_development_matched": nearest_neighbor_distances(pigment_points(random_dev_result["final_frame"])),
        "random_mask": nearest_neighbor_distances(pigment_points(random_mask_result["final_frame"])),
    }
    labels = {
        "self": MODE_LABELS["self"],
        "random_development_matched": MODE_LABELS["random_development_matched"],
        "random_mask": MODE_LABELS["random_mask"],
    }
    colors = {
        "self": MODE_COLORS["self"],
        "random_development_matched": MODE_COLORS["random_development_matched"],
        "random_mask": MODE_COLORS["random_mask"],
    }

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), constrained_layout=True)
    draw_nnd_distribution(axes[0], nnd_map, labels, colors)
    order = ["self", "random_development_matched", "random_mask"]
    bar_labels = ["self", "random-dev", "random-mask"]
    bar_colors = [colors[key] for key in order]
    draw_cv_bar(axes[1], summary_df, order, bar_labels, bar_colors)
    fig.savefig(output_dir / "Fig2_nnd_distribution_and_cv.png", dpi=220, bbox_inches="tight")
    print(f"Saved: {output_dir / 'Fig2_nnd_distribution_and_cv.png'}")
    return summary_df


# %% Fig3 color composition
def make_fig3(self_result: dict, output_dir: Path | None = None):
    if output_dir is None:
        output_dir = RESULTS_DIR

    metrics_df = timeline_to_dataframe(self_result["timeline"])
    metrics_df.to_csv(output_dir / "metrics_over_time.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2), constrained_layout=True)
    draw_color_composition(axes[0], metrics_df, fraction=False)
    draw_color_composition(axes[1], metrics_df, fraction=True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.03))
    fig.savefig(output_dir / "Fig3_color_composition_over_time.png", dpi=220, bbox_inches="tight")
    print(f"Saved: {output_dir / 'Fig3_color_composition_over_time.png'}")
    return metrics_df


# %% Fig4 parameter heatmap
def make_fig4(local_params: dict | None = None, output_dir: Path | None = None):
    if output_dir is None:
        output_dir = RESULTS_DIR
    current_params = deepcopy(params if local_params is None else local_params)
    scan_df = scan_parameter_landscape(
        current_params,
        absolute_min_distance_values=[1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
        target_gap_values=[3.0, 4.0, 5.0, 6.0, 7.0],
        seeds=[7, 17, 27],
    )
    scan_df.to_csv(output_dir / "parameter_scan.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.6), constrained_layout=True)
    image1 = draw_parameter_heatmap(axes[0], scan_df, "final_cv_nnd", cmap="magma_r", title="Mean Final CV_NND")
    cbar1 = fig.colorbar(image1, ax=axes[0], shrink=0.92)
    cbar1.set_label("CV_NND")
    image2 = draw_parameter_heatmap(axes[1], scan_df, "order_score", cmap="viridis", title="Mean Order Score")
    cbar2 = fig.colorbar(image2, ax=axes[1], shrink=0.92)
    cbar2.set_label("Order score")
    fig.suptitle("Local-rule Parameter Landscape of Spacing Order", y=1.02, fontsize=17, fontweight="bold")
    fig.savefig(output_dir / "Fig4_parameter_phase_heatmap.png", dpi=220, bbox_inches="tight")
    print(f"Saved: {output_dir / 'Fig4_parameter_phase_heatmap.png'}")
    return scan_df


# %% Fig5 pair correlation
def make_fig5(self_result: dict, random_mask_result: dict, output_path: Path | None = None):
    if output_path is None:
        output_path = RESULTS_DIR / "Fig5_pair_correlation.png"

    curve_map = {
        "self": pair_correlation_like(pigment_points(self_result["final_frame"])),
        "random_mask": pair_correlation_like(pigment_points(random_mask_result["final_frame"])),
    }
    labels = {
        "self": MODE_LABELS["self"],
        "random_mask": MODE_LABELS["random_mask"],
    }
    colors = {
        "self": MODE_COLORS["self"],
        "random_mask": MODE_COLORS["random_mask"],
    }
    fig, ax = plt.subplots(figsize=(8.0, 5.2), constrained_layout=True)
    draw_pair_correlation(ax, curve_map, labels, colors)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    print(f"Saved: {output_path}")


# %% Fig6 ablation summary
def make_fig6(self_result: dict, random_dev_result: dict, random_mask_result: dict, no_repulsion_result: dict, no_gap_birth_result: dict, no_growth_displacement_result: dict, output_dir: Path | None = None):
    if output_dir is None:
        output_dir = RESULTS_DIR

    results = [
        self_result,
        no_repulsion_result,
        no_gap_birth_result,
        no_growth_displacement_result,
        random_dev_result,
        random_mask_result,
    ]
    summary_df = build_summary_dataframe(results)
    summary_df.to_csv(output_dir / "ablation_summary.csv", index=False)

    order = [
        "self",
        "no_repulsion",
        "no_gap_birth",
        "no_growth_displacement",
        "random_development_matched",
        "random_mask",
    ]
    labels = ["full", "no-rep", "no-gap", "no-move", "rand-dev", "rand-mask"]
    colors = [MODE_COLORS[key] for key in order]

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.6), constrained_layout=True)
    draw_cv_bar(axes[0], summary_df, order, labels, colors, value_col="CV_NND", ylabel="CV_NND", title="CV_NND")
    draw_cv_bar(axes[1], summary_df, order, labels, colors, value_col="N", ylabel="N", title="Pigment count")
    axes[0].tick_params(axis="x", rotation=25)
    axes[1].tick_params(axis="x", rotation=25)
    draw_fraction_stack(axes[2], summary_df, order, labels)
    fig.suptitle("Ablation Summary", y=1.02, fontsize=17, fontweight="bold")
    fig.savefig(output_dir / "Fig6_ablation_summary.png", dpi=220, bbox_inches="tight")
    print(f"Saved: {output_dir / 'Fig6_ablation_summary.png'}")
    return summary_df


# %% Fig7 ablation final patterns
def make_fig7(self_result: dict, random_dev_result: dict, random_mask_result: dict, no_repulsion_result: dict, no_gap_birth_result: dict, no_growth_displacement_result: dict, output_path: Path | None = None):
    if output_path is None:
        output_path = RESULTS_DIR / "Fig7_ablation_final_patterns.png"

    results = [
        self_result,
        no_repulsion_result,
        no_gap_birth_result,
        no_growth_displacement_result,
        random_dev_result,
        random_mask_result,
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 8.0), constrained_layout=True)
    for ax, result in zip(axes.flat, results):
        frame = result["final_frame"]
        draw_frame(
            ax,
            frame,
            params,
            title=f"{MODE_LABELS[result['mode']]}\nN={frame.pigment_count} | CV={frame.nnd_cv:.3f}",
        )
    fig.suptitle("Final Patterns Across Mechanism Ablations", y=1.02, fontsize=17, fontweight="bold")
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    print(f"Saved: {output_path}")


# %% Lightweight smoke test
def main() -> None:
    print("CuttleFish notebook-style workflow")
    print("Run cells individually in VS Code / Jupyter to tweak figures.")
    print()
    print("Key imports are available:")
    print("from CuttleFishModel.core import CuttlefishCA, DEFAULT_PARAMS")
    print("from CuttleFishModel.metrics import summarize_nnd, timeline_to_dataframe")
    print("from CuttleFishModel.controls import run_random_development_matched, run_random_mask, run_ablation")
    print("from CuttleFishModel.render import draw_frame")
    print()

    self_result = run_self_model()
    random_dev_result, random_mask_result = run_matched_controls(self_result)
    no_gap_birth_result = run_ablation(params, "no_gap_birth", target_N=self_result["final_frame"].pigment_count, seed=params["seed"])

    preview_path = RESULTS_DIR / "final_frame_preview.png"
    fig, _ = preview_final_frame(self_result, output_path=preview_path)
    plt.close(fig)

    print()
    print(f"self final {frame_summary_text(self_result['final_frame'])}")
    print(
        f"random-development matched final N={random_dev_result['final_frame'].pigment_count}, "
        f"CV_NND={random_dev_result['final_frame'].nnd_cv:.6f}"
    )
    print(
        f"random-mask final N={random_mask_result['final_frame'].pigment_count}, "
        f"CV_NND={random_mask_result['final_frame'].nnd_cv:.6f}"
    )
    print(
        f"ablation no_gap_birth final N={no_gap_birth_result['final_frame'].pigment_count}, "
        f"CV_NND={no_gap_birth_result['final_frame'].nnd_cv:.6f}"
    )
    print()
    print(f"Preview saved to: {preview_path}")


if __name__ == "__main__":
    main()

