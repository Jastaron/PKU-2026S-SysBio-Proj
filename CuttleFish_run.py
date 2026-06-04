# %% imports / config
from pathlib import Path

from CuttleFishModel.model import PARAMS
from CuttleFishModel.workflow import run_intercalated_gif, run_report_analysis


PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"


# %% inspect default model configuration
def print_default_config() -> None:
    keys = [
        "seed",
        "grid_size",
        "n_steps",
        "yellow_duration",
        "red_duration",
        "absolute_min_distance",
        "target_gap_to_black",
    ]
    for key in keys:
        print(f"{key}={PARAMS[key]}")


# %% generate intercalated CA GIF
def run_gif_section() -> None:
    run_intercalated_gif()


# %% run matched controls, ablations, statistics, and report figures
def run_report_section() -> None:
    run_report_analysis()


# %% inspect output directory
def print_result_paths() -> None:
    expected = [
        "01_intercalated_development.gif",
        "Fig1_development_slices_self_vs_random.png",
        "Fig2_nnd_distribution_and_cv.png",
        "Fig3_color_composition_over_time.png",
        "Fig4_parameter_phase_heatmap.png",
        "Fig5_pair_correlation.png",
        "Fig6_ablation_summary.png",
        "Fig7_ablation_final_patterns.png",
        "random_vs_self_summary.csv",
        "ablation_summary.csv",
        "metrics_over_time.csv",
        "parameter_scan.csv",
        "model_math_summary.md",
    ]
    for name in expected:
        print(RESULTS_DIR / name)


# %% unified workflow
def main() -> None:
    print_default_config()
    run_gif_section()
    run_report_section()
    print_result_paths()


# %% script entrypoint
if __name__ == "__main__":
    main()
