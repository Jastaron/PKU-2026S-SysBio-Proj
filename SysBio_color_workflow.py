# %% imports / config
from CuttleFishModel.model import PARAMS
from CuttleFishModel.workflow import run_intercalated_gif, run_report_analysis


# %% inspect CA model parameters
def print_default_config() -> None:
    for key in ["seed", "grid_size", "n_steps", "yellow_duration", "red_duration", "absolute_min_distance", "target_gap_to_black"]:
        print(f"{key}={PARAMS[key]}")


# %% run full self-organized GIF
def run_gif_section() -> None:
    run_intercalated_gif()


# %% run matched random controls and ablations
def run_analysis_section() -> None:
    run_report_analysis()


# %% generate statistics and figures
def main() -> None:
    print_default_config()
    run_analysis_section()


# %% script entrypoint
if __name__ == "__main__":
    main()
