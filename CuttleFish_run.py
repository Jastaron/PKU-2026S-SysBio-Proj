# %% Notebook-style entrypoint
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
NOTEBOOK_PATH = PROJECT_ROOT / "CuttleFish_run.ipynb"


# %% Usage note
def print_usage() -> None:
    print("Use CuttleFish_run.ipynb for the interactive workflow.")
    print(f"Notebook path: {NOTEBOOK_PATH}")
    print("This .py file is only a lightweight percent-cell companion.")
    print("It does not run the full pipeline by default.")


# %% Script entrypoint
if __name__ == "__main__":
    print_usage()

