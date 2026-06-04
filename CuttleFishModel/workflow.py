from __future__ import annotations

from . import model
from .analysis import main as analysis_main


def run_intercalated_gif() -> None:
    """Run the main intercalated GIF workflow."""
    model.main()


def run_report_analysis() -> None:
    """Run the report analysis workflow."""
    analysis_main()
