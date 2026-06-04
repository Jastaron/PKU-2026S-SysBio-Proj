# PKU-2026S-SysBio-Proj

Cuttlefish pigment cell generation cellular automaton model.

## Structure

- `CuttleFishModel/core.py`: core cellular automaton model only
- `CuttleFishModel/metrics.py`: NND, pair-correlation, and timeline summaries
- `CuttleFishModel/controls.py`: matched random controls and ablations
- `CuttleFishModel/render.py`: matplotlib drawing helpers that draw on passed axes
- `CuttleFish_run.py`: notebook-style entry script with `# %%` cells

## Run

```bash
python CuttleFish_run.py
```

This script is notebook-style. Running it from the terminal executes only a lightweight smoke test and saves one final-frame preview. In VS Code / Jupyter-style editors, run cells individually to generate the GIF, figures, and CSV files into `results/`.
