# PKU-2026S-SysBio-Proj

Cuttlefish pigment cell generation cellular automaton model.

## Structure

- `CuttleFishModel/core.py`: core cellular automaton model only
- `CuttleFishModel/metrics.py`: NND, pair-correlation, and timeline summaries
- `CuttleFishModel/controls.py`: matched random controls and ablations
- `CuttleFishModel/render.py`: matplotlib drawing helpers that draw on passed axes
- `CuttleFish_run.py`: notebook-style entry script with `# %%` cells

## Interactive Workflow

```bash
python CuttleFish_run.py
```

Use `CuttleFish_run.ipynb` as the primary workflow. Each figure has its own cell, and figure size, dpi, titles, bins, colors, legends, and save paths are all editable directly in the notebook.

`CuttleFish_run.py` is only a lightweight companion entrypoint that prints the notebook path and does not run the full pipeline.
