from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .core import Frame


def pigment_points(frame: Frame) -> np.ndarray:
    if not frame.pigments:
        return np.empty((0, 2), dtype=float)
    return np.asarray([pigment.pos for pigment in frame.pigments], dtype=float)


def nearest_neighbor_distances(points: np.ndarray) -> np.ndarray:
    if len(points) < 2:
        return np.array([], dtype=float)
    diff = points[:, None, :] - points[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    return dist.min(axis=1)


def summarize_nnd(points: np.ndarray) -> tuple[float, float, float]:
    nnd = nearest_neighbor_distances(points)
    if len(nnd) == 0:
        return math.nan, math.nan, math.nan
    mean_val = float(np.mean(nnd))
    std_val = float(np.std(nnd))
    return mean_val, std_val, float(std_val / (mean_val + 1e-12))


def pair_correlation_like(points: np.ndarray, mask: np.ndarray | None = None, bins: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    del mask
    if bins is None:
        bins = np.linspace(0.0, 18.0, 31)
    if len(points) < 2:
        centers = 0.5 * (bins[:-1] + bins[1:])
        return centers, np.zeros(len(centers), dtype=float)

    diff = points[:, None, :] - points[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    iu = np.triu_indices(len(points), k=1)
    values = dist[iu]
    counts, _ = np.histogram(values, bins=bins)
    annulus_area = math.pi * (bins[1:] ** 2 - bins[:-1] ** 2)
    density = counts / (len(points) * annulus_area + 1e-12)
    centers = 0.5 * (bins[:-1] + bins[1:])
    return centers, density


def color_counts(frame: Frame) -> dict[str, int]:
    return {
        "yellow": frame.yellow_count,
        "red": frame.red_count,
        "black": frame.black_count,
    }


def timeline_to_dataframe(timeline: list[Frame]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "step": frame.step,
                "skin_area": frame.skin_area,
                "pigment_count": frame.pigment_count,
                "yellow_count": frame.yellow_count,
                "red_count": frame.red_count,
                "black_count": frame.black_count,
                "nnd_mean": frame.nnd_mean,
                "nnd_std": frame.nnd_std,
                "nnd_cv": frame.nnd_cv,
            }
            for frame in timeline
        ]
    )

