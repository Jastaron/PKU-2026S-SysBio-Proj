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


def knn_distances(points: np.ndarray, k: int = 1) -> np.ndarray | float:
    if len(points) < 2:
        return np.nan
    diff = points[:, None, :] - points[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    k = max(1, min(int(k), len(points) - 1))
    nearest = np.partition(dist, kth=k - 1, axis=1)[:, :k]
    return np.mean(nearest, axis=1)


def mean_knn_distance(points: np.ndarray, k: int = 1) -> float:
    distances = knn_distances(points, k=k)
    if isinstance(distances, float) and np.isnan(distances):
        return np.nan
    if distances.size == 0:
        return np.nan
    return float(np.mean(distances))


def std_knn_distance(points: np.ndarray, k: int = 1) -> float:
    distances = knn_distances(points, k=k)
    if isinstance(distances, float) and np.isnan(distances):
        return np.nan
    if distances.size == 0:
        return np.nan
    return float(np.std(distances))


def short_range_pair_density(bin_centers, density, cutoff=3.0):
    '''
    Compute the mean pair-density value within a short-distance window.

    Parameters
    ----------
    bin_centers : array-like
        Distance bin centers.
    density : array-like
        Pair-density values corresponding to bin_centers.
    cutoff : float
        Distances <= cutoff are treated as the short-range region.

    Returns
    -------
    float
        Mean pair density in the short-range window. Return np.nan if no bins are available.
    '''
    x = np.asarray(bin_centers, dtype=float)
    y = np.asarray(density, dtype=float)
    mask = x <= float(cutoff)
    if not np.any(mask):
        return np.nan
    return float(np.nanmean(y[mask]))


def first_pair_density_peak_distance(
    bin_centers,
    density,
    max_distance=10.0,
    smooth_window=3,
    min_rel_height=0.5,
):
    '''
    Estimate the first major peak distance of a pair-density curve.

    The curve is restricted to bin_centers <= max_distance and lightly smoothed.
    The function returns the first local maximum whose height is at least
    min_rel_height times the maximum smoothed density in the window.
    If no such local maximum is found, return the distance of the maximum
    smoothed density within the window.
    Return np.nan if the window is empty or all density values are NaN.
    '''
    x = np.asarray(bin_centers, dtype=float)
    y = np.asarray(density, dtype=float)

    mask = x <= float(max_distance)
    x = x[mask]
    y = y[mask]

    valid = ~np.isnan(y)
    x = x[valid]
    y = y[valid]

    if x.size == 0 or y.size == 0:
        return np.nan

    if int(smooth_window) <= 1:
        y_smooth = y.copy()
    else:
        window = max(int(smooth_window), 1)
        kernel = np.ones(window, dtype=float) / float(window)
        y_smooth = np.convolve(y, kernel, mode="same")

    if y_smooth.size == 0 or np.all(np.isnan(y_smooth)):
        return np.nan

    peak_threshold = float(min_rel_height) * float(np.nanmax(y_smooth))

    if y_smooth.size >= 3:
        for i in range(1, y_smooth.size - 1):
            if y_smooth[i] >= y_smooth[i - 1] and y_smooth[i] >= y_smooth[i + 1]:
                if y_smooth[i] >= peak_threshold:
                    return float(x[i])

    return float(x[int(np.nanargmax(y_smooth))])


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
