from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .core import BLACK, DEFAULT_PARAMS, CuttlefishCA, Frame, choose_final_step


MODE_ORDER = [
    "self",
    "no_repulsion",
    "no_gap_birth",
    "no_growth_displacement",
    "random_development_matched",
    "random_mask",
]

MODE_LABELS = {
    "self": "Full self-organized",
    "no_repulsion": "No repulsion",
    "no_gap_birth": "No gap-biased birth",
    "no_growth_displacement": "No growth displacement",
    "random_development_matched": "Random-development matched",
    "random_mask": "Random-mask matched",
}

MODE_COLORS = {
    "self": "#171411",
    "no_repulsion": "#355c7d",
    "no_gap_birth": "#f08a24",
    "no_growth_displacement": "#7b3294",
    "random_development_matched": "#c94b23",
    "random_mask": "#7c94b6",
}

DEFAULT_BIRTH_SCALE_GUESSES = {
    "random_development_matched": 0.0016,
    "no_repulsion": 0.0025,
    "no_gap_birth": 0.082,
}


def run_self(params: dict | None = None, seed: int | None = None) -> dict:
    return _run_mode_impl("self", params=params, seed=seed, birth_rate_scale=1.0, capture_timeline=True)


def run_random_development_matched(
    params: dict | None,
    target_N: int,
    seed: int,
    comparison_step: int | None = None,
) -> dict:
    matched = _run_matched_mode(
        "random_development_matched",
        params=params,
        target_N=target_N,
        seed=seed,
        comparison_step=comparison_step,
    )
    return matched


def run_random_mask(frame: Frame, target_N: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    skin_sites = np.argwhere(frame.skin > 0)
    n_points = min(target_N, len(skin_sites))
    chosen_idx = rng.choice(len(skin_sites), size=n_points, replace=False)
    chosen = skin_sites[chosen_idx].astype(float)
    jitter = rng.uniform(-0.45, 0.45, size=chosen.shape)
    positions = np.clip(chosen + jitter, 0.0, frame.skin.shape[0] - 1.0)

    pigments = [pigment.copy() for pigment in frame.pigments[:n_points]]
    perm = rng.permutation(len(positions))
    for pigment, pos in zip(pigments, positions[perm]):
        pigment.pos = pos.copy()

    random_frame = Frame(
        step=frame.step,
        skin=frame.skin.copy(),
        pigments=pigments,
        skin_area=frame.skin_area,
        pigment_count=len(pigments),
        yellow_count=frame.yellow_count if len(pigments) == frame.pigment_count else frame.yellow_count,
        red_count=frame.red_count,
        black_count=frame.black_count,
        nnd_mean=math.nan,
        nnd_std=math.nan,
        nnd_cv=math.nan,
    )
    random_frame = _rebuild_frame_metrics(random_frame)
    return {
        "mode": "random_mask",
        "seed": seed,
        "timeline": None,
        "final_step": random_frame.step,
        "final_frame": random_frame,
        "birth_rate_scale": math.nan,
        "switches": {
            "use_repulsion": False,
            "use_gap_birth": False,
            "use_growth_displacement": False,
        },
    }


def run_ablation(
    params: dict | None,
    mode: str,
    target_N: int | None = None,
    seed: int | None = None,
    comparison_step: int | None = None,
) -> dict:
    if mode == "random_mask":
        raise ValueError("Use run_random_mask(frame, target_N, seed) for random_mask.")
    if mode == "self":
        return run_self(params=params, seed=seed)
    if mode == "random_development_matched":
        if target_N is None:
            raise ValueError("target_N is required for random_development_matched.")
        return run_random_development_matched(
            params=params,
            target_N=target_N,
            seed=seed if seed is not None else DEFAULT_PARAMS["seed"],
            comparison_step=comparison_step,
        )

    if mode in {"no_repulsion", "no_gap_birth"} and target_N is not None:
        return _run_matched_mode(
            mode,
            params=params,
            target_N=target_N,
            seed=seed if seed is not None else DEFAULT_PARAMS["seed"],
            comparison_step=comparison_step,
        )
    return _run_mode_impl(
        mode,
        params=params,
        seed=seed,
        birth_rate_scale=1.0,
        capture_timeline=True,
        comparison_step=comparison_step,
    )


def scan_parameter_landscape(
    params: dict | None,
    absolute_min_distance_values: list[float],
    target_gap_values: list[float],
    seeds: list[int],
) -> pd.DataFrame:
    base_params = dict(DEFAULT_PARAMS)
    if params:
        base_params.update(params)

    records: list[dict] = []
    for min_dist in absolute_min_distance_values:
        for gap in target_gap_values:
            for seed in seeds:
                scan_params = dict(base_params)
                scan_params["absolute_min_distance"] = min_dist
                scan_params["target_gap_to_black"] = gap
                scan_params["candidate_sample_size"] = min(scan_params["candidate_sample_size"], 900)
                scan_params["n_steps"] = 120
                scan_params["relax_iterations"] = 1
                result = run_self(scan_params, seed=seed)
                frame = result["final_frame"]
                y_frac, r_frac, b_frac = _color_fractions(frame)
                records.append(
                    {
                        "seed": seed,
                        "absolute_min_distance": min_dist,
                        "target_gap_to_black": gap,
                        "final_step": result["final_step"],
                        "final_cv_nnd": frame.nnd_cv,
                        "pigment_count": frame.pigment_count,
                        "yellow_fraction": y_frac,
                        "red_fraction": r_frac,
                        "black_fraction": b_frac,
                        "order_score": _order_score(frame, scan_params),
                    }
                )
    return pd.DataFrame.from_records(records)


def summarize_result(result: dict) -> dict:
    frame = result["final_frame"]
    y_frac, r_frac, b_frac = _color_fractions(frame)
    return {
        "model": result["mode"],
        "seed": result["seed"],
        "N": frame.pigment_count,
        "mean_NND": frame.nnd_mean,
        "std_NND": frame.nnd_std,
        "CV_NND": frame.nnd_cv,
        "yellow_count": frame.yellow_count,
        "red_count": frame.red_count,
        "black_count": frame.black_count,
        "yellow_fraction": y_frac,
        "red_fraction": r_frac,
        "black_fraction": b_frac,
        "use_repulsion": result.get("switches", {}).get("use_repulsion"),
        "use_gap_birth": result.get("switches", {}).get("use_gap_birth"),
        "use_growth_displacement": result.get("switches", {}).get("use_growth_displacement"),
        "calibrated_birth_rate": result["birth_rate_scale"],
        "knn_distance": math.nan,
        "knn_distance_std": math.nan,
        "short_range_pair_density": math.nan,
        "first_pair_density_peak_distance": math.nan,
    }


def _run_mode(
    mode: str,
    params: dict | None,
    seed: int | None,
    birth_rate_scale: float,
    comparison_step: int | None = None,
) -> dict:
    return _run_mode_impl(
        mode,
        params=params,
        seed=seed,
        birth_rate_scale=birth_rate_scale,
        capture_timeline=True,
        comparison_step=comparison_step,
    )


def _run_mode_impl(
    mode: str,
    params: dict | None,
    seed: int | None,
    birth_rate_scale: float,
    capture_timeline: bool,
    comparison_step: int | None = None,
) -> dict:
    model = CuttlefishCA(params=params, mode=mode, seed=seed, birth_rate_scale=birth_rate_scale)
    if capture_timeline:
        timeline = model.simulate()
        if comparison_step is None:
            final_step = choose_final_step(timeline, model.params)
        else:
            final_step = int(np.clip(comparison_step, 0, len(timeline) - 1))
        final_frame = timeline[final_step]
    else:
        model.initialize()
        while model.current_step < model.params["n_steps"] - 1:
            model._advance(record=False)
        final_frame = model.snapshot()
        timeline = None
        final_step = final_frame.step
    return {
        "mode": mode,
        "seed": model.params["seed"],
        "timeline": timeline,
        "final_step": final_step,
        "final_frame": final_frame,
        "birth_rate_scale": birth_rate_scale,
        "switches": model.switches,
        "params": model.params,
    }


def _run_matched_mode(
    mode: str,
    params: dict | None,
    target_N: int,
    seed: int,
    comparison_step: int | None = None,
) -> dict:
    base_params = dict(DEFAULT_PARAMS)
    if params:
        base_params.update(params)

    rough_scale = DEFAULT_BIRTH_SCALE_GUESSES.get(mode)
    if rough_scale is None:
        rough_scale = _coarse_birth_scale_search(base_params, mode, target_N, seed)
    matched_scale = _refine_birth_scale_search(
        base_params,
        mode,
        target_N,
        seed,
        rough_scale,
        comparison_step=comparison_step,
    )
    return _run_mode(
        mode,
        params=base_params,
        seed=seed,
        birth_rate_scale=matched_scale,
        comparison_step=comparison_step,
    )


def _coarse_birth_scale_search(params: dict, mode: str, target_N: int, seed: int) -> float:
    candidates = np.geomspace(0.0005, 0.2, 10)
    best_scale = float(candidates[0])
    best_error = float("inf")

    for scale in candidates:
        result = _run_mode_impl(mode, params=params, seed=seed, birth_rate_scale=float(scale), capture_timeline=False)
        frame = result["final_frame"]
        error = abs(frame.pigment_count - target_N)
        if error < best_error:
            best_error = error
            best_scale = float(scale)
    return best_scale


def _refine_birth_scale_search(
    params: dict,
    mode: str,
    target_N: int,
    seed: int,
    initial_scale: float,
    comparison_step: int | None = None,
) -> float:
    scale = max(initial_scale, 0.0002)
    best_scale = scale
    best_score: tuple[float, float, float, float] | None = None

    for _ in range(3):
        multipliers = [0.60, 0.75, 0.90, 1.00, 1.10, 1.25, 1.50]
        candidate_scales = sorted({max(0.0002, scale * m) for m in multipliers})
        best_local_scale = scale

        for candidate_scale in candidate_scales:
            result = _run_mode_impl(
                mode,
                params=params,
                seed=seed,
                birth_rate_scale=float(candidate_scale),
                capture_timeline=True,
                comparison_step=comparison_step,
            )
            frame = result["final_frame"]
            rel_error = abs(frame.pigment_count - target_N) / max(target_N, 1)
            y_frac, r_frac, b_frac = _color_fractions(frame)
            overfill_penalty = max(0.0, frame.pigment_count - 1.10 * target_N) / max(target_N, 1)
            score = (
                0.0 if rel_error <= 0.05 else 1.0,
                rel_error,
                r_frac,
                overfill_penalty - max(0.0, b_frac - y_frac),
            )
            if best_score is None or score < best_score:
                best_score = score
                best_scale = float(candidate_scale)
                best_local_scale = float(candidate_scale)
        scale = best_local_scale
        if best_score is not None and best_score[1] <= 0.05:
            break
    return best_scale


def _rebuild_frame_metrics(frame: Frame) -> Frame:
    points = np.asarray([pigment.pos for pigment in frame.pigments], dtype=float) if frame.pigments else np.empty((0, 2))
    nnd = _nearest_neighbor(points)
    if len(nnd) == 0:
        mean_val = math.nan
        std_val = math.nan
        cv_val = math.nan
    else:
        mean_val = float(np.mean(nnd))
        std_val = float(np.std(nnd))
        cv_val = float(std_val / (mean_val + 1e-12))
    return Frame(
        step=frame.step,
        skin=frame.skin,
        pigments=frame.pigments,
        skin_area=frame.skin_area,
        pigment_count=frame.pigment_count,
        yellow_count=frame.yellow_count,
        red_count=frame.red_count,
        black_count=frame.black_count,
        nnd_mean=mean_val,
        nnd_std=std_val,
        nnd_cv=cv_val,
    )


def _nearest_neighbor(points: np.ndarray) -> np.ndarray:
    if len(points) < 2:
        return np.array([], dtype=float)
    diff = points[:, None, :] - points[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    return dist.min(axis=1)


def _color_fractions(frame: Frame) -> tuple[float, float, float]:
    total = max(frame.pigment_count, 1)
    return (
        frame.yellow_count / total,
        frame.red_count / total,
        frame.black_count / total,
    )


def _order_score(frame: Frame, params: dict) -> float:
    y_frac, r_frac, b_frac = _color_fractions(frame)
    cv = frame.nnd_cv
    if not np.isfinite(cv) or cv <= 0:
        return 0.0
    expected_count = frame.skin_area / params["target_area_per_pigment"]
    count_ratio = frame.pigment_count / max(expected_count, 1e-6)
    count_score = float(np.exp(-((count_ratio - 1.0) ** 2) / 0.18))
    color_score = max(0.0, b_frac - 0.2 * y_frac) * np.clip(y_frac / 0.22, 0.0, 1.2) * np.exp(-10.0 * r_frac)
    if not (b_frac > y_frac > r_frac):
        color_score *= 0.6
    return float((1.0 / cv) * color_score * count_score)
