from __future__ import annotations

from pathlib import Path
import math
import shutil

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from PIL import Image


EMPTY = 0
SKIN = 1

YELLOW = 2
RED = 3
BLACK = 4


PARAMS = {
    "seed": 7,
    "grid_size": 100,
    "n_steps": 170,
    "initial_radius_x": 8.0,
    "initial_radius_y": 6.5,
    "initial_pigments": 4,
    "growth_rate": 0.94,
    "growth_power": 0.95,
    "growth_curve_exponent": 1.35,
    "radial_growth_softness": 1.2,
    "radial_growth_margin": 0.9,
    "late_growth_floor": 0.24,
    "final_growth_radius": 49.0,
    "min_skin_age_for_diff": 2,
    "candidate_sample_size": 2400,
    "base_birth_rate": 0.82,
    "birth_prob_cap": 0.78,
    "max_new_pigments_per_step": 10,
    "birth_quota_base": 1.0,
    "birth_quota_skin_scale": 0.0011,
    "birth_quota_deficit_scale": 0.08,
    "target_area_per_pigment": 28.0,
    "absolute_min_distance": 2.6,
    "all_spacing_softness": 0.9,
    "field_threshold": 0.92,
    "field_softness": 0.18,
    "target_gap_to_black": 5.0,
    "target_gap_sigma": 1.6,
    "bootstrap_gap_to_all": 5.5,
    "bootstrap_black_count": 18,
    "black_gap_weight": 1.55,
    "fallback_gap_weight": 1.15,
    "min_boundary_distance": 3.2,
    "boundary_softness": 0.9,
    "center_birth_floor": 0.28,
    "center_birth_sigma_fraction": 0.48,
    "yellow_duration": 28,
    "red_duration": 3,
    "black_growth_duration": 26,
    "inhibition_birth_radius": 8.6,
    "inhibition_mature_radius": 4.2,
    "inhibition_radius_decay": 0.18,
    "growth_motion_strength": 1.0,
    "relax_iterations": 2,
    "relax_strength": 0.38,
    "relax_buffer": 0.35,
    "size_start_scale": 0.52,
    "size_yellow_end_scale": 0.90,
    "size_red_end_scale": 1.05,
    "size_black_end_scale": 1.42,
    "size_softness_power": 0.78,
    "frame_stride": 2,
    "gif_duration_ms": 150,
    "render_scale": 8,
}


def pairwise_distances(points_a: np.ndarray, points_b: np.ndarray) -> np.ndarray:
    """Compute Euclidean distances between two sets of 2D points."""
    if len(points_a) == 0 or len(points_b) == 0:
        return np.empty((len(points_a), len(points_b)))
    diff = points_a[:, None, :] - points_b[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def nearest_neighbor_distances(points: np.ndarray) -> np.ndarray:
    """Return nearest-neighbour distances for a point cloud."""
    if len(points) < 2:
        return np.array([])
    dist = pairwise_distances(points, points)
    np.fill_diagonal(dist, np.inf)
    return dist.min(axis=1)


def summarize_nnd(points: np.ndarray) -> tuple[float, float, float]:
    """Summarize mean/std/CV of nearest-neighbour spacing."""
    nnd = nearest_neighbor_distances(points)
    if len(nnd) == 0:
        return math.nan, math.nan, math.nan
    mean_val = float(np.mean(nnd))
    std_val = float(np.std(nnd))
    return mean_val, std_val, float(std_val / (mean_val + 1e-12))


def pigment_stage(age: int, params: dict) -> int:
    """Map chromatophore age to yellow -> red -> black developmental colour."""
    if age < params["yellow_duration"]:
        return YELLOW
    if age < params["yellow_duration"] + params["red_duration"]:
        return RED
    return BLACK


def inhibition_radius(age: int, params: dict) -> float:
    """Age-dependent inhibitory surround: broad at birth, shrinking with age."""
    radius = params["inhibition_birth_radius"] - params["inhibition_radius_decay"] * age
    return max(params["inhibition_mature_radius"], radius)


def pigment_display_size(pigment: dict, params: dict) -> tuple[float, float]:
    """Return major/minor axis for rendering one chromatophore blob.

    Size grows continuously with age, so yellow chromatophores already enlarge,
    red is only a short bridge, and black keeps enlarging gradually instead of jumping.
    """
    age = pigment["age"]
    base_major = pigment["base_major"]
    base_minor = pigment["base_minor"]

    yellow_duration = max(params["yellow_duration"], 1)
    red_duration = max(params["red_duration"], 1)
    black_growth_duration = max(params["black_growth_duration"], 1)
    softness = params["size_softness_power"]

    if age < yellow_duration:
        progress = (age / yellow_duration) ** softness
        scale = params["size_start_scale"] + (params["size_yellow_end_scale"] - params["size_start_scale"]) * progress
    elif age < yellow_duration + red_duration:
        progress = ((age - yellow_duration) / red_duration) ** softness
        scale = params["size_yellow_end_scale"] + (params["size_red_end_scale"] - params["size_yellow_end_scale"]) * progress
    else:
        progress = min(1.0, (age - yellow_duration - red_duration) / black_growth_duration) ** softness
        scale = params["size_red_end_scale"] + (params["size_black_end_scale"] - params["size_red_end_scale"]) * progress
    return base_major * scale, base_minor * scale


def growth_schedule(params: dict, step: int) -> tuple[float, float]:
    """Near-circular skin growth that slows down toward later development."""
    total_steps = max(params["n_steps"] - 1, 1)
    progress = np.clip(step / total_steps, 0.0, 1.0)
    exponent = params["growth_curve_exponent"]
    eased = 1.0 - (1.0 - progress) ** exponent

    start_radius = 0.5 * (params["initial_radius_x"] + params["initial_radius_y"])
    final_radius = params["final_growth_radius"]
    target_radius = start_radius + (final_radius - start_radius) * eased
    growth_velocity = (1.0 - progress) ** max(exponent - 1.0, 0.0)
    return target_radius, growth_velocity


def initialize_skin(params: dict) -> tuple[np.ndarray, np.ndarray]:
    """Seed a small central juvenile skin patch on a larger blank field."""
    n = params["grid_size"]
    center = 0.5 * (n - 1)
    yy, xx = np.indices((n, n))
    ellipse = (
        ((xx - center) / params["initial_radius_x"]) ** 2
        + ((yy - center) / params["initial_radius_y"]) ** 2
        <= 1.0
    )
    skin = np.zeros((n, n), dtype=np.int8)
    skin_age = -np.ones((n, n), dtype=np.int16)
    skin[ellipse] = SKIN
    skin_age[ellipse] = 0
    return skin, skin_age


def initialize_pigments(skin: np.ndarray, params: dict, rng: np.random.Generator) -> list[dict]:
    """Create a few initial pale chromatophores to bootstrap later intercalation."""
    skin_sites = np.argwhere(skin == SKIN)
    rng.shuffle(skin_sites)
    pigments: list[dict] = []
    for row, col in skin_sites:
        pos = np.array([float(row), float(col)])
        if any(np.linalg.norm(pos - pigment["pos"]) < 6.5 for pigment in pigments):
            continue
        pigments.append(make_pigment(pos, params, rng, age=rng.integers(0, 8)))
        if len(pigments) >= params["initial_pigments"]:
            break
    return pigments


def make_pigment(pos: np.ndarray, params: dict, rng: np.random.Generator, age: int = 0) -> dict:
    """Instantiate one chromatophore with persistent shape parameters."""
    major = rng.uniform(1.8, 2.9)
    minor = major * rng.uniform(0.58, 0.92)
    return {
        "pos": pos.astype(float),
        "age": int(age),
        "base_major": float(major),
        "base_minor": float(minor),
        "angle": float(rng.uniform(0.0, 180.0)),
        "tone": float(rng.uniform(-0.08, 0.08)),
    }


def grow_skin(skin: np.ndarray, skin_age: np.ndarray, params: dict, rng: np.random.Generator, step: int) -> tuple[int, float]:
    """Grow only at the current tissue edge, but gate by a decelerating circular target radius."""
    occupied_mask = skin == SKIN
    empty_mask = skin == EMPTY
    padded = np.pad(occupied_mask.astype(int), 1, mode="constant")
    neighbors = np.zeros_like(occupied_mask, dtype=int)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            neighbors += padded[1 + di : 1 + di + skin.shape[0], 1 + dj : 1 + dj + skin.shape[1]]
    frontier = empty_mask & (neighbors > 0)

    target_radius, growth_velocity = growth_schedule(params, step)
    yy, xx = np.indices(skin.shape)
    center = 0.5 * (skin.shape[0] - 1)
    radial_distance = np.sqrt((xx - center) ** 2 + (yy - center) ** 2)
    radial_gate = 1.0 / (1.0 + np.exp((radial_distance - target_radius) / params["radial_growth_softness"]))
    radial_gate[radial_distance > target_radius + params["radial_growth_margin"]] = 0.0
    speed_scale = params["late_growth_floor"] + (1.0 - params["late_growth_floor"]) * growth_velocity

    prob = params["growth_rate"] * speed_scale * (neighbors / 8.0) ** params["growth_power"] * radial_gate
    prob = np.clip(prob, 0.0, 0.90)
    new_skin = frontier & (rng.random(skin.shape) < prob)
    skin[new_skin] = SKIN
    skin_age[new_skin] = 0
    return int(np.count_nonzero(new_skin)), target_radius


def move_with_skin_growth(pigments: list[dict], previous_radius: float, current_radius: float, params: dict, center: np.ndarray) -> None:
    """Advect chromatophores passively as the skin patch expands isotropically."""
    if previous_radius <= 1e-6:
        return
    growth_ratio = 1.0 + params["growth_motion_strength"] * (current_radius / previous_radius - 1.0)
    if abs(growth_ratio - 1.0) < 1e-7:
        return
    for pigment in pigments:
        pigment["pos"] = center + (pigment["pos"] - center) * growth_ratio


def project_inside_skin(pigments: list[dict], skin: np.ndarray, center: np.ndarray) -> None:
    """Keep chromatophores inside the currently grown skin patch."""
    n = skin.shape[0]
    for pigment in pigments:
        pos = pigment["pos"]
        for _ in range(12):
            rr = int(np.clip(round(pos[0]), 0, n - 1))
            cc = int(np.clip(round(pos[1]), 0, n - 1))
            if skin[rr, cc] == SKIN:
                break
            direction = center - pos
            norm = np.linalg.norm(direction)
            if norm < 1e-8:
                break
            pos = pos + 0.55 * direction / norm
        pigment["pos"] = np.clip(pos, 0.0, n - 1.0)


def relax_overlaps(pigments: list[dict], params: dict, center: np.ndarray) -> None:
    """Apply mild local repulsion only when chromatophores become unrealistically close."""
    if len(pigments) < 2:
        return
    for _ in range(params["relax_iterations"]):
        for i in range(len(pigments)):
            for j in range(i + 1, len(pigments)):
                pi = pigments[i]
                pj = pigments[j]
                delta = pj["pos"] - pi["pos"]
                dist = float(np.linalg.norm(delta))
                if dist < 1e-8:
                    direction = np.array([1.0, 0.0])
                else:
                    direction = delta / dist
                size_i = max(pigment_display_size(pi, params))
                size_j = max(pigment_display_size(pj, params))
                min_sep = 0.48 * (size_i + size_j) + params["relax_buffer"]
                if dist >= min_sep:
                    continue
                push = 0.5 * (min_sep - dist) * params["relax_strength"]
                pi["pos"] = pi["pos"] - push * direction
                pj["pos"] = pj["pos"] + push * direction

        for pigment in pigments:
            vec = pigment["pos"] - center
            radius = np.linalg.norm(vec)
            if radius > params["final_growth_radius"] + 1.3 and radius > 1e-8:
                pigment["pos"] = center + vec / radius * (params["final_growth_radius"] + 1.3)


def compute_boundary_positions(skin: np.ndarray) -> np.ndarray:
    """Return skin pixels touching blank space; births near them are penalized."""
    occupied = skin == SKIN
    padded = np.pad(occupied.astype(int), 1, mode="constant")
    neighbors = np.zeros_like(occupied, dtype=int)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            neighbors += padded[1 + di : 1 + di + occupied.shape[0], 1 + dj : 1 + dj + occupied.shape[1]]
    return np.argwhere(occupied & (neighbors < 8)).astype(float)


def age_inhibition_field(sampled: np.ndarray, pigments: list[dict], params: dict) -> tuple[np.ndarray, np.ndarray]:
    """Compute total inhibitory field and nearest distance to all chromatophores."""
    if not pigments:
        return np.zeros(len(sampled)), np.full(len(sampled), np.inf)
    positions = np.asarray([pigment["pos"] for pigment in pigments], dtype=float)
    dist = pairwise_distances(sampled, positions)
    radii = np.asarray([inhibition_radius(pigment["age"], params) for pigment in pigments], dtype=float)
    field = 1.0 / (1.0 + np.exp((dist - radii[None, :]) / params["all_spacing_softness"]))
    return field.sum(axis=1), dist.min(axis=1)


def center_birth_gate(sampled: np.ndarray, skin: np.ndarray, params: dict) -> np.ndarray:
    """Keep births broadly interior while still allowing them outside the core."""
    center = np.array([(skin.shape[0] - 1) / 2.0, (skin.shape[1] - 1) / 2.0], dtype=float)
    d_center = np.sqrt(np.sum((sampled - center) ** 2, axis=1))
    effective_radius = max(np.sqrt(np.count_nonzero(skin == SKIN) / np.pi), 1.0)
    sigma = max(8.0, effective_radius * params["center_birth_sigma_fraction"])
    core = np.exp(-(d_center ** 2) / (2.0 * sigma ** 2))
    return params["center_birth_floor"] + (1.0 - params["center_birth_floor"]) * core


def choose_new_births(
    skin: np.ndarray,
    skin_age: np.ndarray,
    pigments: list[dict],
    params: dict,
    rng: np.random.Generator,
) -> list[dict]:
    """Insert pale chromatophores into gaps, especially between mature black chromatophores."""
    eligible = (skin == SKIN) & (skin_age >= params["min_skin_age_for_diff"])
    candidate_positions = np.argwhere(eligible)
    if len(candidate_positions) == 0:
        return []

    sample_size = min(params["candidate_sample_size"], len(candidate_positions))
    sampled_idx = rng.choice(len(candidate_positions), size=sample_size, replace=False)
    sampled = candidate_positions[sampled_idx].astype(float)

    inhibition_field, d_all = age_inhibition_field(sampled, pigments, params)

    black_positions = np.asarray(
        [pigment["pos"] for pigment in pigments if pigment_stage(pigment["age"], params) == BLACK],
        dtype=float,
    )
    if len(black_positions) > 0:
        d_black = pairwise_distances(sampled, black_positions).min(axis=1)
    else:
        d_black = np.full(len(sampled), np.inf)

    boundary_positions = compute_boundary_positions(skin)
    if len(boundary_positions) > 0:
        d_boundary = pairwise_distances(sampled, boundary_positions).min(axis=1)
    else:
        d_boundary = np.full(len(sampled), np.inf)

    boundary_gate = 1.0 / (
        1.0 + np.exp(-(d_boundary - params["min_boundary_distance"]) / params["boundary_softness"])
    )
    center_gate = center_birth_gate(sampled, skin, params)
    inhibition_allow = 1.0 / (
        1.0 + np.exp((inhibition_field - params["field_threshold"]) / params["field_softness"])
    )

    if len(black_positions) >= params["bootstrap_black_count"]:
        gap_pref = np.exp(-((d_black - params["target_gap_to_black"]) ** 2) / (2.0 * params["target_gap_sigma"] ** 2))
        gap_pref *= 1.0 / (1.0 + np.exp(-(d_black - params["absolute_min_distance"]) / 0.7))
        gap_pref *= params["black_gap_weight"]
    else:
        gap_pref = np.exp(
            -((d_all - params["bootstrap_gap_to_all"]) ** 2) / (2.0 * (params["target_gap_sigma"] * 1.25) ** 2)
        )
        gap_pref *= 1.0 / (1.0 + np.exp(-(d_all - params["absolute_min_distance"]) / 0.8))
        gap_pref *= params["fallback_gap_weight"]

    birth_prob = np.clip(
        params["base_birth_rate"] * inhibition_allow * gap_pref * boundary_gate * center_gate,
        0.0,
        params["birth_prob_cap"],
    )
    birth_prob[d_all < params["absolute_min_distance"]] = 0.0
    birth_prob[d_boundary < params["min_boundary_distance"] * 0.55] = 0.0

    target_pigment_count = np.count_nonzero(eligible) / params["target_area_per_pigment"]
    pigment_deficit = max(0.0, target_pigment_count - len(pigments))
    dynamic_quota = int(
        np.clip(
            np.ceil(
                params["birth_quota_base"]
                + params["birth_quota_skin_scale"] * np.count_nonzero(eligible)
                + params["birth_quota_deficit_scale"] * pigment_deficit
            ),
            1,
            params["max_new_pigments_per_step"],
        )
    )

    order = np.argsort(birth_prob + rng.uniform(0.0, 1e-6, size=len(sampled)))[::-1]
    chosen: list[dict] = []
    existing_positions = [pigment["pos"] for pigment in pigments]
    for idx in order:
        if len(chosen) >= dynamic_quota:
            break
        if rng.random() >= birth_prob[idx]:
            continue
        pos = sampled[idx]
        if existing_positions:
            dist_existing = np.sqrt(np.sum((np.asarray(existing_positions) - pos) ** 2, axis=1))
            if np.any(dist_existing < params["absolute_min_distance"]):
                continue
        if chosen:
            chosen_positions = np.asarray([pigment["pos"] for pigment in chosen], dtype=float)
            dist_chosen = np.sqrt(np.sum((chosen_positions - pos) ** 2, axis=1))
            if np.any(dist_chosen < params["absolute_min_distance"]):
                continue
        chosen.append(make_pigment(pos.copy(), params, rng, age=0))
    return chosen


def compute_counts(pigments: list[dict], params: dict) -> dict:
    """Count chromatophores by developmental colour state."""
    yellow = 0
    red = 0
    black = 0
    for pigment in pigments:
        stage = pigment_stage(pigment["age"], params)
        if stage == YELLOW:
            yellow += 1
        elif stage == RED:
            red += 1
        else:
            black += 1
    return {"yellow": yellow, "red": red, "black": black}


def simulate(params: dict) -> tuple[list[dict], int]:
    """Run the developmental simulation and return per-step snapshots plus chosen final step."""
    rng_init = np.random.default_rng(params["seed"] + 1)
    rng_growth = np.random.default_rng(params["seed"] + 2)
    rng_birth = np.random.default_rng(params["seed"] + 3)

    skin, skin_age = initialize_skin(params)
    pigments = initialize_pigments(skin, params, rng_init)
    center = np.array([(params["grid_size"] - 1) / 2.0, (params["grid_size"] - 1) / 2.0], dtype=float)

    timeline: list[dict] = []
    previous_radius = 0.5 * (params["initial_radius_x"] + params["initial_radius_y"])

    for step in range(params["n_steps"]):
        if step > 0:
            occupied = skin == SKIN
            skin_age[occupied] += 1

            _, current_radius = grow_skin(skin, skin_age, params, rng_growth, step)
            move_with_skin_growth(pigments, previous_radius, current_radius, params, center)
            project_inside_skin(pigments, skin, center)
            newborn = choose_new_births(skin, skin_age, pigments, params, rng_birth)
            pigments.extend(newborn)
            relax_overlaps(pigments, params, center)
            project_inside_skin(pigments, skin, center)
            for pigment in pigments:
                pigment["age"] += 1
            previous_radius = current_radius

        points = np.asarray([pigment["pos"] for pigment in pigments], dtype=float)
        nnd_mean, nnd_std, nnd_cv = summarize_nnd(points)
        counts = compute_counts(pigments, params)
        timeline.append(
            {
                "step": step,
                "skin": skin.copy(),
                "pigments": [
                    {
                        "pos": pigment["pos"].copy(),
                        "age": pigment["age"],
                        "base_major": pigment["base_major"],
                        "base_minor": pigment["base_minor"],
                        "angle": pigment["angle"],
                        "tone": pigment["tone"],
                    }
                    for pigment in pigments
                ],
                "skin_area": int(np.count_nonzero(skin == SKIN)),
                "pigment_count": int(len(pigments)),
                "yellow_count": counts["yellow"],
                "red_count": counts["red"],
                "black_count": counts["black"],
                "nnd_mean": nnd_mean,
                "nnd_std": nnd_std,
                "nnd_cv": nnd_cv,
            }
        )

    final_step = choose_final_step(timeline, params)
    return timeline, final_step


def choose_final_step(timeline: list[dict], params: dict) -> int:
    """Pick a still-developing stage with many black cells, many yellow cells, and few red cells."""
    best_step = len(timeline) - 1
    best_score = -np.inf
    target_area = math.pi * params["final_growth_radius"] ** 2

    for frame in timeline[max(45, params["yellow_duration"]) :]:
        total = max(frame["pigment_count"], 1)
        yellow_frac = frame["yellow_count"] / total
        red_frac = frame["red_count"] / total
        black_frac = frame["black_count"] / total

        positions = np.asarray([pigment["pos"] for pigment in frame["pigments"]], dtype=float)
        black_positions = np.asarray(
            [pigment["pos"] for pigment in frame["pigments"] if pigment_stage(pigment["age"], params) == BLACK],
            dtype=float,
        )
        yellow_positions = np.asarray(
            [pigment["pos"] for pigment in frame["pigments"] if pigment_stage(pigment["age"], params) == YELLOW],
            dtype=float,
        )
        if len(black_positions) > 0 and len(yellow_positions) > 0:
            mix_dist = pairwise_distances(yellow_positions, black_positions).min(axis=1)
            mix_score = float(np.mean((mix_dist > 2.5) & (mix_dist < 7.0)))
        else:
            mix_score = 0.0

        area_score = 1.0 - abs(frame["skin_area"] - target_area) / target_area
        score = (
            3.0 * black_frac
            + 2.8 * min(yellow_frac, 0.32) / 0.32
            - 3.5 * abs(yellow_frac - 0.22)
            - 8.0 * red_frac
            + 1.8 * mix_score
            + 1.3 * area_score
            + 0.002 * frame["step"]
        )
        if frame["black_count"] > frame["yellow_count"] > frame["red_count"] and score > best_score:
            best_score = score
            best_step = frame["step"]

    return best_step


def make_skin_texture(n: int, rng: np.random.Generator) -> np.ndarray:
    """Create a CA-cell-aligned pale skin texture.

    The important point here is that each skin cell still lives on the 100x100 board.
    So the texture is per-cell, not a continuous free-form photograph-like wash.
    """
    base = np.ones((n, n, 3), dtype=float)
    base[:] = np.array([0.962, 0.955, 0.900])
    cell_noise = rng.normal(0.0, 0.018, size=(n, n, 1))
    warm_shift = rng.normal(0.0, 0.010, size=(n, n, 1))
    base[..., 0:1] += 0.55 * cell_noise + 0.35 * warm_shift
    base[..., 1:2] += 0.45 * cell_noise
    base[..., 2:3] += 0.30 * cell_noise - 0.20 * warm_shift
    return np.clip(base, 0.0, 1.0)


def pigment_color(pigment: dict, params: dict) -> tuple[float, float, float]:
    """Map developmental age to a paper-like chromatophore colour."""
    stage = pigment_stage(pigment["age"], params)
    tone = pigment["tone"]
    if stage == YELLOW:
        color = np.array([0.94, 0.78, 0.20])
    elif stage == RED:
        color = np.array([0.70, 0.22, 0.12])
    else:
        color = np.array([0.07, 0.06, 0.05])
    return tuple(np.clip(color + tone * np.array([0.40, 0.25, 0.10]), 0.0, 1.0))


def build_render_colormap() -> tuple[ListedColormap, BoundaryNorm]:
    """Discrete CA colours so skin and pigments are visibly on the same board."""
    colors = [
        "#f7f4ee",  # empty
        "#e7e3cc",  # skin
        "#ffcc33",  # yellow
        "#c94b23",  # red
        "#171411",  # black
    ]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, 5.5, 1), cmap.N)
    return cmap, norm


def rasterize_pigment_cells(pigment: dict, params: dict, grid_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize one chromatophore to explicit CA lattice cells.

    Each chromatophore has a centre coordinate, but its displayed body occupies
    multiple CA cells that expand gradually with age.
    """
    major, minor = pigment_display_size(pigment, params)
    half_h = max(1, int(math.ceil(major * 0.75)))
    half_w = max(1, int(math.ceil(major * 0.75)))
    rr_center = pigment["pos"][0]
    cc_center = pigment["pos"][1]
    angle = math.radians(pigment["angle"])
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    rows = []
    cols = []
    for rr in range(int(math.floor(rr_center - half_h)), int(math.ceil(rr_center + half_h)) + 1):
        if rr < 0 or rr >= grid_size:
            continue
        for cc in range(int(math.floor(cc_center - half_w)), int(math.ceil(cc_center + half_w)) + 1):
            if cc < 0 or cc >= grid_size:
                continue
            dy = rr - rr_center
            dx = cc - cc_center
            x_rot = cos_a * dx + sin_a * dy
            y_rot = -sin_a * dx + cos_a * dy
            value = (x_rot / max(major * 0.5, 1e-6)) ** 2 + (y_rot / max(minor * 0.5, 1e-6)) ** 2
            if value <= 1.0:
                rows.append(rr)
                cols.append(cc)

    if not rows:
        rr = int(np.clip(round(rr_center), 0, grid_size - 1))
        cc = int(np.clip(round(cc_center), 0, grid_size - 1))
        rows = [rr]
        cols = [cc]
    return np.asarray(rows, dtype=int), np.asarray(cols, dtype=int)


def compose_ca_render_grid(frame: dict, params: dict) -> tuple[np.ndarray, np.ndarray]:
    """Build explicit 100x100 CA state and owner maps for rendering."""
    skin = frame["skin"]
    state_grid = skin.copy()
    owner_grid = -np.ones_like(skin, dtype=int)
    score_grid = np.full(skin.shape, -np.inf, dtype=float)

    priority_map = {YELLOW: 1.0, RED: 2.0, BLACK: 3.0}
    for idx, pigment in enumerate(frame["pigments"]):
        stage = pigment_stage(pigment["age"], params)
        rows, cols = rasterize_pigment_cells(pigment, params, skin.shape[0])
        dist2 = (rows - pigment["pos"][0]) ** 2 + (cols - pigment["pos"][1]) ** 2
        local_score = priority_map[stage] * 100.0 - dist2 - 0.02 * pigment["age"]
        for rr, cc, score in zip(rows, cols, local_score):
            if skin[rr, cc] != SKIN:
                continue
            if score > score_grid[rr, cc]:
                state_grid[rr, cc] = stage
                owner_grid[rr, cc] = idx
                score_grid[rr, cc] = score
    return state_grid, owner_grid


def render_frame(frame: dict, params: dict, output_path: Path) -> None:
    """Render one frame strictly from the 100x100 CA state."""
    state_grid, _ = compose_ca_render_grid(frame, params)
    cmap, norm = build_render_colormap()
    fig, ax = plt.subplots(figsize=(6.2, 6.2), constrained_layout=True)
    ax.imshow(state_grid, cmap=cmap, norm=norm, interpolation="nearest")

    title = (
        f"step {frame['step']} | skin={frame['skin_area']} | pigments={frame['pigment_count']} | "
        f"Y={frame['yellow_count']} R={frame['red_count']} B={frame['black_count']}"
    )
    ax.set_title(title, fontsize=10, pad=8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.5, state_grid.shape[1] - 0.5)
    ax.set_ylim(state_grid.shape[0] - 0.5, -0.5)
    step = 5
    for k in range(0, state_grid.shape[0] + 1, step):
        ax.axhline(k - 0.5, color=(1.0, 1.0, 1.0, 0.18), linewidth=0.35)
        ax.axvline(k - 0.5, color=(1.0, 1.0, 1.0, 0.18), linewidth=0.35)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.savefig(output_path, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


def create_gif(frame_paths: list[Path], output_path: Path, duration_ms: int) -> None:
    """Assemble rendered PNG frames into a development GIF."""
    images = []
    for path in frame_paths:
        with Image.open(path) as img:
            images.append(img.convert("P", palette=Image.ADAPTIVE))
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
        disposal=2,
    )


def main() -> None:
    base_dir = Path(__file__).resolve().parent.parent
    result_dir = base_dir / "results"
    tmp_dir = result_dir / "_tmp_frames"
    result_dir.mkdir(parents=True, exist_ok=True)
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    params = dict(PARAMS)
    timeline, final_step = simulate(params)

    frame_steps = list(range(0, final_step + 1, params["frame_stride"]))
    if frame_steps[-1] != final_step:
        frame_steps.append(final_step)

    frame_paths: list[Path] = []
    for step in frame_steps:
        output_path = tmp_dir / f"frame_{step:04d}.png"
        render_frame(timeline[step], params, output_path)
        frame_paths.append(output_path)

    gif_path = result_dir / "01_intercalated_development.gif"
    if gif_path.exists():
        gif_path.unlink()
    create_gif(frame_paths, gif_path, params["gif_duration_ms"])
    shutil.rmtree(tmp_dir)

    final_frame = timeline[final_step]
    print("Intercalated self-organized run complete")
    print(
        f"final_step={final_step}, skin_area={final_frame['skin_area']}, pigment_count={final_frame['pigment_count']}, "
        f"yellow={final_frame['yellow_count']}, red={final_frame['red_count']}, black={final_frame['black_count']}, "
        f"CV_NND={final_frame['nnd_cv']:.6f}"
    )
    print(f"GIF saved to: {gif_path}")


if __name__ == "__main__":
    main()
