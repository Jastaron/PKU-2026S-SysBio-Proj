from pathlib import Path
import re
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from PIL import Image


EMPTY = 0
SKIN = 1
YELLOW = 2
RED = 3
BLACK = 4


DEFAULT_PARAMS = {
    "seed": 42,
    "grid_size": 100,
    "n_steps": 100,
    "initial_radius_x": 10,
    "initial_radius_y": 8,
    "initial_yellow_count": 7,
    "growth_rate": 0.98,
    "growth_power": 0.86,
    "growth_curve_exponent": 1.1,
    "radial_growth_softness": 1.0,
    "radial_growth_margin": 0.8,
    "late_growth_floor": 0.30,
    "final_growth_radius": 52.0,
    "min_skin_age_for_diff": 2,
    "candidate_sample_size": 4200,
    "base_birth_rate": 0.96,
    "random_birth_lambda": 1.1,
    "min_pigment_distance": 4.0,
    "target_spacing": 6.2,
    "spacing_sigma": 2.8,
    "softness": 1.0,
    "far_field_weight": 1.00,
    "far_field_softness": 4.0,
    "min_boundary_distance": 3.0,
    "boundary_softness": 1.0,
    "center_birth_sigma_fraction": 0.40,
    "center_birth_floor": 0.50,
    "max_new_pigments_per_step": 14,
    "birth_quota_base": 1.0,
    "birth_quota_skin_scale": 0.00060,
    "birth_quota_deficit_scale": 0.10,
    "target_area_per_pigment_factor": 0.72,
    "interstitial_push_rate": 0.030,
    "interstitial_push_interval": 1,
    "birth_push_radius": 10.0,
    "birth_push_strength": 0.0,
    "birth_push_falloff": 3.6,
    "birth_push_search_radius": 2,
    "birth_push_max_shift": 1.0,
    "birth_push_min_norm": 0.10,
    "birth_push_duration": 0,
    "pair_repulsion_strength": 1.1,
    "hardcore_repulsion_strength": 1.8,
    "movement_threshold": 0.12,
    "black_push_multiplier": 1.7,
    "red_push_multiplier": 1.2,
    "yellow_push_multiplier": 1.0,
    "y_to_r": 3,
    "r_to_b": 8,
}


STATE_LABELS = {
    EMPTY: "Blank",
    SKIN: "Skin",
    YELLOW: "Yellow",
    RED: "Red",
    BLACK: "Black",
}


MODE_LABELS = {
    "self_organized": "Self-organized",
    "ablation_no_growth_motion": "No growth-driven motion",
    "ablation_random_birth": "Random birth placement",
    "ablation_no_pair_repulsion": "No pair repulsion",
    "random_mask": "Random mask control",
}


MODE_CONFIGS = {
    "self_organized": {
        "spacing_aware_birth": True,
        "growth_motion": True,
        "pair_repulsion": True,
    },
    "ablation_no_growth_motion": {
        "spacing_aware_birth": True,
        "growth_motion": False,
        "pair_repulsion": True,
    },
    "ablation_random_birth": {
        "spacing_aware_birth": False,
        "growth_motion": True,
        "pair_repulsion": True,
    },
    "ablation_no_pair_repulsion": {
        "spacing_aware_birth": True,
        "growth_motion": True,
        "pair_repulsion": False,
    },
}


def setup_plot_style():
    """Use a cleaner figure style for report-ready outputs."""
    plt.rcParams.update(
        {
            "figure.facecolor": "#fcfaf4",
            "axes.facecolor": "#fcfaf4",
            "savefig.facecolor": "#fcfaf4",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.family": "DejaVu Sans",
            "axes.titleweight": "semibold",
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )


def build_colormap():
    """Create a muted biological palette that keeps pigment states distinct."""
    colors = [
        "#f6f1e9",
        "#d7e7d0",
        "#f2d36b",
        "#d4625a",
        "#202521",
    ]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, 5.5, 1), cmap.N)
    return cmap, norm, colors


def count_neighbors(mask):
    """Count occupied neighbors to grow only from existing tissue boundaries."""
    padded = np.pad(mask.astype(int), 1, mode="constant")
    neighbors = np.zeros_like(mask, dtype=int)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            neighbors += padded[1 + di : 1 + di + mask.shape[0], 1 + dj : 1 + dj + mask.shape[1]]
    return neighbors


def pairwise_distances(points_a, points_b):
    """Compute Euclidean distances between lattice coordinates."""
    if len(points_a) == 0 or len(points_b) == 0:
        return np.empty((len(points_a), len(points_b)))
    diff = points_a[:, None, :] - points_b[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def nearest_neighbor_distances(points):
    """Measure local spacing between pigment cells."""
    if len(points) < 2:
        return np.array([])
    dist = pairwise_distances(points, points)
    np.fill_diagonal(dist, np.inf)
    return dist.min(axis=1)


def compute_nnd_metrics(points):
    """Summarize how regular the pigment network is."""
    nnd = nearest_neighbor_distances(points)
    if len(nnd) == 0:
        return {"nnd_mean": np.nan, "nnd_std": np.nan, "nnd_cv": np.nan}
    mean_val = float(np.mean(nnd))
    std_val = float(np.std(nnd))
    return {
        "nnd_mean": mean_val,
        "nnd_std": std_val,
        "nnd_cv": float(std_val / (mean_val + 1e-12)),
    }


def summarize_nnd_array(points):
    """Create robust NND summary values even when there are too few pigment cells."""
    nnd = nearest_neighbor_distances(points)
    if len(nnd) == 0:
        return nnd, np.nan, np.nan, np.nan
    mean_val = float(np.mean(nnd))
    std_val = float(np.std(nnd))
    cv_val = float(std_val / (mean_val + 1e-12))
    return nnd, mean_val, std_val, cv_val


def initialize_grids(params, rng):
    """Seed a small central skin patch and a few widely spaced yellow chromatophores."""
    n = params["grid_size"]
    center = (n - 1) / 2.0
    yy, xx = np.indices((n, n))
    ellipse = (
        ((xx - center) / params["initial_radius_x"]) ** 2
        + ((yy - center) / params["initial_radius_y"]) ** 2
        <= 1.0
    )

    grid = np.zeros((n, n), dtype=np.int8)
    skin_age = -np.ones((n, n), dtype=np.int16)
    pigment_age = -np.ones((n, n), dtype=np.int16)

    grid[ellipse] = SKIN
    skin_age[ellipse] = 0

    ordinary_positions = np.argwhere(grid == SKIN)
    rng.shuffle(ordinary_positions)
    selected = []
    min_dist = max(params["min_pigment_distance"], 6.0)

    for pos in ordinary_positions:
        if len(selected) >= params["initial_yellow_count"]:
            break
        if not selected:
            selected.append(pos)
            continue
        existing = np.asarray(selected, dtype=float)
        if np.all(np.sqrt(np.sum((existing - pos) ** 2, axis=1)) >= min_dist):
            selected.append(pos)

    for row, col in selected:
        grid[row, col] = YELLOW
        pigment_age[row, col] = 0

    return grid, skin_age, pigment_age


def growth_schedule(params, step):
    """Set a target tissue radius that grows quickly early and slows down near adulthood."""
    total_steps = max(params["n_steps"] - 1, 1)
    progress = np.clip(step / total_steps, 0.0, 1.0)
    exponent = params["growth_curve_exponent"]
    eased = 1.0 - (1.0 - progress) ** exponent

    start_radius = 0.5 * (params["initial_radius_x"] + params["initial_radius_y"])
    final_radius = params.get("final_growth_radius", 0.5 * (params["grid_size"] - 1))
    target_radius = start_radius + (final_radius - start_radius) * eased
    growth_velocity = (1.0 - progress) ** max(exponent - 1.0, 0.0)
    return target_radius, growth_velocity


def grow_skin(grid, skin_age, params, rng, step):
    """Expand the tissue boundary so empty pixels only turn into skin next to existing tissue."""
    occupied_mask = grid != EMPTY
    empty_mask = grid == EMPTY
    neighbor_count = count_neighbors(occupied_mask)
    frontier = empty_mask & (neighbor_count > 0)

    target_radius, growth_velocity = growth_schedule(params, step)
    yy, xx = np.indices(grid.shape)
    center = 0.5 * (grid.shape[0] - 1)
    radial_distance = np.sqrt((xx - center) ** 2 + (yy - center) ** 2)
    radial_gate = 1.0 / (
        1.0 + np.exp((radial_distance - target_radius) / params["radial_growth_softness"])
    )
    radial_gate[radial_distance > target_radius + params["radial_growth_margin"]] = 0.0
    speed_scale = params["late_growth_floor"] + (1.0 - params["late_growth_floor"]) * growth_velocity

    prob = params["growth_rate"] * speed_scale * (neighbor_count / 8.0) ** params["growth_power"] * radial_gate
    prob = np.clip(prob, 0.0, 0.92)
    new_skin = frontier & (rng.random(grid.shape) < prob)

    grid[new_skin] = SKIN
    skin_age[new_skin] = 0
    return int(np.count_nonzero(new_skin))


def compute_boundary_positions(grid):
    """Identify interior tissue sites that touch blank space and thus define the skin edge."""
    occupied_mask = grid != EMPTY
    neighbor_count = count_neighbors(occupied_mask)
    boundary_mask = occupied_mask & (neighbor_count < 8)
    return np.argwhere(boundary_mask).astype(float)


def apply_interstitial_expansion(grid, skin_age, pigment_age, params, step):
    """Passively advect pigment cells outward as ordinary skin inserts between them during growth."""
    _, growth_velocity = growth_schedule(params, step)
    effective_push = params["interstitial_push_rate"] * growth_velocity
    if effective_push <= 1e-6:
        return 0

    pigment_positions = np.argwhere(grid >= YELLOW)
    if len(pigment_positions) == 0:
        return 0

    center = np.array([(grid.shape[0] - 1) / 2.0, (grid.shape[1] - 1) / 2.0], dtype=float)
    states = grid[pigment_positions[:, 0], pigment_positions[:, 1]].copy()
    ages = pigment_age[pigment_positions[:, 0], pigment_positions[:, 1]].copy()

    base_grid = grid.copy()
    base_pigment_age = pigment_age.copy()
    base_grid[pigment_positions[:, 0], pigment_positions[:, 1]] = SKIN
    base_pigment_age[pigment_positions[:, 0], pigment_positions[:, 1]] = -1
    skin_age[pigment_positions[:, 0], pigment_positions[:, 1]] = 0

    vectors = pigment_positions.astype(float) - center
    radii = np.sqrt(np.sum(vectors * vectors, axis=1))
    max_radius = max(float(radii.max()), 1.0)
    order = np.argsort(radii)[::-1]

    used = set()
    moved = 0
    search_offsets = []
    for radius in range(0, 4):
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if max(abs(dr), abs(dc)) != radius:
                    continue
                search_offsets.append((dr, dc))

    for idx in order:
        pos = pigment_positions[idx].astype(float)
        vec = pos - center
        radius = radii[idx]

        if radius < 1e-9:
            target = pos
        else:
            scale = 1.0 + effective_push * (0.45 + radius / max_radius)
            target = center + vec * scale

        target_int = np.rint(target).astype(int)
        if (
            target_int[0] < 0
            or target_int[0] >= grid.shape[0]
            or target_int[1] < 0
            or target_int[1] >= grid.shape[1]
        ):
            continue

        candidates = []
        for dr, dc in search_offsets:
            rr = int(target_int[0] + dr)
            cc = int(target_int[1] + dc)
            if rr < 0 or rr >= grid.shape[0] or cc < 0 or cc >= grid.shape[1]:
                continue
            if (rr, cc) in used:
                continue
            if base_grid[rr, cc] == EMPTY:
                continue
            candidate_radius = np.linalg.norm(np.array([rr, cc], dtype=float) - center)
            if candidate_radius + 0.2 < radius:
                continue
            if used:
                used_positions = np.asarray(list(used), dtype=float)
                used_dist = np.sqrt(np.sum((used_positions - np.array([rr, cc], dtype=float)) ** 2, axis=1))
                if np.any(used_dist < params["min_pigment_distance"] * 0.78):
                    continue
            score = np.linalg.norm(np.array([rr, cc], dtype=float) - target)
            candidates.append((score, rr, cc))

        if not candidates:
            continue
        else:
            _, rr, cc = min(candidates, key=lambda item: item[0])

        used.add((rr, cc))
        if rr != int(pos[0]) or cc != int(pos[1]):
            moved += 1
        base_grid[rr, cc] = states[idx]
        base_pigment_age[rr, cc] = ages[idx]

    grid[:, :] = base_grid
    pigment_age[:, :] = base_pigment_age
    return moved


def move_pigments_one_step(grid, skin_age, pigment_age, active_sources, params, pair_repulsion=True):
    """Relax local overcrowding so nearby pigment cells are gradually separated."""
    pigment_positions = np.argwhere(grid >= YELLOW)
    if len(pigment_positions) == 0:
        empty = np.empty((0, 2), dtype=int)
        return empty, empty

    positions = pigment_positions.astype(float)
    states = grid[pigment_positions[:, 0], pigment_positions[:, 1]].copy()
    ages = pigment_age[pigment_positions[:, 0], pigment_positions[:, 1]].copy()

    diff = positions[:, None, :] - positions[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    unit = diff / (dist[..., None] + 1e-9)

    if pair_repulsion:
        within_target = dist < params["target_spacing"]
        within_min = dist < params["min_pigment_distance"]
        repulsion = (
            params["pair_repulsion_strength"]
            * np.clip((params["target_spacing"] - dist) / params["target_spacing"], 0.0, None)
            * within_target
        )
        hardcore = (
            params["hardcore_repulsion_strength"]
            * np.clip((params["min_pigment_distance"] - dist) / max(params["min_pigment_distance"], 1e-6), 0.0, None)
            * within_min
        )
        total_vectors = np.sum(unit * (repulsion + hardcore)[..., None], axis=1)
    else:
        total_vectors = np.zeros_like(positions, dtype=float)

    state_scale = np.where(
        states == BLACK,
        params["black_push_multiplier"],
        np.where(states == RED, params["red_push_multiplier"], params["yellow_push_multiplier"]),
    )
    for source in active_sources:
        vec = positions - source["position"][None, :]
        d_src = np.sqrt(np.sum(vec * vec, axis=1))
        active = (d_src > 0.05) & (d_src < params["birth_push_radius"])
        if not np.any(active):
            continue
        direction = vec[active] / d_src[active, None]
        time_scale = source["ttl"] / max(source["initial_ttl"], 1)
        strength = (
            params["birth_push_strength"]
            * time_scale
            * np.exp(-(d_src[active] ** 2) / (2.0 * params["birth_push_falloff"] ** 2))
            * state_scale[active]
        )
        total_vectors[active] += direction * strength[:, None]

    norms = np.sqrt(np.sum(total_vectors * total_vectors, axis=1))
    priority = np.where(states == BLACK, 3.0, np.where(states == RED, 2.0, 1.0))
    order = np.argsort(priority * 1000.0 + norms)[::-1]

    base_grid = grid.copy()
    base_age = pigment_age.copy()
    base_grid[pigment_positions[:, 0], pigment_positions[:, 1]] = SKIN
    base_age[pigment_positions[:, 0], pigment_positions[:, 1]] = -1
    skin_age[pigment_positions[:, 0], pigment_positions[:, 1]] = 0

    used = set()
    moved_from = []
    moved_to = []

    for idx in order:
        pos = positions[idx]
        if norms[idx] < params["movement_threshold"]:
            primary = np.rint(pos).astype(int)
        else:
            step_vec = total_vectors[idx] / (norms[idx] + 1e-9)
            step_vec = np.clip(np.rint(step_vec), -1, 1).astype(int)
            primary = np.rint(pos).astype(int) + step_vec

        candidates = [primary]
        if norms[idx] >= params["movement_threshold"]:
            sign = np.clip(np.rint(total_vectors[idx]), -1, 1).astype(int)
            candidates.extend(
                [
                    np.rint(pos).astype(int) + np.array([sign[0], 0], dtype=int),
                    np.rint(pos).astype(int) + np.array([0, sign[1]], dtype=int),
                ]
            )
        candidates.append(np.rint(pos).astype(int))

        chosen = None
        best_score = -np.inf
        for candidate in candidates:
            rr = int(candidate[0])
            cc = int(candidate[1])
            if rr < 0 or rr >= grid.shape[0] or cc < 0 or cc >= grid.shape[1]:
                continue
            if not skin_age[rr, cc] >= 0:
                continue
            if (rr, cc) in used:
                continue
            candidate_point = np.array([rr, cc], dtype=float)
            if used:
                used_arr = np.asarray(list(used), dtype=float)
                d_used = np.sqrt(np.sum((used_arr - candidate_point) ** 2, axis=1))
                min_used = float(d_used.min())
            else:
                min_used = params["target_spacing"]
            score = min_used - 0.2 * np.linalg.norm(candidate_point - positions[idx])
            if score > best_score:
                best_score = score
                chosen = (rr, cc)

        if chosen is None:
            # boundary encounter: pigment exits the observed board
            continue

        rr, cc = chosen
        used.add((rr, cc))
        base_grid[rr, cc] = states[idx]
        base_age[rr, cc] = ages[idx]
        if rr != int(pos[0]) or cc != int(pos[1]):
            moved_from.append([int(pos[0]), int(pos[1])])
            moved_to.append([rr, cc])

    grid[:, :] = base_grid
    pigment_age[:, :] = base_age

    if moved_from:
        return np.asarray(moved_from, dtype=int), np.asarray(moved_to, dtype=int)
    empty = np.empty((0, 2), dtype=int)
    return empty, empty


def pigment_birth_probability(d_min, params):
    """Favor births in medium-sized gaps while strongly suppressing short-range crowding."""
    inhibit = 1.0 / (1.0 + np.exp(-(d_min - params["min_pigment_distance"]) / params["softness"]))
    spacing_window = np.exp(
        -((d_min - params["target_spacing"]) ** 2) / (2.0 * params["spacing_sigma"] ** 2)
    )
    far_field = params["far_field_weight"] / (
        1.0 + np.exp(-(d_min - params["target_spacing"]) / params["far_field_softness"])
    )
    return params["base_birth_rate"] * inhibit * (spacing_window + far_field)


def boundary_gate(d_boundary, params):
    """Suppress pigment births too close to the expanding tissue edge."""
    return 1.0 / (
        1.0 + np.exp(-(d_boundary - params["min_boundary_distance"]) / params["boundary_softness"])
    )


def center_birth_gate(sampled_positions, grid, params):
    """Bias differentiation toward the central interior of the growing skin patch."""
    center = np.array([(grid.shape[0] - 1) / 2.0, (grid.shape[1] - 1) / 2.0], dtype=float)
    d_center = np.sqrt(np.sum((sampled_positions - center) ** 2, axis=1))
    effective_radius = max(np.sqrt(np.count_nonzero(grid != EMPTY) / np.pi), 1.0)
    sigma = max(8.0, effective_radius * params["center_birth_sigma_fraction"])
    core = np.exp(-(d_center ** 2) / (2.0 * sigma ** 2))
    return params["center_birth_floor"] + (1.0 - params["center_birth_floor"]) * core


def place_new_pigments(grid, pigment_age, new_positions):
    """Insert newly differentiated pigment cells as yellow cells."""
    if len(new_positions) == 0:
        return
    rows = new_positions[:, 0]
    cols = new_positions[:, 1]
    grid[rows, cols] = YELLOW
    pigment_age[rows, cols] = 0


def update_birth_sources(active_sources, new_positions, params):
    """Add slow outward-pushing sources that persist for several developmental steps."""
    for pos in np.asarray(new_positions, dtype=int):
        active_sources.append(
            {
                "position": pos.astype(float),
                "ttl": params["birth_push_duration"],
                "initial_ttl": params["birth_push_duration"],
            }
        )


def decay_birth_sources(active_sources):
    """Age all active push sources and remove exhausted ones."""
    next_sources = []
    for source in active_sources:
        source["ttl"] -= 1
        if source["ttl"] > 0:
            next_sources.append(source)
    return next_sources


def differentiate_cells(grid, skin_age, params, rng, spacing_aware_birth=True):
    """Differentiate new yellow cells with or without spacing-aware birth selection."""
    eligible = (grid == SKIN) & (skin_age >= params["min_skin_age_for_diff"])
    candidate_positions = np.argwhere(eligible)
    if len(candidate_positions) == 0:
        return np.empty((0, 2), dtype=int)

    sample_size = min(params["candidate_sample_size"], len(candidate_positions))
    sampled_idx = rng.choice(len(candidate_positions), size=sample_size, replace=False)
    sampled = candidate_positions[sampled_idx].astype(float)

    pigment_positions = np.argwhere(grid >= YELLOW).astype(float)
    if len(pigment_positions) == 0:
        d_min = np.full(len(sampled), np.inf)
    else:
        d_min = pairwise_distances(sampled, pigment_positions).min(axis=1)

    boundary_positions = compute_boundary_positions(grid)
    if len(boundary_positions) == 0:
        d_boundary = np.full(len(sampled), np.inf)
    else:
        d_boundary = pairwise_distances(sampled, boundary_positions).min(axis=1)

    center_bias = center_birth_gate(sampled, grid, params)
    if spacing_aware_birth:
        birth_core = pigment_birth_probability(d_min, params)
    else:
        birth_core = np.full(len(sampled), params["base_birth_rate"])

    birth_prob = np.clip(
        birth_core * boundary_gate(d_boundary, params) * center_bias,
        0.0,
        0.88,
    )
    birth_prob[d_boundary < params["min_boundary_distance"]] = 0.0
    order = np.argsort(birth_prob + rng.uniform(0.0, 1e-6, size=len(sampled)))[::-1]

    selected = []
    occupied = pigment_positions.copy()
    current_pigment_count = len(pigment_positions)
    allowed_candidate_count = int(np.count_nonzero(eligible))
    target_area_per_pigment = params["target_area_per_pigment_factor"] * params["target_spacing"] ** 2
    target_pigment_count = allowed_candidate_count / max(target_area_per_pigment, 1e-6)
    pigment_deficit = max(0.0, target_pigment_count - current_pigment_count)
    dynamic_birth_quota = int(
        np.clip(
            np.ceil(
                params["birth_quota_base"]
                + params["birth_quota_skin_scale"] * allowed_candidate_count
                + params["birth_quota_deficit_scale"] * pigment_deficit
            ),
            1,
            params["max_new_pigments_per_step"],
        )
    )

    for idx in order:
        if len(selected) >= dynamic_birth_quota:
            break
        if rng.random() >= birth_prob[idx]:
            continue
        pos = sampled[idx]
        if spacing_aware_birth and len(occupied) > 0:
            dist_existing = np.sqrt(np.sum((occupied - pos) ** 2, axis=1))
            if np.any(dist_existing < params["min_pigment_distance"]):
                continue
        if spacing_aware_birth and selected:
            selected_arr = np.asarray(selected, dtype=float)
            dist_new = np.sqrt(np.sum((selected_arr - pos) ** 2, axis=1))
            if np.any(dist_new < params["min_pigment_distance"]):
                continue
        selected.append(pos.astype(int))

    if not selected:
        return np.empty((0, 2), dtype=int)
    return np.asarray(selected, dtype=int)


def mature_pigments(grid, pigment_age, params):
    """Advance yellow -> red -> black maturation while keeping differentiation history explicit."""
    pigment_mask = grid >= YELLOW
    pigment_age[pigment_mask] += 1

    yellow_mask = pigment_mask & (pigment_age < params["y_to_r"])
    red_mask = pigment_mask & (pigment_age >= params["y_to_r"]) & (pigment_age < params["r_to_b"])
    black_mask = pigment_mask & (pigment_age >= params["r_to_b"])

    grid[yellow_mask] = YELLOW
    grid[red_mask] = RED
    grid[black_mask] = BLACK


def extract_pigment_points(grid):
    """Return pigment-cell coordinates for spacing analyses."""
    return np.argwhere(grid >= YELLOW).astype(float)


def summarize_step(grid, step, mode):
    """Collect developmental metrics at one time step."""
    points = extract_pigment_points(grid)
    metrics = compute_nnd_metrics(points)
    metrics.update(
        {
            "mode": mode,
            "step": step,
            "skin_area": int(np.count_nonzero(grid != EMPTY)),
            "pigment_count": int(len(points)),
        }
    )
    return metrics


def simulate_growth_ca(params, mode="self_organized", snapshot_steps=None, frame_steps=None):
    """Run the growth CA with a chosen mechanism combination."""
    mode_config = MODE_CONFIGS[mode]
    init_rng = np.random.default_rng(params["seed"] + 1)
    growth_rng = np.random.default_rng(params["seed"] + 2)
    mode_offset = list(MODE_CONFIGS).index(mode) * 1000
    pigment_rng = np.random.default_rng(params["seed"] + 101 + mode_offset)

    grid, skin_age, pigment_age = initialize_grids(params, init_rng)
    history = []
    snapshots = {}
    frames = {}
    active_sources = []

    snapshot_set = set(snapshot_steps or [])
    frame_set = set(frame_steps or [])

    empty = np.empty((0, 2), dtype=int)
    history.append(summarize_step(grid, 0, mode))
    if 0 in snapshot_set:
        snapshots[0] = grid.copy()
    if 0 in frame_set:
        frames[0] = {
            "grid": grid.copy(),
            "births": empty,
            "moves_from": empty,
            "moves_to": empty,
        }

    for step in range(1, params["n_steps"]):
        occupied_mask = grid != EMPTY
        skin_age[occupied_mask] += 1

        grow_skin(grid, skin_age, params, growth_rng, step)
        if (
            mode_config["growth_motion"]
            and params["interstitial_push_interval"] > 0
            and step % params["interstitial_push_interval"] == 0
        ):
            apply_interstitial_expansion(grid, skin_age, pigment_age, params, step)

        new_births = differentiate_cells(
            grid,
            skin_age,
            params,
            pigment_rng,
            spacing_aware_birth=mode_config["spacing_aware_birth"],
        )

        place_new_pigments(grid, pigment_age, new_births)
        moved_from, moved_to = move_pigments_one_step(
            grid,
            skin_age,
            pigment_age,
            active_sources,
            params,
            pair_repulsion=mode_config["pair_repulsion"],
        )
        active_sources = decay_birth_sources(active_sources)

        mature_pigments(grid, pigment_age, params)
        history.append(summarize_step(grid, step, mode))

        if step in snapshot_set:
            snapshots[step] = grid.copy()
        if step in frame_set:
            frames[step] = {
                "grid": grid.copy(),
                "births": new_births.copy(),
                "moves_from": moved_from.copy(),
                "moves_to": moved_to.copy(),
            }

    if params["n_steps"] - 1 not in snapshots:
        snapshots[params["n_steps"] - 1] = grid.copy()
    if params["n_steps"] - 1 not in frames:
        frames[params["n_steps"] - 1] = {
            "grid": grid.copy(),
            "births": empty,
            "moves_from": empty,
            "moves_to": empty,
        }

    return grid, pd.DataFrame(history), snapshots, frames


def sample_random_mask_control(final_grid, n_pigments, rng):
    """Place pigment cells uniformly in the final skin region as a strict random spatial baseline."""
    skin_sites = np.argwhere(final_grid != EMPTY)
    if n_pigments > len(skin_sites):
        raise ValueError("Random mask control cannot place more pigment cells than available skin sites.")
    choice = rng.choice(len(skin_sites), size=n_pigments, replace=False)
    return skin_sites[choice].astype(float)


def radial_pair_curve(points, bin_edges):
    """Compute a pair-count curve to visualize short-range exclusion."""
    if len(points) < 2:
        centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        return centers, np.zeros(len(bin_edges) - 1)
    dist = pairwise_distances(points, points)
    upper = dist[np.triu_indices(len(points), k=1)]
    hist, edges = np.histogram(upper, bins=bin_edges)
    shell_area = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
    density_like = hist / (shell_area + 1e-12)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, density_like


def run_parameter_scan(base_params):
    """Scan spacing parameters to see which local rules give the lowest final CV_NND."""
    scan_rows = []
    min_dist_values = [6.5, 7.5, 8.5]
    target_spacing_values = [10.0, 11.0, 12.0]
    replicates = 2

    for min_dist in min_dist_values:
        for target_spacing in target_spacing_values:
            for rep in range(replicates):
                scan_params = dict(base_params)
                scan_params.update(
                    {
                        "seed": 1000 + 100 * rep + int(min_dist * 10) + int(target_spacing * 10),
                        "grid_size": 100,
                        "n_steps": 100,
                        "candidate_sample_size": 360,
                        "min_pigment_distance": min_dist,
                        "target_spacing": target_spacing,
                    }
                )
                final_grid, history, _, _ = simulate_growth_ca(scan_params, mode="self_organized")
                final_points = extract_pigment_points(final_grid)
                final_metrics = compute_nnd_metrics(final_points)
                scan_rows.append(
                    {
                        "min_pigment_distance": min_dist,
                        "target_spacing": target_spacing,
                        "replicate": rep,
                        "pigment_count": len(final_points),
                        "skin_area": int(np.count_nonzero(final_grid != EMPTY)),
                        "CV_NND": final_metrics["nnd_cv"],
                        "mean_NND": final_metrics["nnd_mean"],
                        "terminal_step": int(history["step"].iloc[-1]),
                    }
                )

    scan_df = pd.DataFrame(scan_rows)
    scan_summary = (
        scan_df.groupby(["min_pigment_distance", "target_spacing"], as_index=False)
        .agg(
            mean_cv_nnd=("CV_NND", "mean"),
            std_cv_nnd=("CV_NND", "std"),
            mean_pigment_count=("pigment_count", "mean"),
            mean_skin_area=("skin_area", "mean"),
        )
        .sort_values(["mean_cv_nnd", "mean_pigment_count"], ascending=[True, False])
        .reset_index(drop=True)
    )
    return scan_df, scan_summary, scan_summary.iloc[0].to_dict()


def draw_grid(ax, grid, title="", subtitle=""):
    """Render a grid panel with clean styling."""
    cmap, norm, _ = build_colormap()
    ax.imshow(grid, cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_title(title, fontsize=11, pad=8)
    if subtitle:
        ax.text(
            0.5,
            -0.06,
            subtitle,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=9,
            color="#4f5b50",
        )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_snapshot_comparison(snapshot_by_mode, history_by_mode, ordered_modes, output_path):
    """Compare developmental trajectories across the self-organized model and single-factor ablations."""
    ordered_steps = sorted(next(iter(snapshot_by_mode.values())))
    fig, axes = plt.subplots(
        len(ordered_modes),
        len(ordered_steps),
        figsize=(3.2 * len(ordered_steps), 2.5 * len(ordered_modes) + 1.4),
        constrained_layout=True,
    )

    if len(ordered_modes) == 1:
        axes = np.asarray([axes])

    for row_idx, mode in enumerate(ordered_modes):
        hist = history_by_mode[mode].set_index("step")
        for col_idx, step in enumerate(ordered_steps):
            row = hist.loc[step]
            draw_grid(
                axes[row_idx, col_idx],
                snapshot_by_mode[mode][step],
                title=f"Step {step}",
                subtitle=f"skin={int(row['skin_area'])}, pigment={int(row['pigment_count'])}",
            )
        axes[row_idx, 0].text(
            -0.18,
            0.5,
            MODE_LABELS[mode],
            transform=axes[row_idx, 0].transAxes,
            rotation=90,
            va="center",
            ha="center",
            fontsize=11,
            color="#234232" if mode == "self_organized" else "#666666",
            weight="bold",
        )

    _, _, colors = build_colormap()
    handles = [
        plt.Line2D([0], [0], marker="s", linestyle="", markerfacecolor=colors[idx], markeredgecolor="none", markersize=10)
        for idx in range(5)
    ]
    fig.legend(handles, [STATE_LABELS[idx] for idx in range(5)], loc="upper center", ncol=5, frameon=False)
    fig.suptitle("Developmental comparison across self-organized and single-factor ablations", fontsize=15, y=1.02)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_final_pattern(final_grid, output_path):
    """Highlight the final non-random but non-crystalline pigment network."""
    cmap, norm, colors = build_colormap()
    fig, ax = plt.subplots(figsize=(8.4, 8.4), constrained_layout=True)
    ax.imshow(np.where(final_grid != EMPTY, SKIN, EMPTY), cmap=cmap, norm=norm, interpolation="nearest")

    for state, color, size in [(YELLOW, colors[YELLOW], 18), (RED, colors[RED], 18), (BLACK, colors[BLACK], 20)]:
        pts = np.argwhere(final_grid == state)
        if len(pts) > 0:
            ax.scatter(pts[:, 1], pts[:, 0], s=size, c=color, edgecolors="none", alpha=0.95)

    ax.set_title("Final self-organized pigment network", fontsize=15)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.savefig(output_path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def plot_nnd_cv(history_by_mode, ordered_modes, output_path):
    """Track spacing regularity across self-organized and ablation scenarios."""
    fig, ax = plt.subplots(figsize=(8.6, 4.9), constrained_layout=True)
    color_map = {
        "self_organized": "#0f766e",
        "ablation_no_growth_motion": "#c17c2e",
        "ablation_random_birth": "#8b8b8b",
        "ablation_no_pair_repulsion": "#b3476b",
    }
    for mode in ordered_modes:
        linewidth = 2.5 if mode == "self_organized" else 2.0
        linestyle = "-" if mode == "self_organized" else "--"
        ax.plot(
            history_by_mode[mode]["step"],
            history_by_mode[mode]["nnd_cv"],
            color=color_map[mode],
            linewidth=linewidth,
            linestyle=linestyle,
            label=MODE_LABELS[mode],
        )
    ax.set_xlabel("Developmental step")
    ax.set_ylabel("CV of nearest-neighbor distance")
    ax.set_title("Which mechanism most strongly stabilizes pigment spacing?")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_random_vs_self(final_nnd_by_mode, summary_df, output_path):
    """Compare self-organized spacing with single-factor ablations and a random mask baseline."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    ordered_models = [model for model in summary_df["model"] if model != "random_mask"]
    available_arrays = [arr for arr in final_nnd_by_mode.values() if len(arr) > 0]
    upper = max((arr.max() for arr in available_arrays), default=1.0) * 1.05
    bins = np.linspace(0, upper, 24)
    color_map = {
        "self_organized": "#0f766e",
        "ablation_no_growth_motion": "#c17c2e",
        "ablation_random_birth": "#8b8b8b",
        "ablation_no_pair_repulsion": "#b3476b",
        "random_mask": "#c7a46a",
    }

    for mode in ordered_models:
        arr = final_nnd_by_mode.get(mode, np.array([]))
        if len(arr) == 0:
            continue
        alpha = 0.7 if mode == "self_organized" else 0.4
        axes[0].hist(arr, bins=bins, density=True, alpha=alpha, color=color_map[mode], label=MODE_LABELS[mode])
    if len(final_nnd_by_mode.get("random_mask", np.array([]))) > 0:
        axes[0].hist(
            final_nnd_by_mode["random_mask"],
            bins=bins,
            density=True,
            alpha=0.35,
            color=color_map["random_mask"],
            label=MODE_LABELS["random_mask"],
        )
    axes[0].set_xlabel("Nearest-neighbor distance")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Nearest-neighbor spacing distributions")
    axes[0].grid(alpha=0.2)
    axes[0].legend(frameon=False)

    axes[1].bar(
        summary_df["model"],
        summary_df["CV_NND"],
        color=[color_map[model] for model in summary_df["model"]],
        width=0.62,
    )
    axes[1].set_ylabel("CV of nearest-neighbor distance")
    axes[1].set_title("Final CV_NND across models")
    axes[1].grid(alpha=0.2, axis="y")
    axes[1].tick_params(axis="x", rotation=10)

    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_pair_curve(self_points, random_points, output_path):
    """Show that self-organization strongly suppresses very short pair distances."""
    bin_edges = np.linspace(0.0, 35.0, 36)
    r_self, g_self = radial_pair_curve(self_points, bin_edges)
    r_rand, g_rand = radial_pair_curve(random_points, bin_edges)

    fig, ax = plt.subplots(figsize=(8.6, 4.9), constrained_layout=True)
    ax.plot(r_rand, g_rand, color="#c7a46a", linewidth=2.0, label="random-mask")
    ax.plot(r_self, g_self, color="#c14953", linewidth=2.4, label="self-organized")
    ax.fill_between(r_self, g_self, color="#c14953", alpha=0.08)
    ax.set_xlabel("Distance r (grid units)")
    ax.set_ylabel("Pair-count density (a.u.)")
    ax.set_title("Short-distance dip reveals local pigment exclusion")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_parameter_scan(scan_summary, output_path):
    """Visualize how spacing parameters tune the final regularity."""
    pivot = scan_summary.pivot(index="min_pigment_distance", columns="target_spacing", values="mean_cv_nnd")
    fig, ax = plt.subplots(figsize=(7.8, 5.8), constrained_layout=True)
    im = ax.imshow(pivot.values, cmap="YlGn_r", aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([f"{val:.1f}" for val in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([f"{val:.1f}" for val in pivot.index])
    ax.set_xlabel("target_spacing")
    ax.set_ylabel("min_pigment_distance")
    ax.set_title("Parameter scan of final CV_NND")

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.values[i, j]:.3f}", ha="center", va="center", fontsize=10, color="#102a21")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean final CV_NND")
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def render_step_frame(frame_payload, metrics_row, mode_label, output_path, step):
    """Render one development frame for the PNG step archive and GIF."""
    fig, ax = plt.subplots(figsize=(5.8, 5.8), constrained_layout=True)
    grid = frame_payload["grid"]
    draw_grid(
        ax,
        grid,
        title=f"{mode_label} | step {int(step)}",
        subtitle=f"skin={int(metrics_row['skin_area'])} | pigment={int(metrics_row['pigment_count'])} | CV_NND={metrics_row['nnd_cv']:.3f}" if not np.isnan(metrics_row["nnd_cv"]) else f"skin={int(metrics_row['skin_area'])} | pigment={int(metrics_row['pigment_count'])}",
    )

    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_step_archive(frames, history, mode, frame_dir):
    """Save development snapshots as numbered PNG files for later inspection."""
    frame_dir.mkdir(parents=True, exist_ok=True)
    for old_file in frame_dir.glob("step_*.png"):
        old_file.unlink()

    history_by_step = history.set_index("step")
    saved_paths = []
    for step in sorted(frames):
        output_path = frame_dir / f"step_{step:04d}.png"
        render_step_frame(frames[step], history_by_step.loc[step], MODE_LABELS[mode], output_path, step)
        saved_paths.append(output_path)
    return saved_paths


def create_gif(frame_paths, output_path, duration_ms=180):
    """Assemble PNG frames into a GIF for the report appendix."""
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


def parse_run_index(path):
    """Extract the numeric run index from a versioned run directory name."""
    match = re.match(r"run_(\d+)", path.name)
    if match:
        return int(match.group(1))
    return -1


def prepare_versioned_run_dirs(base_dir):
    """Archive any old root-level outputs, then create a fresh versioned run folder."""
    runs_root = base_dir / "experiment_runs"
    runs_root.mkdir(exist_ok=True)

    legacy_items = [base_dir / "results_growth_CA", base_dir / "development_steps"]
    if any(item.exists() for item in legacy_items):
        legacy_dir = runs_root / "run_000_legacy_root_outputs"
        legacy_dir.mkdir(exist_ok=True)
        for item in legacy_items:
            if item.exists():
                destination = legacy_dir / item.name
                if destination.exists():
                    shutil.rmtree(destination) if destination.is_dir() else destination.unlink()
                shutil.move(str(item), str(destination))

    existing_runs = [path for path in runs_root.iterdir() if path.is_dir() and path.name.startswith("run_")]
    next_index = max((parse_run_index(path) for path in existing_runs), default=0) + 1
    run_dir = runs_root / f"run_{next_index:03d}_self_only_gif"
    output_dir = run_dir / "results_growth_CA"
    steps_dir = run_dir / "development_steps"
    output_dir.mkdir(parents=True, exist_ok=True)
    steps_dir.mkdir(parents=True, exist_ok=True)
    return runs_root, run_dir, output_dir, steps_dir


def write_markdown_report(output_path, params, history_by_mode, summary_df, scan_summary):
    """Generate a Chinese report focused on single-factor ablation comparisons."""
    final_self = history_by_mode["self_organized"].iloc[-1]
    self_row = summary_df.loc[summary_df["model"] == "self_organized"].iloc[0]
    no_growth_row = summary_df.loc[summary_df["model"] == "ablation_no_growth_motion"].iloc[0]
    random_birth_row = summary_df.loc[summary_df["model"] == "ablation_random_birth"].iloc[0]
    no_repulsion_row = summary_df.loc[summary_df["model"] == "ablation_no_pair_repulsion"].iloc[0]
    random_mask_row = summary_df.loc[summary_df["model"] == "random_mask"].iloc[0]
    best_row = scan_summary.iloc[0]

    text = f"""# 色素细胞发育自组织 toy model 报告

## 1. 问题与理论目标

### 1.1 这个模型要回答什么问题
本模型讨论的重点不是成年乌贼如何瞬时神经变色，而是更早期的发育问题：皮肤中的色素细胞网络是否可能仅通过局部规则，在组织持续生长的过程中，自发形成整体均匀、但又不是机械晶格的空间结构。

### 1.2 本模型的理论立场
这里的目标不是证明乌贼真实发育机制，而是验证一种理论可能性：**皮肤扩张 + 普通细胞填充 + 局部抑制性色素细胞分化**，是否已经足以产生均匀但非随机的色素细胞空间网络。

## 2. 模型为什么属于发育自组织模型

### 2.1 没有全局坐标蓝图
模型中没有预设“第几个色素细胞应该长在第几个绝对坐标”的规则。所有图案都来自局部条件，而不是来自全局模板。

### 2.2 只有局部规则
局部规则主要包括三类：
1. 皮肤只会从已有皮肤边界向外扩张。
2. 只有普通皮肤细胞才允许分化为新色素细胞。
3. 新色素细胞会受到已有色素细胞的短程抑制，并倾向于在较大的局部空隙中出现。

### 2.3 涌现性体现在哪里
单个格点只有简单状态转变：空白、普通皮肤、黄色、红色、黑色。但在时间推进后，整体会涌现出一种能够支持成年快速变色的“皮肤像素阵列”式空间网络。

## 3. 模型结构

### 3.1 状态定义
二维格点元胞自动机使用如下状态：
1. `0` = 空白区域 / 尚未长出的皮肤
2. `1` = 普通皮肤细胞
3. `2` = 黄色色素细胞
4. `3` = 红色色素细胞
5. `4` = 黑色色素细胞

### 3.2 皮肤生长如何实现
初始时只有中心一小块椭圆形皮肤存在。每个时间步，只有紧邻现有皮肤的空白格才有概率变为普通皮肤细胞，因此皮肤是从中心连续向外扩张，而不是在远处凭空生成。

### 3.3 为什么内部也会继续长出新的色素细胞
本次版本把“普通皮肤细胞间插生长”实现为**皮肤扩张驱动的被动外移**。随着整体皮肤面积变大，已有色素细胞会被普通皮肤细胞逐步拉开，尤其是原本靠近边缘的细胞，在发育后期仍然更接近边缘，而不是永远固定在最初坐标。这样一来，内部原先较密的区域会逐渐出现新的间隙，于是新的色素细胞可以继续在内部空隙中分化出来。

### 3.4 新色素细胞如何发生
只有普通皮肤细胞才允许分化。对于自组织模型，每个候选普通细胞都会计算：
1. 到最近已有色素细胞的距离。
2. 到皮肤边界的距离。

分化概率的逻辑是：
1. 距离已有色素细胞太近时，概率接近 0。
2. 距离接近目标间距时，概率最高。
3. 距离极大时仍保留小概率，用于填补大空隙。
4. 距离皮肤边界过近时，概率被显著压低，因此新色素细胞不会贴着新长出的外缘生成。

### 3.5 颜色成熟如何实现
新生色素细胞先以黄色出现，再随着 `pigment_age` 增长转为红色和黑色。本模型中的颜色成熟是发育标记，不代表成年乌贼的快速神经控制变色。

## 4. 单因素对照如何设置

### 4.1 三个独立条件
本次不再只用一个笼统的 random 对照，而是围绕三个独立条件做单因素消融：
1. 色素细胞是否会随皮肤生长而被缓慢外移。
2. 新色素细胞出生时是否会避免离已有色素细胞过近。
3. 色素细胞之间是否存在把过近邻居慢慢推开的短程斥力。

### 4.2 对照组定义
在其余两个条件保持与 self-organized 相同的前提下，分别关闭一个条件：
1. `ablation_no_growth_motion`
2. `ablation_random_birth`
3. `ablation_no_pair_repulsion`

另外仍保留一个 `random_mask` 终末空间对照，用来和最终皮肤区域中的纯随机放置做比较。

## 5. 关键结果

### 5.1 发育图像结果
主模型最终达到皮肤面积 **{int(final_self["skin_area"])}**、色素细胞数 **{int(final_self["pigment_count"])}**。这一版皮肤生长被显式约束为近圆形扩张，并且在后期逐渐放缓，因此终局皮肤面积接近一个半径约 50 的圆盘，而不是把整个 100x100 方形棋盘完全填满。四行快照对比可以直接看到：关闭不同条件以后，网络均匀性会以不同方式退化。

### 5.2 nearest-neighbor CV 说明了什么
最终最近邻距离变异系数为：
1. `self-organized`: `CV_NND={self_row["CV_NND"]:.3f}`
2. `ablation_no_growth_motion`: `CV_NND={no_growth_row["CV_NND"]:.3f}`
3. `ablation_random_birth`: `CV_NND={random_birth_row["CV_NND"]:.3f}`
4. `ablation_no_pair_repulsion`: `CV_NND={no_repulsion_row["CV_NND"]:.3f}`
5. `random_mask`: `CV_NND={random_mask_row["CV_NND"]:.3f}`

CV 越低，说明色素细胞之间的最近邻间距越稳定、越均匀。这个结果可以直接回答三个机制各自的重要性，而不是把所有差异混在一个 random 里。

### 5.3 pair-correlation-like 曲线说明了什么
`pair-correlation-like` 曲线中，自组织模型在极短距离处明显低于随机空间对照。这意味着近距离成对出现被强烈抑制，也就是新色素细胞不会在已有色素细胞附近任意堆积，而更倾向于在较大的间隙中插入。

### 5.4 参数扫描说明了什么
本次扫描比较了 `min_pigment_distance` 与 `target_spacing` 的组合。当前最优组合为：
1. `min_pigment_distance = {best_row["min_pigment_distance"]:.1f}`
2. `target_spacing = {best_row["target_spacing"]:.1f}`
3. `mean CV_NND = {best_row["mean_cv_nnd"]:.3f}`

这说明局部抑制尺度和偏好间距需要匹配。排斥太弱，色素细胞会过于接近；目标间距设定不合适，则会留下不必要的大孔洞。

## 6. 生物学解释

### 6.1 为什么这不是“随机撒点”
虽然模型里使用了随机数，但随机数只表示发育噪声和细胞分化机会。色素细胞并不是在平面上凭空被撒下去，而是必须满足：
1. 该位置已经长成皮肤。
2. 该位置先是普通皮肤细胞。
3. 该位置离边界不能太近。
4. 在 self-organized 中，还要满足与已有色素细胞的局部间距规则，并在之后接受皮肤增长驱动的被动拉开与短程斥力的整理。

### 6.2 为什么这个结构可以作为成年快速变色的基础
如果发育阶段已经先形成一个近均匀的色素细胞空间网络，那么成年后的神经控制只需要调节这些现成“像素”的开闭和扩张程度，就更容易在皮肤上快速生成条纹、斑点和动态伪装图案。

## 7. 模型限制

### 7.1 这不是对真实机制的证明
本模型不是乌贼真实发育的直接证据，也没有包含真实的细胞谱系、分子信号通路、机械力学或三维组织几何。

### 7.2 这是一个理论可行性检验
它的意义在于说明：即使没有全局定位蓝图，只要有局部生长、普通细胞填充、被动组织拉开和局部抑制性分化，就已经有可能产生一个适合作为成年动态变色基础设施的色素细胞网络。
"""
    output_path.write_text(text, encoding="utf-8")


def save_csv_tables(all_history, summary_df, scan_df, output_dir):
    """Persist quantitative outputs for the self-organized model and all ablations."""
    all_history.to_csv(output_dir / "metrics_over_time.csv", index=False)
    summary_df.to_csv(output_dir / "random_vs_self_summary.csv", index=False)
    scan_df.to_csv(output_dir / "parameter_scan.csv", index=False)


def main():
    setup_plot_style()

    base_dir = Path(__file__).resolve().parent
    runs_root = base_dir / "experiment_runs"
    run_dir = runs_root / "run_014_self_only_gif"
    output_dir = run_dir / "results_growth_CA"
    steps_dir = run_dir / "development_steps"
    runs_root.mkdir(exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    if steps_dir.exists():
        shutil.rmtree(steps_dir)
    steps_dir.mkdir(parents=True, exist_ok=True)

    params = dict(DEFAULT_PARAMS)
    snapshot_steps = [0, 20, 40, 60, 80, params["n_steps"] - 1]
    frame_steps = list(range(params["n_steps"]))
    ordered_modes = list(MODE_CONFIGS.keys())
    final_grids = {}
    histories = {}
    snapshots = {}
    frames = {}
    final_nnd_by_mode = {}
    summary_rows = []

    for mode in ordered_modes:
        final_grid, history, mode_snapshots, mode_frames = simulate_growth_ca(
            params,
            mode=mode,
            snapshot_steps=snapshot_steps,
            frame_steps=frame_steps,
        )
        final_grids[mode] = final_grid
        histories[mode] = history
        snapshots[mode] = mode_snapshots
        frames[mode] = mode_frames

        points = extract_pigment_points(final_grid)
        nnd, mean_val, std_val, cv_val = summarize_nnd_array(points)
        final_nnd_by_mode[mode] = nnd
        summary_rows.append(
            {
                "model": mode,
                "N": len(points),
                "mean_NND": mean_val,
                "std_NND": std_val,
                "CV_NND": cv_val,
            }
        )

    self_points = extract_pigment_points(final_grids["self_organized"])
    random_mask_points = sample_random_mask_control(
        final_grids["self_organized"],
        len(self_points),
        np.random.default_rng(params["seed"] + 999),
    )
    random_mask_nnd, random_mask_mean, random_mask_std, random_mask_cv = summarize_nnd_array(random_mask_points)
    final_nnd_by_mode["random_mask"] = random_mask_nnd
    summary_rows.append(
        {
            "model": "random_mask",
            "N": len(random_mask_points),
            "mean_NND": random_mask_mean,
            "std_NND": random_mask_std,
            "CV_NND": random_mask_cv,
        }
    )

    summary_df = pd.DataFrame(summary_rows)
    scan_df, scan_summary, best_scan = run_parameter_scan(params)

    plot_snapshot_comparison(
        snapshots,
        histories,
        ordered_modes,
        output_dir / "01_development_snapshots.png",
    )
    plot_final_pattern(final_grids["self_organized"], output_dir / "02_final_pattern.png")
    plot_nnd_cv(histories, ordered_modes, output_dir / "03_nearest_neighbor_cv.png")
    plot_random_vs_self(
        final_nnd_by_mode,
        summary_df,
        output_dir / "04_random_vs_self_organized_nnd.png",
    )
    plot_pair_curve(self_points, random_mask_points, output_dir / "05_pair_correlation.png")
    plot_parameter_scan(scan_summary, output_dir / "06_parameter_scan.png")

    self_frame_paths = build_step_archive(
        frames["self_organized"],
        histories["self_organized"],
        "self_organized",
        steps_dir / "self_organized",
    )
    random_birth_frame_paths = build_step_archive(
        frames["ablation_random_birth"],
        histories["ablation_random_birth"],
        "ablation_random_birth",
        steps_dir / "ablation_random_birth",
    )
    build_step_archive(
        frames["ablation_no_growth_motion"],
        histories["ablation_no_growth_motion"],
        "ablation_no_growth_motion",
        steps_dir / "ablation_no_growth_motion",
    )
    build_step_archive(
        frames["ablation_no_pair_repulsion"],
        histories["ablation_no_pair_repulsion"],
        "ablation_no_pair_repulsion",
        steps_dir / "ablation_no_pair_repulsion",
    )

    create_gif(self_frame_paths, output_dir / "07_self_organized_development.gif", duration_ms=160)
    create_gif(random_birth_frame_paths, output_dir / "08_ablation_random_birth.gif", duration_ms=160)
    create_gif(self_frame_paths, output_dir / "01_self_organized_development.gif", duration_ms=160)

    all_history = pd.concat([histories[mode] for mode in ordered_modes], ignore_index=True)
    save_csv_tables(all_history, summary_df, scan_df, output_dir)
    write_markdown_report(output_dir / "model_report.md", params, histories, summary_df, scan_summary)

    print("Final summary")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nBest parameter combination")
    print(pd.DataFrame([best_scan]).to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"\nRun directory: {run_dir}")
    print(f"Results saved to: {output_dir}")
    print(f"Step frames saved to: {steps_dir}")
    print(f"All versioned runs live under: {runs_root}")


if __name__ == "__main__":
    main()
