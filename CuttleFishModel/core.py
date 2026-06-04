from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


EMPTY = 0
SKIN = 1
YELLOW = 2
RED = 3
BLACK = 4


DEFAULT_PARAMS = {
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
    # Short-range exclusion between chromatophores.
    "absolute_min_distance": 2.6,
    "all_spacing_softness": 0.9,
    "field_threshold": 0.92,
    "field_softness": 0.18,
    # Preferred gap size relative to mature black chromatophores.
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
    # Age-dependent maturation.
    "yellow_duration": 28,
    "red_duration": 3,
    "black_growth_duration": 26,
    # Age-dependent inhibition field.
    "inhibition_birth_radius": 8.6,
    "inhibition_mature_radius": 4.2,
    "inhibition_radius_decay": 0.18,
    # Passive motion from tissue growth.
    "growth_motion_strength": 1.0,
    # Mild local relaxation after births / growth.
    "relax_iterations": 2,
    "relax_strength": 0.38,
    "relax_buffer": 0.35,
    # Display growth of one chromatophore footprint on the CA lattice.
    "size_start_scale": 0.52,
    "size_yellow_end_scale": 0.90,
    "size_red_end_scale": 1.05,
    "size_black_end_scale": 1.42,
    "size_softness_power": 0.78,
    "frame_stride": 2,
    "gif_duration_ms": 150,
    "render_scale": 8,
}

# Backward-compatible alias for notebook use.
PARAMS = DEFAULT_PARAMS


MODE_SWITCHES = {
    "self": {
        "use_repulsion": True,
        "use_gap_birth": True,
        "use_growth_displacement": True,
        "use_boundary_penalty": True,
    },
    "random_development_matched": {
        "use_repulsion": False,
        "use_gap_birth": False,
        "use_growth_displacement": True,
        "use_boundary_penalty": True,
    },
    "no_repulsion": {
        "use_repulsion": False,
        "use_gap_birth": True,
        "use_growth_displacement": True,
        "use_boundary_penalty": True,
    },
    "no_gap_birth": {
        "use_repulsion": True,
        "use_gap_birth": False,
        "use_growth_displacement": True,
        "use_boundary_penalty": True,
    },
    "no_growth_displacement": {
        "use_repulsion": True,
        "use_gap_birth": True,
        "use_growth_displacement": False,
        "use_boundary_penalty": True,
    },
}


@dataclass
class Pigment:
    pos: np.ndarray
    age: int
    base_major: float
    base_minor: float
    angle: float
    tone: float

    def copy(self) -> "Pigment":
        return Pigment(
            pos=self.pos.copy(),
            age=int(self.age),
            base_major=float(self.base_major),
            base_minor=float(self.base_minor),
            angle=float(self.angle),
            tone=float(self.tone),
        )

    def stage(self, params: dict) -> int:
        return pigment_stage(self.age, params)


@dataclass
class Frame:
    step: int
    skin: np.ndarray
    pigments: list[Pigment]
    skin_area: int
    pigment_count: int
    yellow_count: int
    red_count: int
    black_count: int
    nnd_mean: float
    nnd_std: float
    nnd_cv: float


def pigment_stage(age: int, params: dict) -> int:
    if age < params["yellow_duration"]:
        return YELLOW
    if age < params["yellow_duration"] + params["red_duration"]:
        return RED
    return BLACK


def pigment_display_size(pigment: Pigment, params: dict) -> tuple[float, float]:
    age = pigment.age
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
    return pigment.base_major * scale, pigment.base_minor * scale


def inhibition_radius(age: int, params: dict) -> float:
    radius = params["inhibition_birth_radius"] - params["inhibition_radius_decay"] * age
    return max(params["inhibition_mature_radius"], radius)


def choose_final_step(timeline: list[Frame], params: dict) -> int:
    best_step = len(timeline) - 1
    best_score = -np.inf
    target_area = math.pi * params["final_growth_radius"] ** 2

    for frame in timeline[max(45, params["yellow_duration"]) :]:
        total = max(frame.pigment_count, 1)
        yellow_frac = frame.yellow_count / total
        red_frac = frame.red_count / total
        black_frac = frame.black_count / total

        black_positions = np.asarray(
            [pigment.pos for pigment in frame.pigments if pigment.stage(params) == BLACK],
            dtype=float,
        )
        yellow_positions = np.asarray(
            [pigment.pos for pigment in frame.pigments if pigment.stage(params) == YELLOW],
            dtype=float,
        )
        if len(black_positions) > 0 and len(yellow_positions) > 0:
            mix_dist = _pairwise_distances(yellow_positions, black_positions).min(axis=1)
            mix_score = float(np.mean((mix_dist > 2.5) & (mix_dist < 7.0)))
        else:
            mix_score = 0.0

        area_score = 1.0 - abs(frame.skin_area - target_area) / target_area
        score = (
            3.0 * black_frac
            + 2.8 * min(yellow_frac, 0.32) / 0.32
            - 3.5 * abs(yellow_frac - 0.22)
            - 8.0 * red_frac
            + 1.8 * mix_score
            + 1.3 * area_score
            + 0.002 * frame.step
        )
        if frame.black_count > frame.yellow_count > frame.red_count and score > best_score:
            best_score = score
            best_step = frame.step
    return best_step


def simulate(params: dict | None = None) -> tuple[list[Frame], int]:
    model = CuttlefishCA(params=params, mode="self", seed=None, birth_rate_scale=1.0)
    timeline = model.simulate()
    final_step = choose_final_step(timeline, model.params)
    return timeline, final_step


class CuttlefishCA:
    def __init__(self, params: dict | None = None, mode: str = "self", seed: int | None = None, birth_rate_scale: float = 1.0):
        self.params = dict(DEFAULT_PARAMS)
        if params:
            self.params.update(params)
        if seed is not None:
            self.params["seed"] = seed
        if mode not in MODE_SWITCHES:
            raise ValueError(f"Unsupported mode: {mode}")

        self.mode = mode
        self.switches = dict(MODE_SWITCHES[mode])
        self.birth_rate_scale = float(birth_rate_scale)
        self.center = np.array(
            [(self.params["grid_size"] - 1) / 2.0, (self.params["grid_size"] - 1) / 2.0],
            dtype=float,
        )

        self.skin: np.ndarray | None = None
        self.skin_age: np.ndarray | None = None
        self.pigments: list[Pigment] = []
        self.timeline: list[Frame] = []
        self.current_step = 0
        self.previous_radius = 0.5 * (self.params["initial_radius_x"] + self.params["initial_radius_y"])

        self.rng_init = np.random.default_rng(self.params["seed"] + 1)
        self.rng_growth = np.random.default_rng(self.params["seed"] + 2)
        self.rng_birth = np.random.default_rng(self.params["seed"] + 3)

    def initialize(self) -> None:
        n = self.params["grid_size"]
        yy, xx = np.indices((n, n))
        ellipse = (
            ((xx - self.center[1]) / self.params["initial_radius_x"]) ** 2
            + ((yy - self.center[0]) / self.params["initial_radius_y"]) ** 2
            <= 1.0
        )
        self.skin = np.zeros((n, n), dtype=np.int8)
        self.skin_age = -np.ones((n, n), dtype=np.int16)
        self.skin[ellipse] = SKIN
        self.skin_age[ellipse] = 0
        self.pigments = self._initialize_pigments()
        self.timeline = []
        self.current_step = 0
        self.record_frame()

    def step(self) -> None:
        self._advance(record=True)

    def simulate(self, n_steps: int | None = None) -> list[Frame]:
        if n_steps is None:
            n_steps = self.params["n_steps"]
        self.initialize()
        while len(self.timeline) < n_steps:
            self._advance(record=True)
        return self.timeline

    def snapshot(self) -> Frame:
        return self.record_frame()

    def _advance(self, record: bool) -> None:
        if self.skin is None or self.skin_age is None:
            self.initialize()

        self.current_step += 1
        occupied = self.skin == SKIN
        self.skin_age[occupied] += 1

        current_radius = self.grow_skin()
        self.move_pigments_with_growth(current_radius)
        self.birth_new_yellow_pigments()
        self.relax_or_repel_pigments()
        self.mature_pigments()
        self.previous_radius = current_radius
        if record:
            self.record_frame()

    def grow_skin(self) -> float:
        occupied_mask = self.skin == SKIN
        empty_mask = self.skin == EMPTY
        padded = np.pad(occupied_mask.astype(int), 1, mode="constant")
        neighbors = np.zeros_like(occupied_mask, dtype=int)
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                neighbors += padded[1 + di : 1 + di + self.skin.shape[0], 1 + dj : 1 + dj + self.skin.shape[1]]
        frontier = empty_mask & (neighbors > 0)

        target_radius, growth_velocity = self._growth_schedule(self.current_step)
        yy, xx = np.indices(self.skin.shape)
        radial_distance = np.sqrt((xx - self.center[1]) ** 2 + (yy - self.center[0]) ** 2)
        radial_gate = 1.0 / (
            1.0 + np.exp((radial_distance - target_radius) / self.params["radial_growth_softness"])
        )
        radial_gate[radial_distance > target_radius + self.params["radial_growth_margin"]] = 0.0
        speed_scale = self.params["late_growth_floor"] + (1.0 - self.params["late_growth_floor"]) * growth_velocity

        prob = (
            self.params["growth_rate"]
            * speed_scale
            * (neighbors / 8.0) ** self.params["growth_power"]
            * radial_gate
        )
        prob = np.clip(prob, 0.0, 0.90)
        new_skin = frontier & (self.rng_growth.random(self.skin.shape) < prob)
        self.skin[new_skin] = SKIN
        self.skin_age[new_skin] = 0
        return target_radius

    def move_pigments_with_growth(self, current_radius: float) -> None:
        if not self.switches["use_growth_displacement"]:
            self._project_inside_skin()
            return

        if self.previous_radius > 1e-6:
            growth_ratio = 1.0 + self.params["growth_motion_strength"] * (current_radius / self.previous_radius - 1.0)
            if abs(growth_ratio - 1.0) >= 1e-7:
                for pigment in self.pigments:
                    pigment.pos = self.center + (pigment.pos - self.center) * growth_ratio
        self._project_inside_skin()

    def birth_new_yellow_pigments(self) -> None:
        eligible = (self.skin == SKIN) & (self.skin_age >= self.params["min_skin_age_for_diff"])
        candidate_positions = np.argwhere(eligible)
        if len(candidate_positions) == 0:
            return

        sample_size = min(self.params["candidate_sample_size"], len(candidate_positions))
        sampled_idx = self.rng_birth.choice(len(candidate_positions), size=sample_size, replace=False)
        sampled = candidate_positions[sampled_idx].astype(float)

        inhibition_allow = np.ones(len(sampled), dtype=float)
        d_all = self._distance_to_nearest(sampled, self.pigments)
        if self.switches["use_repulsion"]:
            inhibition_field, d_all = self._age_inhibition_field(sampled)
            inhibition_allow = 1.0 / (
                1.0 + np.exp((inhibition_field - self.params["field_threshold"]) / self.params["field_softness"])
            )

        black_pigments = [pigment for pigment in self.pigments if pigment.stage(self.params) == BLACK]
        d_black = self._distance_to_nearest(sampled, black_pigments)

        boundary_gate = np.ones(len(sampled), dtype=float)
        d_boundary = np.full(len(sampled), np.inf)
        if self.switches["use_boundary_penalty"]:
            boundary_positions = self._boundary_positions()
            d_boundary = self._distance_between_point_sets(sampled, boundary_positions)
            boundary_gate = 1.0 / (
                1.0
                + np.exp(
                    -(d_boundary - self.params["min_boundary_distance"]) / self.params["boundary_softness"]
                )
            )

        center_gate = self._center_birth_gate(sampled)
        gap_pref = np.ones(len(sampled), dtype=float)
        if self.switches["use_gap_birth"]:
            if len(black_pigments) >= self.params["bootstrap_black_count"]:
                gap_pref = np.exp(
                    -((d_black - self.params["target_gap_to_black"]) ** 2)
                    / (2.0 * self.params["target_gap_sigma"] ** 2)
                )
                if self.switches["use_repulsion"]:
                    gap_pref *= 1.0 / (
                        1.0 + np.exp(-(d_black - self.params["absolute_min_distance"]) / 0.7)
                    )
                gap_pref *= self.params["black_gap_weight"]
            else:
                gap_pref = np.exp(
                    -((d_all - self.params["bootstrap_gap_to_all"]) ** 2)
                    / (2.0 * (self.params["target_gap_sigma"] * 1.25) ** 2)
                )
                if self.switches["use_repulsion"]:
                    gap_pref *= 1.0 / (
                        1.0 + np.exp(-(d_all - self.params["absolute_min_distance"]) / 0.8)
                    )
                gap_pref *= self.params["fallback_gap_weight"]

        birth_prob = (
            self.params["base_birth_rate"]
            * self.birth_rate_scale
            * inhibition_allow
            * gap_pref
            * boundary_gate
            * center_gate
        )
        birth_prob = np.clip(birth_prob, 0.0, self.params["birth_prob_cap"])
        if self.switches["use_repulsion"]:
            birth_prob[d_all < self.params["absolute_min_distance"]] = 0.0
        if self.switches["use_boundary_penalty"]:
            birth_prob[d_boundary < self.params["min_boundary_distance"] * 0.55] = 0.0

        dynamic_quota = self._dynamic_birth_quota(eligible)
        order = np.argsort(birth_prob + self.rng_birth.uniform(0.0, 1e-6, size=len(sampled)))[::-1]

        chosen_positions: list[np.ndarray] = []
        existing_positions = self._pigment_points(self.pigments)
        for idx in order:
            if len(chosen_positions) >= dynamic_quota:
                break
            if self.rng_birth.random() >= birth_prob[idx]:
                continue

            pos = sampled[idx]
            if self.switches["use_repulsion"] and len(existing_positions) > 0:
                dist_existing = np.sqrt(np.sum((existing_positions - pos) ** 2, axis=1))
                if np.any(dist_existing < self.params["absolute_min_distance"]):
                    continue
            if self.switches["use_repulsion"] and chosen_positions:
                chosen_array = np.asarray(chosen_positions, dtype=float)
                dist_chosen = np.sqrt(np.sum((chosen_array - pos) ** 2, axis=1))
                if np.any(dist_chosen < self.params["absolute_min_distance"]):
                    continue
            chosen_positions.append(pos.copy())

        for pos in chosen_positions:
            self.pigments.append(self._make_pigment(pos, age=0, rng=self.rng_birth))

    def mature_pigments(self) -> None:
        for pigment in self.pigments:
            pigment.age += 1

    def relax_or_repel_pigments(self) -> None:
        if not self.switches["use_growth_displacement"] or not self.switches["use_repulsion"]:
            self._project_inside_skin()
            return
        if len(self.pigments) < 2:
            self._project_inside_skin()
            return

        for _ in range(self.params["relax_iterations"]):
            for i in range(len(self.pigments)):
                for j in range(i + 1, len(self.pigments)):
                    pi = self.pigments[i]
                    pj = self.pigments[j]
                    delta = pj.pos - pi.pos
                    dist = float(np.linalg.norm(delta))
                    direction = np.array([1.0, 0.0]) if dist < 1e-8 else delta / dist
                    size_i = max(pigment_display_size(pi, self.params))
                    size_j = max(pigment_display_size(pj, self.params))
                    min_sep = 0.48 * (size_i + size_j) + self.params["relax_buffer"]
                    if dist >= min_sep:
                        continue
                    push = 0.5 * (min_sep - dist) * self.params["relax_strength"]
                    pi.pos = pi.pos - push * direction
                    pj.pos = pj.pos + push * direction

            for pigment in self.pigments:
                vec = pigment.pos - self.center
                radius = np.linalg.norm(vec)
                if radius > self.params["final_growth_radius"] + 1.3 and radius > 1e-8:
                    pigment.pos = self.center + vec / radius * (self.params["final_growth_radius"] + 1.3)
        self._project_inside_skin()

    def record_frame(self) -> Frame:
        points = self._pigment_points(self.pigments)
        nnd_mean, nnd_std, nnd_cv = self._summarize_points(points)
        yellow_count, red_count, black_count = self._color_counts()
        frame = Frame(
            step=self.current_step,
            skin=self.skin.copy(),
            pigments=[pigment.copy() for pigment in self.pigments],
            skin_area=int(np.count_nonzero(self.skin == SKIN)),
            pigment_count=len(self.pigments),
            yellow_count=yellow_count,
            red_count=red_count,
            black_count=black_count,
            nnd_mean=nnd_mean,
            nnd_std=nnd_std,
            nnd_cv=nnd_cv,
        )
        self.timeline.append(frame)
        return frame

    def _growth_schedule(self, step: int) -> tuple[float, float]:
        total_steps = max(self.params["n_steps"] - 1, 1)
        progress = np.clip(step / total_steps, 0.0, 1.0)
        exponent = self.params["growth_curve_exponent"]
        eased = 1.0 - (1.0 - progress) ** exponent
        start_radius = 0.5 * (self.params["initial_radius_x"] + self.params["initial_radius_y"])
        final_radius = self.params["final_growth_radius"]
        target_radius = start_radius + (final_radius - start_radius) * eased
        growth_velocity = (1.0 - progress) ** max(exponent - 1.0, 0.0)
        return target_radius, growth_velocity

    def _initialize_pigments(self) -> list[Pigment]:
        skin_sites = np.argwhere(self.skin == SKIN)
        self.rng_init.shuffle(skin_sites)
        pigments: list[Pigment] = []
        for row, col in skin_sites:
            pos = np.array([float(row), float(col)])
            if any(np.linalg.norm(pos - pigment.pos) < 6.5 for pigment in pigments):
                continue
            pigments.append(self._make_pigment(pos, age=int(self.rng_init.integers(0, 8)), rng=self.rng_init))
            if len(pigments) >= self.params["initial_pigments"]:
                break
        return pigments

    def _make_pigment(self, pos: np.ndarray, age: int = 0, rng: np.random.Generator | None = None) -> Pigment:
        if rng is None:
            rng = self.rng_birth
        major = rng.uniform(1.8, 2.9)
        minor = major * rng.uniform(0.58, 0.92)
        return Pigment(
            pos=pos.astype(float),
            age=int(age),
            base_major=float(major),
            base_minor=float(minor),
            angle=float(rng.uniform(0.0, 180.0)),
            tone=float(rng.uniform(-0.08, 0.08)),
        )

    def _pigment_points(self, pigments: list[Pigment]) -> np.ndarray:
        if not pigments:
            return np.empty((0, 2), dtype=float)
        return np.asarray([pigment.pos for pigment in pigments], dtype=float)

    def _color_counts(self) -> tuple[int, int, int]:
        yellow = 0
        red = 0
        black = 0
        for pigment in self.pigments:
            stage = pigment.stage(self.params)
            if stage == YELLOW:
                yellow += 1
            elif stage == RED:
                red += 1
            else:
                black += 1
        return yellow, red, black

    def _dynamic_birth_quota(self, eligible_mask: np.ndarray) -> int:
        eligible_area = np.count_nonzero(eligible_mask)
        target_pigment_count = eligible_area / self.params["target_area_per_pigment"]
        pigment_deficit = max(0.0, target_pigment_count - len(self.pigments))
        quota = np.ceil(
            self.params["birth_quota_base"]
            + self.params["birth_quota_skin_scale"] * eligible_area
            + self.params["birth_quota_deficit_scale"] * pigment_deficit
        )
        return int(np.clip(quota, 1, self.params["max_new_pigments_per_step"]))

    def _project_inside_skin(self) -> None:
        n = self.skin.shape[0]
        for pigment in self.pigments:
            pos = pigment.pos
            for _ in range(12):
                rr = int(np.clip(round(pos[0]), 0, n - 1))
                cc = int(np.clip(round(pos[1]), 0, n - 1))
                if self.skin[rr, cc] == SKIN:
                    break
                direction = self.center - pos
                norm = np.linalg.norm(direction)
                if norm < 1e-8:
                    break
                pos = pos + 0.55 * direction / norm
            pigment.pos = np.clip(pos, 0.0, n - 1.0)

    def _boundary_positions(self) -> np.ndarray:
        occupied = self.skin == SKIN
        padded = np.pad(occupied.astype(int), 1, mode="constant")
        neighbors = np.zeros_like(occupied, dtype=int)
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                neighbors += padded[1 + di : 1 + di + occupied.shape[0], 1 + dj : 1 + dj + occupied.shape[1]]
        return np.argwhere(occupied & (neighbors < 8)).astype(float)

    def _age_inhibition_field(self, sampled: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not self.pigments:
            return np.zeros(len(sampled)), np.full(len(sampled), np.inf)
        positions = self._pigment_points(self.pigments)
        dist = _pairwise_distances(sampled, positions)
        radii = np.asarray([inhibition_radius(pigment.age, self.params) for pigment in self.pigments], dtype=float)
        field = 1.0 / (1.0 + np.exp((dist - radii[None, :]) / self.params["all_spacing_softness"]))
        return field.sum(axis=1), dist.min(axis=1)

    def _center_birth_gate(self, sampled: np.ndarray) -> np.ndarray:
        d_center = np.sqrt(np.sum((sampled - self.center) ** 2, axis=1))
        effective_radius = max(np.sqrt(np.count_nonzero(self.skin == SKIN) / np.pi), 1.0)
        sigma = max(8.0, effective_radius * self.params["center_birth_sigma_fraction"])
        core = np.exp(-(d_center**2) / (2.0 * sigma**2))
        return self.params["center_birth_floor"] + (1.0 - self.params["center_birth_floor"]) * core

    def _distance_to_nearest(self, sampled: np.ndarray, pigments: list[Pigment]) -> np.ndarray:
        if len(sampled) == 0:
            return np.array([], dtype=float)
        positions = self._pigment_points(pigments)
        return self._distance_between_point_sets(sampled, positions)

    def _distance_between_point_sets(self, points_a: np.ndarray, points_b: np.ndarray) -> np.ndarray:
        if len(points_a) == 0:
            return np.array([], dtype=float)
        if len(points_b) == 0:
            return np.full(len(points_a), np.inf)
        return _pairwise_distances(points_a, points_b).min(axis=1)

    def _summarize_points(self, points: np.ndarray) -> tuple[float, float, float]:
        if len(points) < 2:
            return math.nan, math.nan, math.nan
        dist = _pairwise_distances(points, points)
        np.fill_diagonal(dist, np.inf)
        nnd = dist.min(axis=1)
        mean_val = float(np.mean(nnd))
        std_val = float(np.std(nnd))
        return mean_val, std_val, float(std_val / (mean_val + 1e-12))


def _pairwise_distances(points_a: np.ndarray, points_b: np.ndarray) -> np.ndarray:
    if len(points_a) == 0 or len(points_b) == 0:
        return np.empty((len(points_a), len(points_b)))
    diff = points_a[:, None, :] - points_b[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))
