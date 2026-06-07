from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


# 元胞自动机中的五种格点状态：空白、普通皮肤、黄色新生、红色过渡、黑色成熟。
EMPTY = 0
SKIN = 1
YELLOW = 2
RED = 3
BLACK = 4


DEFAULT_PARAMS = {
    # 基本设置
    "seed": 7,  # 随机种子，保证整套发育过程可复现。
    "grid_size": 100,  # 元胞自动机棋盘边长，即皮肤区域所在的二维网格大小。
    "n_steps": 170,  # 总模拟步数，对应一个完整的发育时间窗口。

    # 初始皮肤与初始色素细胞
    "initial_radius_x": 8.0,  # 初始椭圆皮肤区域在 x 方向的半轴长度。
    "initial_radius_y": 6.5,  # 初始椭圆皮肤区域在 y 方向的半轴长度。
    "initial_pigments": 4,  # 初始皮肤中预放置的少量色素细胞数量，用于启动阵列形成。

    # 皮肤生长
    "growth_rate": 0.94,  # 基础皮肤生长率，控制边界空白格转为皮肤的总体概率尺度。
    "growth_power": 0.95,  # 邻域皮肤依赖指数，邻居越多则生长概率提升越明显。
    "growth_curve_exponent": 1.35,  # 生长调度的缓动指数，决定整体发育由快到慢的时间曲线。
    "radial_growth_softness": 1.2,  # 径向生长门控的平滑尺度，控制目标半径附近的过渡宽度。
    "radial_growth_margin": 0.9,  # 目标半径之外允许的额外生长边缘宽度。
    "late_growth_floor": 0.24,  # 发育后期仍保留的最低生长速度比例，避免完全停止生长。
    "final_growth_radius": 49.0,  # 最终皮肤目标半径，决定成熟时皮肤区域的大致尺度。

    # 色素细胞出生候选与出生配额
    "min_skin_age_for_diff": 2,  # 普通皮肤细胞至少经历多少步后才允许分化为色素细胞。
    "candidate_sample_size": 2400,  # 每一步最多抽样多少个候选皮肤位置来评估出生概率。
    "base_birth_rate": 0.82,  # 新生黄色色素细胞的基础出生率系数。
    "birth_prob_cap": 0.78,  # 单个候选位置出生概率的上限，避免某一步出现过强爆发。
    "max_new_pigments_per_step": 10,  # 每一步最多允许新增的色素细胞数。
    "birth_quota_base": 1.0,  # 动态出生配额的基础项，保证早期至少有少量新生细胞。
    "birth_quota_skin_scale": 0.0011,  # 出生配额随可分化皮肤面积增加的比例系数。
    "birth_quota_deficit_scale": 0.08,  # 出生配额随目标细胞数缺口增加的补偿强度。
    "target_area_per_pigment": 28.0,  # 目标上每个色素细胞平均对应的皮肤面积，用于估计目标密度。

    # 短程抑制与最小间距
    "absolute_min_distance": 2.6,  # 新生细胞与已有细胞之间的绝对最小允许距离。
    "all_spacing_softness": 0.9,  # 抑制场随距离衰减的平滑尺度。
    "field_threshold": 0.92,  # 总抑制场阈值，超过后出生允许度迅速下降。
    "field_softness": 0.18,  # 抑制场阈值函数的平滑程度。

    # 成熟黑色阵列间隙偏好
    "target_gap_to_black": 5.0,  # 新生黄色细胞最偏好的成熟黑色阵列间隙距离。
    "target_gap_sigma": 1.6,  # 对目标插空距离允许的波动范围。
    "bootstrap_gap_to_all": 5.5,  # 黑色细胞尚少时，改为参考全部色素细胞的启动间隙距离。
    "bootstrap_black_count": 18,  # 黑色细胞达到该数量后，才正式切换到成熟黑阵列插空规则。
    "black_gap_weight": 1.55,  # 成熟黑色阵列插空偏好的权重。
    "fallback_gap_weight": 1.15,  # 启动阶段基于全部细胞插空偏好的权重。

    # 边界惩罚与中心启动
    "min_boundary_distance": 3.2,  # 距离皮肤边界过近时，新生色素细胞会被抑制的阈值距离。
    "boundary_softness": 0.9,  # 边界惩罚从强到弱过渡的平滑尺度。
    "center_birth_floor": 0.28,  # 中心启动门控的最低值，避免外围区域完全失去出生机会。
    "center_birth_sigma_fraction": 0.48,  # 中心门控的空间尺度，相对于当前皮肤有效半径设定。

    # 年龄依赖颜色成熟
    "yellow_duration": 28,  # 黄色阶段持续步数，表示新生色素细胞的较长早期阶段。
    "red_duration": 3,  # 红色阶段持续步数，表示短暂过渡态。
    "black_growth_duration": 26,  # 进入黑色后继续变大的时间尺度，只影响显示尺寸不改颜色。

    # 年龄依赖抑制半径
    "inhibition_birth_radius": 8.6,  # 新生细胞刚出生时的抑制半径。
    "inhibition_mature_radius": 4.2,  # 成熟后抑制半径衰减到的下限。
    "inhibition_radius_decay": 0.18,  # 抑制半径随年龄减小的速度。

    # 随皮肤生长的被动位移
    "growth_motion_strength": 1.0,  # 皮肤扩张时色素细胞被组织拉开的径向位移强度。

    # 局部斥力与重排
    "relax_iterations": 2,  # 每一步局部重排迭代次数。
    "relax_strength": 0.38,  # 过近细胞被推开的强度。
    "relax_buffer": 0.35,  # 在显示尺寸之外额外保留的最小缓冲距离。

    # 元胞格点显示尺寸
    "size_start_scale": 0.52,  # 色素细胞刚出生时，相对其基准尺寸的显示比例。
    "size_yellow_end_scale": 0.90,  # 黄色阶段结束时的显示比例。
    "size_red_end_scale": 1.05,  # 红色阶段结束时的显示比例。
    "size_black_end_scale": 1.42,  # 黑色成熟后最终显示比例。
    "size_softness_power": 0.78,  # 尺寸随年龄增长时的缓动指数。

    # 输出帧采样与 GIF 渲染
    "frame_stride": 2,  # 导出动画时每隔多少步取一帧。
    "gif_duration_ms": 150,  # GIF 中每帧停留时间，单位毫秒。
    "render_scale": 8,  # 可视化时将 CA 格点放大的倍率。
}


# 完整模型与各类消融/随机对照使用的机制开关组合。
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
    """连续空间中的单个色素细胞对象，与离散 CA 皮肤格点是两层表示。"""

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
    """某一步发育状态的快照，供统计分析与可视化直接使用。"""

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
    """色素细胞颜色完全由年龄决定：黄色 → 红色 → 黑色。"""
    if age < params["yellow_duration"]:
        return YELLOW
    if age < params["yellow_duration"] + params["red_duration"]:
        return RED
    return BLACK


def pigment_display_size(pigment: Pigment, params: dict) -> tuple[float, float]:
    """根据年龄计算显示层尺寸；这只影响可视化外观，不改变出生或运动规则。"""
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
    """新生细胞抑制半径较大，随年龄增长逐步减小，成熟后不再继续减小。"""
    radius = params["inhibition_birth_radius"] - params["inhibition_radius_decay"] * age
    return max(params["inhibition_mature_radius"], radius)


def choose_final_step(timeline: list[Frame], params: dict) -> int:
    """从整段时间序列中挑选一个最能代表“持续发育中成熟阵列”的时刻。"""
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
    """运行完整 self-organized 模型，并返回时间序列与代表性 final_step。"""
    model = CuttlefishCA(params=params, mode="self", seed=None, birth_rate_scale=1.0)
    timeline = model.simulate()
    final_step = choose_final_step(timeline, model.params)
    return timeline, final_step


class CuttlefishCA:
    """乌贼皮肤色素细胞自组织的主模型：离散皮肤 CA + 连续色素细胞位置。"""

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
        """生成初始椭圆形皮肤区域，并在其中放入少量初始色素细胞。"""
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

        # 1. 皮肤年龄推进：已长出的皮肤区域继续变老。
        self.skin_age[occupied] += 1

        # 2. 皮肤向外生长：边界附近空白格有概率转化为普通皮肤细胞。
        current_radius = self.grow_skin()

        # 3. 色素细胞随组织生长发生被动位移，而不是主动迁移。
        self.move_pigments_with_growth(current_radius)

        # 4. 普通皮肤细胞按局部规则分化出新生黄色色素细胞。
        self.birth_new_yellow_pigments()

        # 5. 若局部过密，则通过短程斥力进行重排。
        self.relax_or_repel_pigments()

        # 6. 颜色成熟：年龄增长后由黄转红，再转黑。
        self.mature_pigments()

        self.previous_radius = current_radius
        if record:
            # 7. 记录当前帧，用于统计、挑选 final_step 和作图。
            self.record_frame()

    def grow_skin(self) -> float:
        """执行皮肤生长，对应 Summary 中依赖邻域与半径门控的生长概率项。"""
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
        """把皮肤整体扩张转化为色素细胞的被动位移，模拟组织生长带来的拉开效应。"""
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
        """按局部抑制、插空偏好与边界惩罚规则生成新生黄色色素细胞。"""
        eligible = (self.skin == SKIN) & (self.skin_age >= self.params["min_skin_age_for_diff"])
        candidate_positions = np.argwhere(eligible)
        if len(candidate_positions) == 0:
            return

        # 候选位置筛选：只在可分化皮肤区域中抽样一批候选点。
        sample_size = min(self.params["candidate_sample_size"], len(candidate_positions))
        sampled_idx = self.rng_birth.choice(len(candidate_positions), size=sample_size, replace=False)
        sampled = candidate_positions[sampled_idx].astype(float)

        # 抑制场允许项：已有色素细胞过密时，候选位置的出生允许度降低。
        inhibition_allow = np.ones(len(sampled), dtype=float)
        d_all = self._distance_to_nearest(sampled, self.pigments)
        if self.switches["use_repulsion"]:
            inhibition_field, d_all = self._age_inhibition_field(sampled)
            inhibition_allow = 1.0 / (
                1.0 + np.exp((inhibition_field - self.params["field_threshold"]) / self.params["field_softness"])
            )

        black_pigments = [pigment for pigment in self.pigments if pigment.stage(self.params) == BLACK]
        d_black = self._distance_to_nearest(sampled, black_pigments)

        # 边界惩罚：尽量避免新生细胞直接贴着皮肤边缘出现。
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

        # 中心启动门控：发育早期略微偏向中部，但不是强制只在中心出生。
        center_gate = self._center_birth_gate(sampled)

        # 黑色阵列间隙偏好：优先插入成熟黑色阵列的空隙；早期黑色不足时退化为全体细胞间隙。
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

        # 动态出生配额：可分化面积越大、目标密度缺口越大，本步允许出生的细胞越多。
        dynamic_quota = self._dynamic_birth_quota(eligible)
        order = np.argsort(birth_prob + self.rng_birth.uniform(0.0, 1e-6, size=len(sampled)))[::-1]

        # 最终逐个接受新生细胞，并再次检查与已有/同批新生细胞的最小距离约束。
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
        """只推进年龄；颜色状态由年龄映射自动决定。"""
        for pigment in self.pigments:
            pigment.age += 1

    def relax_or_repel_pigments(self) -> None:
        """若色素细胞过近，则通过短程斥力进行局部重排。"""
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
        """保存当前一步的完整快照，用于后续统计分析和图像输出。"""
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
        """发育时间调度函数：皮肤半径先增长较快，后期逐渐放缓。"""
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
        """根据可分化面积和目标密度缺口，动态决定本步最多出生多少新细胞。"""
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
        """把越界到空白区的色素细胞重新拉回当前皮肤区域内。"""
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
        """提取当前皮肤边缘格点，用于计算边界惩罚项。"""
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
        """把所有已有色素细胞的年龄依赖抑制场叠加到候选位置上。"""
        if not self.pigments:
            return np.zeros(len(sampled)), np.full(len(sampled), np.inf)
        positions = self._pigment_points(self.pigments)
        dist = _pairwise_distances(sampled, positions)
        radii = np.asarray([inhibition_radius(pigment.age, self.params) for pigment in self.pigments], dtype=float)
        field = 1.0 / (1.0 + np.exp((dist - radii[None, :]) / self.params["all_spacing_softness"]))
        return field.sum(axis=1), dist.min(axis=1)

    def _center_birth_gate(self, sampled: np.ndarray) -> np.ndarray:
        """早期略微帮助中心区域启动出生，但不等于强制在中心生成。"""
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
        """计算最近邻距离的均值、标准差和 CV，用于量化空间均匀性。"""
        if len(points) < 2:
            return math.nan, math.nan, math.nan
        dist = _pairwise_distances(points, points)
        np.fill_diagonal(dist, np.inf)
        nnd = dist.min(axis=1)
        mean_val = float(np.mean(nnd))
        std_val = float(np.std(nnd))
        return mean_val, std_val, float(std_val / (mean_val + 1e-12))


def _pairwise_distances(points_a: np.ndarray, points_b: np.ndarray) -> np.ndarray:
    """底层距离矩阵工具：返回两组点之间的两两欧氏距离。"""
    if len(points_a) == 0 or len(points_b) == 0:
        return np.empty((len(points_a), len(points_b)))
    diff = points_a[:, None, :] - points_b[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))
