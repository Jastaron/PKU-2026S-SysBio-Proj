# 模型数学摘要

## 1. 状态变量

记二维格点元胞自动机状态为 $S_{ij}(t) \in \{0,1,2,3,4\}$：

- 0：空白区域，尚未长出的皮肤
- 1：普通皮肤细胞
- 2：黄色新生色素细胞
- 3：红色/橙红色过渡态色素细胞
- 4：黑色成熟色素细胞

## 2. 皮肤生长概率

皮肤只在已有皮肤边界附近扩张，概率可概括为：

$$
P_{grow}(i,j,t)=g\left(\frac{n_{skin}}{8}\right)^\alpha R_{radial}(i,j,t)
$$

其中 $g$ 对应 `growth_rate`，$\alpha$ 对应 `growth_power`，$R_{radial}$ 为径向门控函数，因此皮肤从中心近圆形扩张并在后期减速。

## 3. 色素细胞出生概率

full self-organized 模型中，普通皮肤细胞分化为新生黄色色素细胞的概率写成：

$$
P_{birth} = \beta \cdot I(d_{all}) \cdot G(d_{black}) \cdot B(d_{boundary})
$$

- $I(d_{all})$：局部抑制/短程排斥项，避免新生细胞离已有色素细胞过近
- $G(d_{black})$：成熟黑色阵列空隙偏好项，鼓励黄色细胞插入黑色网络 gap
- $B(d_{boundary})$：边界惩罚项，避免新生色素细胞总贴边出现

## 4. 颜色成熟

颜色严格由年龄决定：

$$
Y \rightarrow R \rightarrow B
$$

当前参数为：

- `yellow_duration = 28`
- `red_duration = 3`

因此黄色持续较久，红色只是短暂过渡态，黑色在发育过程中不断累积。

## 5. 为什么要做 matched random-development

原始 random-development 会产生远多于 self-organized 的 pigment，因此最近邻距离和 CV_NND 会受到密度差异污染，无法构成公平对照。

因此本次对 random-development 使用了 birth-rate calibration，使其最终 pigment 数量接近 self-organized。当前：

- full self-organized：N = 191，CV_NND = 0.1731
- random-development matched：N = 187，CV_NND = 0.5478
- random-mask matched：N = 191，CV_NND = 0.5352

random-development matched 使用的校准 birth rate scale 为 0.0016。

## 6. 三项机制与 ablation

full self-organized 同时包含三项局部机制：

1. pigment-pigment repulsion  
   控制短距离排斥和最小间距约束。

2. gap-biased birth / intercalation  
   控制新生黄色细胞优先插入成熟黑色阵列空隙。

3. growth-associated displacement  
   控制皮肤生长与内部插入导致的新旧 pigment 局部位移与重排。

对应 ablation：

- no_repulsion：去掉短程排斥，只保留空隙偏好与组织位移。
- no_gap_birth：保留排斥，但去掉“向黑色阵列 gap 插空”的出生偏好。
- no_growth_displacement：保留排斥与 gap birth，但 pigment 出生后位置不再随组织生长重排。

本次结果：

- no_repulsion：N = 206，CV_NND = 0.4543
- no_gap_birth：N = 191，CV_NND = 0.1786
- no_growth_displacement：N = 205，CV_NND = 0.2117

如果去掉某机制后 CV_NND 上升，或者图像变得更随机、更拥挤或更空洞，就说明该局部规则对全局均匀网络形成有贡献。

## 7. 自组织与涌现

### 自组织判据

模型没有全局坐标蓝图；每个格点只依据局部皮肤环境、边界距离、局部抑制场和成熟黑色阵列 gap 来决定是否分化，但整体仍能形成较均匀的色素细胞间距。

### 涌现性判据

单格点规则很简单，但群体层面会形成可作为成年神经快速变色基础的“皮肤像素阵列”。这说明全局空间秩序可以由多个局部规则叠加涌现。

## 8. 主要图的解释

- `Fig1_development_slices_self_vs_random.png`：比较 full self-organized 与 matched random-development 的发育切片。
- `Fig2_nnd_distribution_and_cv.png`：比较 self、matched random-development、matched random-mask 的 NND 分布与 CV_NND。
- `Fig3_color_composition_over_time.png`：展示 black 累积、yellow 持续补充、red 始终较少。
- `Fig4_parameter_phase_heatmap.png`：展示 full self-organized 在局部规则参数上的 spacing order landscape。
- `Fig5_pair_correlation.png`：显示 self-organized 在短距离处的 pair-density dip。
- `Fig6_ablation_summary.png` 与 `Fig7_ablation_final_patterns.png`：总结去掉不同机制后，空间均匀性和最终图景如何变化。
