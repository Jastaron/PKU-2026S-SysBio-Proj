# 模型数学摘要

## 1. 状态变量

记二维格点元胞自动机状态为 $S_{ij}(t) \in \{0,1,2,3,4\}$：

- 0：空白区域，尚未长出的皮肤
- 1：普通皮肤细胞
- 2：黄色新生色素细胞
- 3：红色/橙红色过渡态色素细胞
- 4：黑色成熟色素细胞

## 2. 皮肤生长

皮肤只在已有皮肤边界附近向外扩张。对空白格点，若其邻域中已有皮肤，则其转化概率可写为：

$$
P_{grow}(i,j,t) = g \cdot \left(\frac{n_{skin}}{8}\right)^\alpha \cdot R_{radial}(i,j,t)
$$

其中 $g$ 对应 `growth_rate`，$\alpha$ 对应 `growth_power`，$R_{radial}$ 是随目标半径减速推进的径向门控函数，因此皮肤从中央近圆形扩张并在后期放缓。

## 3. 色素细胞出生

自组织模型中，普通皮肤细胞分化为新生黄色色素细胞的概率可概括为：

$$
P_{birth} = \beta \cdot I(d_{all}) \cdot G(d_{black}) \cdot B(d_{boundary})
$$

- $\beta$：基础出生率，对应 `base_birth_rate`
- $I(d_{all})$：局部抑制项，反映与所有已有色素细胞的抑制场关系
- $G(d_{black})$：黑色成熟阵列的间隙偏好项，鼓励新生黄色细胞插入成熟黑色网络空隙
- $B(d_{boundary})$：边界惩罚项，避免色素细胞贴皮肤边界生成

random-development 消融则保留皮肤生长、年龄依赖成熟和边界惩罚，但去掉局部抑制与黑色间隙偏好，因此普通皮肤细胞更接近随机分化。

## 4. 颜色成熟

颜色严格由年龄决定，遵循：

$$
Y \rightarrow R \rightarrow B
$$

本版默认参数为：

- `yellow_duration = 28`
- `red_duration = 3`

因此黄色阶段较长，红色仅为短暂过渡态，黑色不断累积为成熟主色。

## 5. 自组织与涌现

### 自组织判据

模型中没有全局坐标蓝图，也没有预设晶格模板。每个格点只依据局部皮肤环境、边界距离、局部抑制场和成熟黑色阵列间隙来决定是否分化，但整体上会产生较均匀的色素细胞间距。

### 涌现性判据

单个格点的状态转移规则很简单，但群体层面会形成可作为神经快速变色基础的“皮肤像素阵列”。这说明复杂空间网络可以由局部规则在发育过程中涌现出来。

## 6. 主要图的解释

- `Fig1_development_slices_self_vs_random.png`：显示自组织模型在成熟黑色阵列中持续插入新生黄色细胞，而随机对照更容易出现无结构填充。
- `Fig2_nnd_distribution_and_cv.png`：展示 self-organized、random-development、random-mask 的最近邻距离分布与 CV_NND，对比空间均匀性。
- `Fig3_color_composition_over_time.png`：显示 black 持续累积、yellow 持续补充、red 始终较少，符合短过渡态设想。
- `Fig4_parameter_phase_heatmap.png`：展示从较随机/较拥挤到较均匀插空网络之间的参数转变边界。
- `Fig5_pair_correlation.png`：显示 self-organized 在极短距离处的明显 dip，说明存在短距离排斥。

## 7. 本次代表性结果

- final step = 147
- Y/R/B = 47/1/143
- self-organized CV_NND = 0.1731
- random-development CV_NND = 0.2136
- random-mask CV_NND = 0.5352
