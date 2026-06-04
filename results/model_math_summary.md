# 模型数学摘要

## 1. 状态变量

二维格点元胞自动机状态记为 $S_{ij}(t) \in \{0,1,2,3,4\}$：

- 0：空白区域
- 1：普通皮肤细胞
- 2：黄色新生色素细胞
- 3：红色过渡态色素细胞
- 4：黑色成熟色素细胞

## 2. 皮肤生长与出生逻辑

皮肤从中心小区域向外扩张，出生概率可概括为：

$$
P_{birth} = \eta \cdot I(d_{all}) \cdot G(d_{black}) \cdot B(d_{boundary})
$$

其中：

- $I(d_{all})$：短程抑制与最小间距
- $G(d_{black})$：对成熟黑色阵列空隙的偏好
- $B(d_{boundary})$：边界惩罚

## 3. 颜色成熟

- yellow_duration = 28
- red_duration = 3

## 4. 匹配随机对照

- self: N = 214, CV_NND = 0.1635
- random-development matched: N = 210, CV_NND = 0.4823
- random-mask matched: N = 214, CV_NND = 0.4895

## 5. Ablation

- no_repulsion: N = 216, CV_NND = 0.4707
- no_gap_birth: N = 196, CV_NND = 0.1989
- no_growth_displacement: N = 195, CV_NND = 0.2139

## 6. 图解释

- Fig1：self 与 matched random-development 的发育切片比较
- Fig2：NND 分布与 CV_NND 对比
- Fig3：yellow / red / black 随时间变化
- Fig4：局部规则参数景观
- Fig5：短距离 pair-density dip
- Fig6/Fig7：ablation 对空间秩序的影响