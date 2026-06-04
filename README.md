# PKU-2026S-SysBio-Proj

这是一个基于二维元胞自动机的乌贼皮肤色素细胞发育 toy model 项目。

## 使用指南

请直接打开并阅读：

- `CuttleFish_run.ipynb`

这个 notebook 是当前项目的主工作流，已经按分析步骤拆成多个单元格，适合：

- 单独运行 self-organized 模型
- 单独运行 matched random / ablation 对照
- 单独生成每一张报告图
- 直接在对应单元格里修改 figsize、dpi、颜色、bins、legend、标题和保存路径

如果你只是想理解项目、复现实验或微调图，优先看 notebook。

## 项目结构

- `CuttleFish_run.ipynb`：主入口，交互式分析与出图工作流
- `CuttleFishModel/core.py`：核心 CA 模型与数据结构
- `CuttleFishModel/metrics.py`：NND、pair-correlation、timeline 指标整理
- `CuttleFishModel/controls.py`：matched random 对照与 ablation
- `CuttleFishModel/render.py`：底层绘图辅助函数
- `results/`：模型 GIF、报告图和导出的 CSV / Markdown

## 推荐流程

1. 在 Jupyter Notebook 或 VS Code Notebook 模式下打开 `CuttleFish_run.ipynb`
2. 先运行 imports / parameters / self-organized model 相关单元格
3. 再按需运行具体图的单元格，而不是一次性全部跑完
4. 修改图形细节时，直接在对应 cell 内调整参数

## 环境

本项目默认你已经安装常见基础科学计算包，例如：

- `numpy`
- `pandas`
- `matplotlib`
