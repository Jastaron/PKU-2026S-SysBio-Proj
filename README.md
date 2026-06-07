# WeiJC-PKU-2026S-SysBio-Proj

这是一个基于二维元胞自动机的乌贼皮肤色素细胞发育 toy model 项目。

## 使用指南

请直接打开：

- `CuttleFish_run.ipynb`

这个 notebook 是本项目主要流程，完成了跑模型，数据统计和可视化等一系列操作，如果想理解项目、复现实验或微调图，请优先看这个 notebook。

## 项目结构

- `CuttleFish_run.ipynb`：交互式脚本
- `CuttleFishModel/core.py`：核心 CA 模型与数据结构
- `CuttleFishModel/metrics.py`：NND、pair-correlation、timeline 指标整理
- `CuttleFishModel/controls.py`：matched random 对照与 ablation
- `results/`：模型 GIF、报告图和导出的 CSV / Markdown

## 推荐流程

1. 在 Jupyter Notebook 或 VS Code Notebook 模式下打开 `CuttleFish_run.ipynb`
2. 先运行 imports / parameters / self-organized model 相关单元格
3. 再按需运行具体图的单元格
4. 修改图形细节时，直接在对应单元格内调整参数

## 环境

本项目默认你已经安装常见基础科学计算包，例如：

- `numpy`
- `pandas`
- `matplotlib`