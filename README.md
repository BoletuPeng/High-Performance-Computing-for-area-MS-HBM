# High-Performance Computing for Areal MS-HBM

> [中文](#中文) | [English](#english)

---

# 中文

## 愿景

本项目致力于构建一个**高性能**、**完全独立**的 Areal MS-HBM (Areal-level Multi-Session Hierarchical Bayesian Model) 计算框架。

我们的目标是：

- **为下一代个体化脑分割**提供准确、可信、可靠的计算基础
- **完全摆脱对付费编程语言的依赖**——我们坚信科学计算不应被商业软件所绑架
- **实现真正的开源与可复现性**——任何研究者都能自由地使用、验证和扩展我们的工作

## 特性

- **纯 Python 实现** — 告别 MATLAB，拥抱开源
- **GPU 加速支持** — 基于 CuPy 的大规模并行计算
- **高性能 CPU 优化** — 使用 Numba JIT 编译加速
- **自包含** — 无需依赖原始 CBIG 工具箱即可独立运行

## 技术栈

| 组件 | 用途 |
|------|------|
| Python 3 | 核心语言 |
| NumPy | 数值计算基础 |
| Numba | CPU 端 JIT 编译加速 |
| CuPy | GPU 并行计算 |

## 项目结构

```
lib/
├── Cdln.py                     # 基于DLMF的高精度高性能贝塞尔函数实现
├── Cdln_Par.py                 # 并行版贝塞尔函数
├── vmf_probability.py          # vMF 概率计算 (CPU)
├── vmf_probability_gpu.py      # vMF 概率计算 (GPU)
├── initialize_concentration.py # 浓度参数初始化
└── initialize_child_params.py  # 子参数初始化 (GPU)
```

## 许可证

本项目采用 [GNU General Public License v3.0](LICENSE) 开源协议。

这意味着你可以自由地使用、修改和分发本项目，但任何衍生作品也必须以相同的协议开源。我们希望通过这种方式确保所有改进都能回馈社区。

*我们相信，高质量的科学计算工具应当对所有人免费开放。*

---

# English

## Vision

This project is dedicated to building a **high-performance**, **fully standalone** computational framework for Areal MS-HBM (Areal-level Multi-Session Hierarchical Bayesian Model).

Our goals are:

- **Provide an accurate, trustworthy, and reliable computational foundation** for next-generation individualized brain parcellation
- **Completely eliminate dependence on proprietary programming languages** — we firmly believe that scientific computing should not be held hostage by commercial software
- **Achieve true open-source reproducibility** — enabling any researcher to freely use, verify, and extend our work

## Features

- **Pure Python Implementation** — Say goodbye to MATLAB, embrace open source
- **GPU Acceleration** — Large-scale parallel computing powered by CuPy
- **High-Performance CPU Optimization** — Accelerated with Numba JIT compilation
- **Standalone** — Runs independently without relying on the original CBIG toolbox

## Tech Stack

| Component | Purpose |
|-----------|---------|
| Python 3 | Core language |
| NumPy | Numerical computing foundation |
| Numba | CPU-side JIT compilation acceleration |
| CuPy | GPU parallel computing |

## Project Structure

```
lib/
├── Cdln.py                     # High-precision, high-performance Bessel function based on DLMF
├── Cdln_Par.py                 # Parallel Bessel function
├── vmf_probability.py          # vMF probability (CPU)
├── vmf_probability_gpu.py      # vMF probability (GPU)
├── initialize_concentration.py # Concentration parameter initialization
└── initialize_child_params.py  # Child parameter initialization (GPU)
```

## License

This project is licensed under the [GNU General Public License v3.0](LICENSE).

This means you are free to use, modify, and distribute this project, but any derivative works must also be open-sourced under the same license. We hope this ensures that all improvements are contributed back to the community.

*We believe that high-quality scientific computing tools should be freely available to everyone.*
