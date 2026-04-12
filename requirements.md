# CMDL 依赖清单

> Phase 1（Core）所需的全部依赖库与开源项目引用。
> 开发步骤与文献阅读节点见 [plan.md](plan.md)。

---

## Python 环境

```
Python 3.10+
CUDA 11.8+ (GPU 训练)
```

## Core 依赖库（Phase 1 必装）

| 库 | 版本建议 | 用途 | 安装 |
|---|---|---|---|
| `torch` | ≥2.1 | 模型训练核心 | `pip install torch` |
| `pandas` | ≥2.0 | 数据处理 | `pip install pandas` |
| `numpy` | ≥1.24 | 数值计算 | `pip install numpy` |
| `scikit-learn` | ≥1.3 | 标准化/指标/分割 | `pip install scikit-learn` |
| `scipy` | ≥1.11 | Spearman 相关等统计检验 | `pip install scipy` |
| `matplotlib` | ≥3.7 | 基础可视化 | `pip install matplotlib` |
| `seaborn` | ≥0.13 | 热力图/分布图 | `pip install seaborn` |
| `mlflow` | ≥2.10 | 实验追踪 | `pip install mlflow` |
| `pytorch-forecasting` | ≥1.1 | `TimeSeriesDataSet` 面板数据组织 | `pip install pytorch-forecasting` |
| `linearmodels` | ≥5.3 | Panel OLS baseline | `pip install linearmodels` |
| `wbgapi` | ≥1.0 | World Bank 数据 API | `pip install wbgapi` |
| `jupyter` | ≥1.0 | Notebook 可视化 | `pip install jupyter` |

## Core 开源项目参考（不安装，仅阅读源码）

| 项目 | 链接 | 参考内容 |
|---|---|---|
| **tft-torch** | https://github.com/PlaytikaOSS/tft-torch | `GatedResidualNetwork` 结构——AC-Gate 的门控设计参考 |
| **pythae** | https://github.com/clementchadebec/benchmark_VAE | VAE 变体实现——消融实验中 VAE 编码器参考 |
| **neuralforecast** | https://github.com/Nixtla/neuralforecast | NHITS/TFT/LSTM baseline 的统一接口（可选替代自写） |

## 一键安装

```bash
# Core 精简集
pip install torch pandas numpy scikit-learn scipy matplotlib seaborn mlflow jupyter
pip install pytorch-forecasting linearmodels wbgapi

# 可选：如用 neuralforecast 跑 TFT baseline
# pip install neuralforecast
```

---

*文档版本：2026-04-12 v2。仅保留依赖信息，开发计划已迁移至 plan.md。*
