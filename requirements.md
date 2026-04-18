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

## 现实数据预处理

> 真实数据 loader（[data/shadow_loader.py](data/shadow_loader.py)、[data/energy_loader.py](data/energy_loader.py)、[data/economics_loader.py](data/economics_loader.py)）必须输出与 [data/synthetic/generate.py](data/synthetic/generate.py) 中 `SyntheticPanel` **结构一致**的张量集合，才能被 [model/cmdl_model.py](model/cmdl_model.py) 直接消费。本节记录适配过程中必须满足的约束与统一处理步骤。

### 一、模型要求的张量结构（硬约束）

| 张量 | 形状 | 含义 | 真实数据对应物（以影子经济域为例） |
|---|---|---|---|
| `X_it` | `[N, T, F]` | 时序输入（驱动变量 / treatment） | 税收负担率、监管强度等年度时序 |
| `p_i` | `[N, M]` | 实体级 AC 代理（**静态**） | WGI 治理指数的时间均值，取 `n_proxies` 维 |
| `s_i` | `[N, S]` | 实体级静态控制特征 | 国家级时不变控制变量 |
| `Y_it` | `[N, T]` | 目标序列 | 影子经济占 GDP 比 |
| `entity_ids` | `[N]` | 0..N-1 连续整数 | 国家编码（必须连续整数，供 `nn.Embedding` lookup） |
| `time_index` | `[T]` | 年份索引 | 1991–2015 |

### 二、6 条强约束（清洗时必须满足）

1. **必须是平衡面板**：所有实体共享同一个 `T`；不平衡数据须先剔除覆盖不足的实体或插值。
2. **`seq_length > max_lag`**（[config 校验](config/cmdl_config.py)）；建议 `seq_length ≥ max_lag + 5`。
3. **`p_i` 必须是实体级静态向量**：真实 proxy 多为时序，需要先按实体聚合（推荐取训练窗口前若干年均值，避免泄漏）。
4. **`n_proxies` 与 config 严格对齐**：若原始 proxy 维度 > `n_proxies`，须先做 PCA 或选最相关维度，或同步修改 config 与 `AdaptiveACEncoder` 入参。
5. **数值尺度必须标准化**：`X_it` / `Y_it` / `p_i` 全部按列做 z-score；保留反标准化参数用于报告原始量纲指标。
6. **行顺序严格对齐**：`p_i[k]`、`s_i[k]`、`X_it[k]`、`Y_it[k]` 必须同属实体 k；merge 后必须 `sort_values + reset_index`。

### 三、统一清洗 pipeline（[data/preprocessing.py](data/preprocessing.py) 应实现的公共函数）

按依赖顺序：

```
1. drop_high_missing(df, entity_col, threshold=0.30)   # 剔除缺失 >30% 的实体/列
2. linear_interpolate(df, group=entity)                # 实体内线性插值剩余缺失
3. align_balanced_panel(df, year_range)                # 截取共同年份，丢弃覆盖不足的实体
4. aggregate_proxy_static(df_long) -> df_static        # 按实体取 proxy 时间均值
5. standardize(df, cols, method='zscore')              # 按列标准化，返回 scaler
6. to_tensors(df_long, df_static, cfg) -> RealPanel    # 转张量；与 SyntheticPanel 字段兼容
```

各 loader 只负责"原始 CSV → long-form DataFrame（columns = entity, year, y, x*, proxy*, static*）"，通用步骤全部下沉到 `preprocessing.py`。中间产物落 parquet。

### 四、与 Step 4 合成数据的关键差异

| 维度 | Step 4 合成 | Step 5 真实 |
|---|---|---|
| `z_true` / `kstar_true` | 有 | **没有**，对应字段设为 `None` |
| 评估指标 | z_spearman / kstar_mae 等识别性指标 | **task RMSE / R²** + baseline 对比（PanelOLS / LSTM / TFT） |
| `n_entities` | 配置固定 200 | 由清洗后实际国家数决定，需**回填到 config**（否则 `entity_embedding` 维度对不上） |
| 训练/测试切分 | 按实体随机切 | **按时间切**（前 80% 年份训练，后 20% 测试），避免时间泄漏 |
| z 表示充分性的指标 | z_spearman_rho（依赖 z_true） | 改用 **proxy_recon_r2** 作为间接证据 |

### 五、Loader 实现的最小可执行下一步

按依赖顺序：

1. 在 [data/preprocessing.py](data/preprocessing.py) 实现上面 6 个通用函数 + 单元测试。
2. 在 [data/shadow_loader.py](data/shadow_loader.py)（当前空文件）实现 Medina-Schneider + WGI 的合并与转换；首先打通主验证域。
3. 在 [config/cmdl_config.py](config/cmdl_config.py) 的 `shadow` / `energy` / `economics` 预设里根据清洗后实际 N 调整 `n_entities`、`seq_length`。
4. 写最小冒烟脚本 [experiments/run_shadow.py](experiments/run_shadow.py)：跑通"加载 → 训练 1 epoch → 预测一次"，验证张量形状与 entity_id 对齐。
5. 通过冒烟后再正式训练并接 baseline 对比。

> **经验提醒**：真实数据 80% 的坑都在张量形状/对齐上，第 4 步冒烟是关键防线，不要跳过。

---

*文档版本：2026-04-18 v3。新增"现实数据预处理"章节，对齐 Step 4 通过后进入 Step 5 的实施需求。*
