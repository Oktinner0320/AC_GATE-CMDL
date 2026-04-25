# CMDL 开发计划

> 面向 Phase 1（Core）的逐步实施指南，含依赖库、开源项目引用与文献阅读节点。
> 预算：3.5–4 个月（16 周）。每个 Step 标注预计耗时、产出物与阅读任务。
> 依赖库与安装命令见 [requirements.md](requirements.md)。
>
> 执行状态更新（2026-04-18）：Step 1–4 已完成并通过 synthetic formal_target；Step 4.5 的 plain-LSTM baseline、核心消融、统一 comparison 与对比图已完成首版实现。当前下一优先级已转为 Step 5 的真实数据预处理与 loader 打通。

---

## 开源项目可复用代码总览

> 下表汇总各阶段可直接复制/修改/调用的开源项目代码，按复用深度排序。
> 每个 Step 的详细修改指南见对应章节的「🔧 可复用开源代码」小节。

| 开源项目 | 核心可复用内容 | 复用深度 | 涉及 Step |
|---|---|---|---|
| **tft-torch** ([PlaytikaOSS](https://github.com/PlaytikaOSS/tft-torch)) | `GatedLinearUnit`（直接复制 ~15 行）、`GatedResidualNetwork`（复制后改造为 LagGate）、`GateAddNorm`（直接复制 ~20 行）、`InputChannelEmbedding`（参考设计模式）、LSTM 初始化模式 | ⭐⭐⭐ 复制+改造 | **Step 2, 3** |
| **pytorch-forecasting** ([sktime](https://github.com/sktime/pytorch-forecasting)) | `TimeSeriesDataSet`（直接实例化，面板数据组织）、`TemporalFusionTransformer`（直接调用作为 TFT baseline） | ⭐⭐⭐ 直接使用 | **Step 5, 6** |
| **pythae** ([clementchadebec](https://github.com/clementchadebec/benchmark_VAE)) | `VAE._sample_gauss` 重参数化（2 行）、KL 散度计算（1 行）、mu/logvar 双头结构、loss 三元组返回模式 | ⭐⭐ 片段复制 | **Step 3, 8** |
| **linearmodels** ([bashtage](https://github.com/bashtage/linearmodels)) | `PanelOLS`（直接调用 ~10 行，Panel OLS baseline） | ⭐⭐ 直接调用 | **Step 6** |
| **neuralforecast** ([Nixtla](https://github.com/Nixtla/neuralforecast)) | `models.TFT`（备选 TFT baseline ~5 行）、`models.LSTM`（备选 LSTM baseline），DataFrame 格式参考 | ⭐ 备选调用 | **Step 5, 6** |
| **mlflow** ([mlflow](https://github.com/mlflow/mlflow)) | `log_metric()` / `log_params()`（直接 API 调用，实验记录） | ⭐ API 调用 | **Step 4–8** |
| **OWID energy-data / co2-data** | CSV URL 直接 `pd.read_csv()` | ⭐ 数据下载 | **Step 1, 5** |
| **wbgapi** | `wb.data.DataFrame()` 一行拉取 WDI 指标 | ⭐ API 调用 | **Step 5** |

---

## 一、Step 1 — 环境搭建 + 合成数据生成器（第 1 周）

### 产出物
- `config/cmdl_config.py`
- `data/synthetic/generate.py`
- 验证脚本能生成 200×30 合成面板并可视化

### 工作内容

1. **初始化项目结构**（按 readme.md §七 的 Core 文件树创建空文件）
2. **编写 `CMDLConfig` 数据类**
   - 四个预设：`synthetic` / `shadow` / `energy` / `economics`
   - 核心参数：`max_lag=10`, `d_model=64`, `n_proxies`, `lambda_r=1.0`
3. **编写合成数据生成器** `generate.py`
   - 线性 ground truth: $k^*(z) = \text{round}(3 + 7(1-z))$, $z \sim U[0,1]$
   - 输出: `(X_it, p_i, s_i, Y_it, z_true, kstar_true)`
   - **增加非线性场景**: $k^*(z) = \text{round}(10 \cdot (1-z)^2)$（验证 MLP 非线性能力）
4. **可视化验证**：画 $z$ vs $k^*$ 散点图，确认 ground truth 正确

### 使用的库
- `torch`, `numpy`, `matplotlib`（仅基础依赖）

### � 可复用开源代码

> 本阶段以自写为主，无需借用外部模型代码。

| 来源 | 文件 / 类 | 复用方式 | 用于 |
|---|---|---|---|
| **OWID energy-data** | 仓库 README 中的 CSV URL | 直接写入 `data/energy/download.py` 的下载地址 | 确认数据列名和格式 |
| **Penn World Table** | 官网 CSV 链接 | 直接写入 `data/economics/download.py` 的下载地址 | 确认 PWT 列名 |

### �📖 文献阅读（开始前读）

> **此时必读**——理解问题定位和现有方法的边界

| # | 文献 | 读什么 | 为什么现在读 |
|---|---|---|---|
| **L1** | Lim et al. (2021) "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting" *IJF* (被引3677) | §3 GatedResidualNetwork 设计 + Variable Selection Network | 理解 TFT 的门控机制——AC-Gate 的主要参考对象；理解为何 TFT 假设同质滞后 |
| **L2** | Schweikl & Obermaier (2020) "Lessons from Three Decades of IT Productivity Research" *Management Review Quarterly* (被引107) | §2-3 关于 lag effect 的文献梳理 | 快速获得 Solow 悖论 + 滞后效应的领域知识，写 Introduction 的素材 |
| **L3** | Cohen & Levinthal (1990) "Absorptive Capacity: A New Perspective on Learning and Innovation" *ASQ* (经典) | 全文，重点 §2 概念定义 | 理解 Absorptive Capacity 的原始含义，确保你的 $z_i$ 设计不偏离学理根基 |

---

## 二、Step 2 — AC 编码器 + Lag Gate 核心模块（第 2–3 周）

### 产出物
- `model/ac_encoder.py`
- `model/lag_gate.py`
- 单元测试：输入随机张量，确认输出形状和梯度正常

### 工作内容

1. **编写 `AdaptiveACEncoder`**（MLP 版）
   ```
   Input: [B, n_proxies]
   Network: Linear(n_proxies, 32) → LayerNorm → GELU → Linear(32, 16) → GELU → Linear(16, 1)
   Output: z_i [B, 1]
   附加: proxy_reconstructor: Linear(1, n_proxies) → p̂_i [B, n_proxies]
   ```
2. **编写 `ScaleInvariantLagGate`**（Core 简化版，去掉时间单位嵌入）
   ```
   Input: z_i = f_phi(p_i) [B, 1]
   Logits: a_i = g_theta(z_i) + b_lag, a_i ∈ R^K
   Transform: omega_i = T_tau(a_i), T ∈ {softmax, sparsemax}
   Effective lag: k_i^* = Σ_{k=1}^K k · omega_{i,k}
   Lag context: c_{i,t} = Σ_{k=1}^K omega_{i,k} · X_{i,t-k}
   ```
3. **单元测试**
   - 验证 ω 求和为 1
   - 验证 k* 范围在 [1, K]
   - 验证梯度可回传到 z_i

### 使用的库
- `torch`（`nn.Module`, `nn.Linear`, `F.softmax`）

### � 可复用开源代码

> ⚡ **本阶段是核心代码借鉴窗口**——AC-Gate 的门控设计直接参考 TFT 的 GRN。

| 来源 | 文件 / 类 | 复用方式 | 用于 |
|---|---|---|---|
| **tft-torch** | [`tft_torch/tft.py → GatedLinearUnit`](https://github.com/PlaytikaOSS/tft-torch/blob/main/tft_torch/tft.py) | **直接复制并精简**。双 `nn.Linear` + sigmoid 门控，代码仅 ~15 行。作为 `ScaleInvariantLagGate` 内部的 GLU 组件 | `model/lag_gate.py` |
| **tft-torch** | [`tft_torch/tft.py → GatedResidualNetwork`](https://github.com/PlaytikaOSS/tft-torch/blob/main/tft_torch/tft.py) | **复制后大幅修改**。保留「input projection → ELU → output projection → GLU → LayerNorm → skip connection」骨架；**去掉** `TimeDistributed` 包装和 `context_dim`；**替换** output_dim 为 `K`（滞后窗口）；**增加**温度缩放 softmax + 相对位置偏置 | `model/lag_gate.py` |
| **tft-torch** | [`tft_torch/base_blocks.py → TimeDistributed`](https://github.com/PlaytikaOSS/tft-torch/blob/main/tft_torch/base_blocks.py) | **可选复制**。若 backbone 需要在时间维度上共享 Linear 层可用；否则用 `einsum` 替代 | `model/backbone.py`（可选） |

**具体修改指南——从 GRN 到 LagGate：**
```python
# tft-torch GRN 原版骨架（约 40 行）：
# fc1(input_dim → hidden_dim) → [+ context_projection] → ELU
# → fc2(hidden_dim → output_dim) → Dropout → GLU → skip → LayerNorm

# 你的 ScaleInvariantLagGate 改动：
# 1. input_dim = 1   (z_i 是标量)
# 2. output_dim = K  (滞后窗口长度，如 10)
# 3. 去掉 context_projection（z_i 本身就是 context）
# 4. 在 fc2 输出后、GLU 前，加入：
#    logits = logits - λ * rel_pos   # rel_pos = arange(K)/K
#    omega = transform(logits / tau)  # softmax 默认，sparsemax 可选
# 5. 输出 omega [B, K]、k_i^* 与 lag context，而非 GRN 的 hidden state
```

### �📖 文献阅读（编码时读）

| # | 文献 | 读什么 | 为什么现在读 |
|---|---|---|---|
| **L4** | Xue et al. (2020) "Not All Attention Is Needed: Gated Attention Network for Sequence Data" *AAAI* (被引56) | §3 门控注意力稀疏化机制 | 直接启发 AC-Gate 的门控设计——如何用 gate 信号选择性加权 |
| **L5** | Yang & Zheng (2020) "Interpretable Neural Networks for Panel Data Analysis in Economics" *arXiv* (被引7) | §2-3 面板数据NN架构设计 | 学习如何在面板结构中嵌入可解释性，与你的 ω/k* 解释性目标对齐 |

---

## 三、Step 3 — LSTM Backbone + 完整模型组装（第 3–4 周）

### 产出物
- `model/backbone.py`
- `model/loss.py`
- `model/cmdl_model.py`
- 能在合成数据上跑通 1 轮 forward + backward

### 工作内容

1. **编写 `UniversalPanelBackbone`**（LSTM 版）
   ```
   Input: concat[context, entity_emb, static, macro] → [B, T, d_input]
   LSTM: 2 层, hidden=d_model, LayerNorm
   Output: hidden [B, d_model]
   ```
2. **编写 `DomainAgnosticLoss`**
   ```
   L = L_task + λ_r L_recon + λ_ω L_entropy_band + λ_z L_z_anchor

   L_task = MSE(ŷ_it, y_it)
   L_recon = MSE(p_hat_i, p_i) 或 anchor-weighted MSE

   H(omega_i) = -Σ_k omega_{i,k} log omega_{i,k}
   L_entropy_band = E_i[(max(0, H_min - H_i) + max(0, H_i - H_max))^2]

   L_z_anchor = max(0, -s_anchor · corr_batch(z_i, p_i^anchor))

   默认 λ_ω = 0, λ_z = 0；因此旧实验行为保持不变。
   ```
3. **组装 `CMDLModel`**
   - 输入适配层 → AC Encoder → Lag Gate → Backbone → RegressionHead
   - 输出: ŷ, ω, z_i, p̂_i, k*
4. **端到端 smoke test**：合成数据 1 个 batch，loss 能下降

### 使用的库
- `torch`（`nn.LSTM`, `nn.Embedding`）

### � 可复用开源代码

| 来源 | 文件 / 类 | 复用方式 | 用于 |
|---|---|---|---|
| **tft-torch** | [`tft.py → past_lstm` 初始化](https://github.com/PlaytikaOSS/tft-torch/blob/main/tft_torch/tft.py) | **参考 LSTM 初始化模式**：`nn.LSTM(input_size, hidden_size, num_layers, dropout, batch_first=True)`。TFT 用两段 LSTM（past+future），你只需单段 | `model/backbone.py` |
| **tft-torch** | [`tft.py → GateAddNorm`](https://github.com/PlaytikaOSS/tft-torch/blob/main/tft_torch/tft.py) | **直接复制**（~20 行）。Dropout → GLU → Residual → LayerNorm 组合。用于 LSTM 输出后的 post-gating | `model/backbone.py` |
| **tft-torch** | [`tft.py → InputChannelEmbedding`](https://github.com/PlaytikaOSS/tft-torch/blob/main/tft_torch/tft.py) + `NumericInputTransformation` | **参考设计模式**：每个 numeric 输入用独立 `nn.Linear(1, state_size)` 投影后拼接。你可简化为 `nn.Linear(d_input, d_model)` + LayerNorm + GELU | `model/cmdl_model.py`（输入适配层） |
| **pythae** | [`vae_model.py → VAE.loss_function`](https://github.com/clementchadebec/benchmark_VAE/blob/main/src/pythae/models/vae/vae_model.py) | **参考 loss 返回结构**。返回 `(total, recon_loss, reg_loss)` 三元组——方便 MLflow 分项记录 | `model/loss.py` |

### �📖 文献阅读（组装时参考）

| # | 文献 | 读什么 | 为什么现在读 |
|---|---|---|---|
| **L6** | tft-torch 源码 `gated_residual_network.py` | GRN 实现细节 + skip connection | 确保你的门控层实现质量不低于 TFT |
| **L7** | Chronopoulos et al. (2023) "Deep Neural Network Estimation in Panel Data Models" *arXiv* (被引15) | §2 面板数据中DNN的固定效应处理 | 学习实体嵌入层的正确做法——你的 `entity_emb` 需借鉴 |

---

## 四、Step 4 — 合成数据实验 E1（第 4–5 周）

### 产出物
- `experiments/run_synthetic.py`
- `evaluation/metrics.py`
- `evaluation/kstar_eval.py`
- E1a/E1b 结果数据 + 图表

### 执行状态更新（2026-04-18）
- `experiments/run_synthetic.py`、评估指标、可视化与 notebook 前端已经完成，`notebooks/01_synthetic_verify.ipynb` 当前可直接复用 `formal_target` 预设重跑 E1a / E1b / E1c。
- 当前 synthetic `formal_target` 三条验收链已经闭合：
   - E1a linear：`kstar_mae = 0.9229`，`kstar_spearman_rho = 0.9805`；
   - E1b identification：`proxy_recon_r2 = 0.9439`，`z_spearman_rho = 0.9892`；
   - E1c nonlinear：`kstar_mae = 0.5348`，`kstar_spearman_rho = 0.9622`。
- 当前 `proxy_recon_r2` 的报告口径已经更新为：best checkpoint 选出后，在冻结 `z_i` 上对线性 proxy head 做闭式重拟合；因此它反映的是 `z_i` 表示的可恢复性，而不是训练期原始 reconstructor 的收敛质量。

### 工作内容

1. **训练循环实现**
   - Adam, lr=1e-3, 200 epochs, early stopping (patience=20)
   - MLflow 记录每轮 loss / k* MAE / proxy recon R²
2. **E1a: 机制验证**
   - 线性 $k^*(z)$: 训练后比对 predicted k* vs ground truth
   - 合格线: k* MAE < 1.0, Spearman $\rho_s > 0.8$
3. **E1b: $z_i$ 识别性验证**
   - 用学到的 $z_i$ 反向预测各 proxy: R² > 0.5
   - 当前实现口径：使用 best checkpoint 后、冻结 `z_i` 上的线性 proxy head refit 来评估该可恢复性
   - $z_i$ vs $z_{true}$ 的 Spearman > 0.8
4. **E1c: 非线性场景**
   - 切换到 $k^*(z) = 10(1-z)^2$，验证 MLP 非线性拟合
5. **生成关键图表**
   - ω 热力图（x=lag k, y=实体按 z 排序, color=ω_k）
   - k* 散点图（predicted vs true）

### 使用的库
- `torch`, `mlflow`, `scipy.stats.spearmanr`, `sklearn.metrics.r2_score`
- `matplotlib`, `seaborn`

### � 可复用开源代码

> 本阶段以 **使用库 API** 为主，无需修改开源项目源码。

| 来源 | 文件 / 类 | 复用方式 | 用于 |
|---|---|---|---|
| **mlflow** | `mlflow.log_metric()` / `mlflow.log_params()` | **直接 API 调用**。每轮 log loss / k* MAE / proxy recon R² | `experiments/run_synthetic.py` |
| **seaborn** | `sns.heatmap()` | **直接调用**。ω 热力图：x=lag k, y=实体按 z 排序 | `visualization/omega_heatmap.py` |

### �📖 文献阅读（跑实验时读）

| # | 文献 | 读什么 | 为什么现在读 |
|---|---|---|---|
| **L8** | Babii et al. (2020) "Machine Learning Panel Data Regressions with an Application to Nowcasting Price Earnings Ratios" *arXiv* (被引14) | §2-3 组级滞后选择方法 | 最接近的计量经济学竞品——理解 CMDL 必须超越的 baseline 方法论 |
| **L9** | 黄睿珍 (2023) "Time and Entity Adaptation on Panel Data Forecasting Via Meta Learning" 首尔大学硕士论文 | §3-4 元学习处理实体异质性 | 思路相近但机制不同的竞品；确认 Related Work 中的定位差异 |

---

## 四点五、Step 4.5 — 合成基线、消融与统一对比（第 5–6 周，已完成首版）

### 产出物
- `baselines/lstm_baseline.py`
- `experiments/run_lstm_baseline.py`
- `experiments/run_ablation.py`
- `evaluation/synthetic_comparison.py`
- `tests/test_lstm_baseline.py`
- `tests/test_ablation_models.py`
- `tests/test_synthetic_comparison.py`
- `notebooks/01_synthetic_verify.ipynb` 中的 Step 4.5 direct run、comparison table 与 comparison figure 单元

### 工作内容

1. **实现 matched plain-LSTM baseline**
   - 保留 entity embedding 与 static conditioning，移除 AC encoder、lag gate 与 proxy reconstruction。
   - 目的不是做最弱 baseline，而是隔离 AC-GATE 的核心机制增益。

2. **实现 baseline 的 post-hoc lag 解释**
   - 使用 lag occlusion 构造 per-entity lag profile，并定义 pseudo-k*。
   - 在统一 comparison 中将其映射为 `effective_kstar_*` 与 `effective_lag_*`，与 CMDL / ablation 直接对齐。

3. **实现三组 synthetic 核心消融**
   - `no_ac_encoder`：所有实体共享一个全局 `z` 与共享 lag 分布。
   - `uniform_lag`：固定 `omega_k = 1 / K`。
   - `no_recon_regularization`：移除 reconstruction loss，仅保留 task loss。

4. **统一 comparison 报告与 notebook 图表**
   - 扫描 CMDL、baseline 与 ablation 的 `summary.json`，构造 recovery / identification 两张统一对比表。
   - 生成 `recovery_comparison.png` 与 `identification_comparison.png`，作为 Step 4.5 正式图表产物。

5. **当前 formal_target 首版结论**
   - 完整 CMDL 相比 plain LSTM，在 linear / nonlinear 两个场景下都显著提升有效滞后恢复。
   - `no_ac_encoder` 与 `uniform_lag` 都会显著破坏 k* 恢复，说明主要增益来自 AC conditioning 与 adaptive lag gating。
   - `no_recon_regularization` 与完整 CMDL 几乎重合，说明当前 synthetic formal_target 中的主要收益并不来自 reconstruction regularization。

### 使用的库
- 同 Step 4
- `pandas`（统一 summary 读取与 comparison table 构造）

### 当前状态说明
- Step 4.5 首版已经完成并通过 notebook formal_target 复核。
- 当前 notebook formal_target 对比默认验证的是单 seed（42）；多 seed 聚合保留到论文表格阶段。

---

## 五、Step 5 — 数据下载 + 预处理管线（第 5–6 周）

### 产出物
- `data/shadow/download.py`
- `data/energy/download.py`
- `data/economics/download.py`
- `data/shadow/shadow_loader.py`
- `data/energy/energy_loader.py`
- `data/economics/economics_loader.py`
- 三个域的清洗后 DataFrame，存为 parquet（各域 loader 内独立实现）

### 工作内容

1. **影子经济数据下载与清洗**（主验证域）
   - 数据源: Medina & Schneider (2018, IMF WP/18/17)，158 国 × 1991–2015
   - 核心变量: 影子经济占 GDP 比 ($Y$)
   - Treatment $X_t$: 税收负担率、监管强度指数（来自 WDI / WGI）
   - AC Proxy $Z_i$: WGI 治理指数（法治、监管质量、腐败控制等 6 维），取时间均值作为实体级静态 proxy
   - 处理缺失值：国家级缺失率 >30% 的列/行删除，其余线性插值

2. **OWID 能源数据下载与清洗**（泛化域 1）
   ```python
   # 一行下载
   df = pd.read_csv("https://raw.githubusercontent.com/owid/energy-data/master/owid-energy-data.csv")
   ```
   - 筛选列: `country`, `year`, `renewables_share_energy`, `co2_per_unit_energy`, `gdp`, `population`
   - 排除聚合实体（"World", "OECD", "EU" 等）
   - 处理缺失值：国家级缺失率 >30% 的列/行删除，其余线性插值
   - 时间窗口：1990–2024（可再生份额在 1990 前大部分国家为 0）

2. **OWID 能源数据下载与清洗**（泛化域 1）
   ```python
   co2 = pd.read_csv("https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv")
   ```
   - 提取 `co2_per_gdp`（CO₂强度，作为 $Y$）

3. **OWID CO₂ 数据合并**
   ```python
   import wbgapi as wb
   # 治理指数 + 人力资本相关
   indicators = ['SE.XPD.TOTL.GD.ZS',  # 教育支出占GDP比
                 'GB.XPD.RSDV.GD.ZS',  # R&D占GDP比
                 'SP.POP.SCIE.RD.P6']  # 研究人员密度
   ```
   - 取时间均值作为实体级静态 AC proxy

4. **World Bank WDI/WGI 拉取 AC Proxy**（影子经济域 + 能源域共用）
   ```python
   pwt = pd.read_csv("https://www.rug.nl/ggdc/docs/pwt1001.csv", encoding='utf-8')
   ```
   - 核心变量: `countrycode`, `year`, `ctfp`(TFP), `rtfpna`(相对TFP), `hc`(人力资本), `ck`(资本), `rgdpna`(实际GDP)
   - 计算资本深化: `cap_deepening = ck / rgdpna`
   - AC proxy: `hc`（人力资本指数，PWT 内置）
   - 增加 `rtfpna` 做稳健性检查

5. **PWT 经济数据下载与清洗**（泛化域 2）
6. **各域独立预处理**（分别在各域 loader 内实现）
   - 标准化：时序变量实体内 z-score，proxy 变量跨实体 z-score
   - 输出 `TimeSeriesDataSet` 兼容格式（或自定义 `torch.utils.data.Dataset`）
   - 训练/验证/测试按时间切分（70/15/15），**不按实体切分**

### 使用的库
- `pandas`, `numpy`, `wbgapi`, `sklearn.preprocessing.StandardScaler`
- `pytorch-forecasting`（`TimeSeriesDataSet`，可选）

### � 可复用开源代码

| 来源 | 文件 / 类 | 复用方式 | 用于 |
|---|---|---|---|
| **pytorch-forecasting** | [`TimeSeriesDataSet`](https://github.com/sktime/pytorch-forecasting) | **直接实例化使用**（首选）。自动处理 static/time-varying 分离、滑动窗口、entity 分组、时间索引对齐 | 各域 loader（`data/shadow/`、`data/energy/`、`data/economics/`） |
| **pytorch-forecasting** | `TimeSeriesDataSet.from_dataset()` | **直接调用**。从训练集自动生成验证集（共享 normalization） | 各域 loader |
| **neuralforecast** | [`neuralforecast/`](https://github.com/Nixtla/neuralforecast) 的 DataFrame 格式 | **参考数据格式**（备选）。要求列名 `unique_id`/`ds`/`y`，如后续想用其 TFT baseline需兼容 | 各域 loader（可选） |
| **OWID energy-data** | [GitHub CSV](https://github.com/owid/energy-data) | **直接 `pd.read_csv(url)`**。无需 clone，列名文档见仓库 README | `data/energy/download.py` |
| **OWID co2-data** | [GitHub CSV](https://github.com/owid/co2-data) | **直接 `pd.read_csv(url)`** | `data/energy/download.py` |
| **wbgapi** | `wb.data.DataFrame()` | **直接 API 调用**，一行拉取 World Bank 指标 | `data/shadow/download.py`（AC proxy） |

**关键代码片段——pytorch-forecasting 面板构造：**
```python
from pytorch_forecasting import TimeSeriesDataSet

training = TimeSeriesDataSet(
    df_train,
    time_idx="year",
    target="co2_per_gdp",                # Y
    group_ids=["country"],                 # 实体标识
    max_encoder_length=10,                 # K=10 滞后窗口
    max_prediction_length=1,               # 单步预测
    static_reals=["hc", "rnd_gdp"],        # AC proxy 作为 static
    time_varying_unknown_reals=["renewables_share"],  # treatment X_t
    add_relative_time_idx=True,
    add_encoder_length=True,
)
```

### �📖 文献阅读（数据处理时读）

| # | 文献 | 读什么 | 为什么现在读 |
|---|---|---|---|
| **L10** | Akram et al. (2020) "Asymmetric effects of energy efficiency and renewable energy on carbon emissions of BRICS" *ESPR* (被引184) | §3 Panel NARDL 变量构造 + 数据来源 | 学习能源域面板变量的标准构造方式；你的变量选择需与此类文献对齐 |
| **L11** | Mirziyoyeva & Salahodjaev (2022) "Renewable energy and CO₂ emissions intensity in the top carbon intense countries" *Renewable Energy* (被引140) | §2 数据 + §3 面板方法 | 直接竞品的数据处理流程；确认你的 OWID 变量选择合理 |
| **L12** | Feenstra et al. (2015) "The Next Generation of the Penn World Table" *AER* | §2-4 PWT 变量定义 | 正确理解 `ctfp`, `hc`, `ck` 的含义——**误用 PWT 变量是经济学实证论文被拒的常见原因** |

---

## 六、Step 6 — 真实域 Baseline 实现（第 6–7 周；synthetic Plain LSTM 已提前完成）

### 产出物
- `baselines/panel_ols.py`
- `baselines/tft_baseline.py`
- `baselines/grouped_ardl.py`（新增强 baseline）
- 三个 baseline 在影子经济域上的初步结果

### 执行状态更新（2026-04-18）
- `baselines/lstm_baseline.py` 与 `experiments/run_lstm_baseline.py` 已提前用于 Step 4.5 synthetic comparison。
- 因此当前 Step 6 剩余重点不是再补一个 synthetic LSTM，而是把 `PanelOLS`、TFT 与 grouped ARDL 打通到真实数据域。

### 工作内容

1. **Panel OLS + 固定滞后**
   ```python
   from linearmodels.panel import PanelOLS
   # Y ~ lag1_X + lag2_X + ... + lagK_X + EntityEffects
   ```
   - 固定滞后 K=10，所有实体共享系数

2. **TFT Baseline**
   - 方案 A（推荐）：用 `pytorch-forecasting` 的 `TemporalFusionTransformer`
   - 方案 B（备选）：用 `neuralforecast` 的 TFT 接口
   - 关键：喂入相同的特征集（含 proxy 作为 static covariates），公平比较

3. **分组 ARDL（强 baseline，新增）** ⚡
   ```
   按 anchor proxy 的训练期分位数将实体分为 low / mid / high 三组
   每组独立拟合 distributed-lag OLS:
   y_it = α_g + Σ_{k=1}^K β_{g,k} x_{i,t-k} + ε_it
   组级 best lag: k_g^best = argmax_k |β_{g,k}|
   组级 effective lag: k_g^eff = Σ_k k · |β_{g,k}| / Σ_k |β_{g,k}|
   ```
   - 这是“人工分组版异质滞后”强 baseline，用于校准 AC-GATE 的连续 AC-conditioned lag gate。
   - 如果 Grouped ARDL 预测更强，论文应降低 forecasting superiority claim，但仍可比较 AC-GATE 的实体级连续 `omega/k*` 解释能力。
   - 当前实现优先使用轻量 distributed-lag OLS，避免额外依赖；后续可再接入 `statsmodels` ARDL。

### 使用的库
- `linearmodels`（PanelOLS, BetweenOLS）
- `pytorch-forecasting`（TemporalFusionTransformer）或 `neuralforecast`
- `statsmodels`（ADF 检验、滞后选择 AIC/BIC）

### � 可复用开源代码

> ⚡ **本阶段是库“拿来即用”密度最高的步骤**—3 个 baseline 均可大幅借用现有实现。

| 来源 | 文件 / 类 | 复用方式 | 用于 |
|---|---|---|---|
| **linearmodels** | [`PanelOLS`](https://github.com/bashtage/linearmodels/blob/main/linearmodels/panel/model.py) | **直接调用**（~10 行）。设 `entity_effects=True`，构造 `lag1_X … lagK_X` 列即可 | `baselines/panel_ols.py` |
| **pytorch-forecasting** | [`TemporalFusionTransformer`](https://github.com/sktime/pytorch-forecasting) | **直接调用 `.from_dataset()`**（首选）。proxy 作为 `static_reals`，训练用 PyTorch Lightning `Trainer` | `baselines/tft_baseline.py` |
| **neuralforecast** | [`models.TFT`](https://github.com/Nixtla/neuralforecast/blob/main/neuralforecast/models/tft.py) | **备选**（~5 行）。接口极简但 static covariates 支持不如 pytorch-forecasting | `baselines/tft_baseline.py`（备选） |
| **neuralforecast** | [`models.LSTM`](https://github.com/Nixtla/neuralforecast/blob/main/neuralforecast/models/lstm.py) | **直接调用**。作为“标准 LSTM（无 Gate）”baseline，可替代自写 | `baselines/lstm_baseline.py`（可选） |
| **statsmodels** | `adfuller` | **直接调用**。ADF 单位根检验 | `baselines/grouped_ardl.py` 前置 |

**pytorch-forecasting vs neuralforecast 对比：**

| | pytorch-forecasting TFT | neuralforecast TFT |
|---|---|---|
| 静态协变量 | ✅ `static_reals` / `static_categoricals` | ⚠️ 有限支持 |
| 训练框架 | PyTorch Lightning | 自封装 |
| 代码量 | ~30 行 | ~5 行 |
| 可解释性输出 | ✅ attention weights、variable importance | ⚠️ 较少 |
| **推荐** | ✅ **首选**（与论文需求匹配） | 备选 |

### �📖 文献阅读（实现 baseline 时读）

| # | 文献 | 读什么 | 为什么现在读 |
|---|---|---|---|
| **L13** | Pesaran & Smith (1995) "Estimating Long-Run Relationships from Dynamic Heterogeneous Panels" *Journal of Econometrics* (经典) | §2-3 异质系数面板模型 | 理解 Pooled Mean Group (PMG) 估计——分组 ARDL baseline 的理论基础 |
| **L14** | pytorch-forecasting 官方教程 "Demand Forecasting with the Temporal Fusion Transformer" | 代码示例 | 快速上手 TFT baseline 的 `TimeSeriesDataSet` 构造 |

---

## 七、Step 7 — 真实数据实验 E2 + E3 + E4（第 7–9 周）

### 产出物
- `experiments/run_shadow.py`
- `experiments/run_energy.py`
- `experiments/run_economics.py`
- E2/E3/E4 结果表 + 核心图表

### 工作内容

1. **E2: 影子经济面板实验**（主验证域，回应核心 RQ）
   - 训练 AC-Gate 模型 + 3 个 baseline
   - Treatment = 税收负担率/监管强度，Outcome = 影子经济占比
   - AC proxy = WGI 治理指数（法治、监管质量、腐败控制）
   - 超参搜索：仅调 `lambda_r` ∈ {0.01, 0.05, 0.1, 0.2}，其余锁死
   - 关键图表:
     - ω 热力图（按 z_i 分位数分 4 组）
     - k* 跨国分布箱型图
     - k* 与治理质量指数的 Spearman 相关
   - 预期结果：高治理质量国家的正规经济政策向影子经济传导更快（k* 更短）

2. **E3: OWID 能源面板实验**（泛化域 1）
   - 同 E2 流程，treatment = 可再生份额，outcome = CO₂ 强度
   - 评估指标: MSE / MAE / R² (主指标) + k* 分布分析 (副产品)

3. **E4: PWT 经济增长面板实验**（泛化域 2）
   - treatment = 资本深化，outcome = TFP (ctfp)
   - AC proxy = 人力资本指数 (hc)
   - 额外: 用 `rtfpna` 替代 `ctfp` 做稳健性检查

4. **跨域对比分析**
   - 合并 E2/E3/E4 的 k* 结果，按国家匹配
   - 三域 k* 分布对比：影子经济 vs 能源 vs 经济增长
   - 这是论文跨域泛化性的核心证据

### 使用的库
- 同 Step 4 + Step 6 的全部库

### � 可复用开源代码

> 本阶段无新增代码借用；复用 Step 4 的训练循环和 Step 6 的 baseline 实现。

### �📖 文献阅读（分析结果时读）

| # | 文献 | 读什么 | 为什么现在读 |
|---|---|---|---|
| **L15** | Schneider & Enste (2000) "Shadow Economies: Size, Causes, and Consequences" *JEL* (经典) | §2-3 影子经济的定义、成因与度量方法 | 影子经济领域的基础文献，确保你的变量定义与领域共识一致 |
| **L16** | Medina & Schneider (2018) "Shadow Economies Around the World" *IMF WP/18/17* (被引 500+) | §2 MIMIC 方法 + §4 国家级估计结果 | 你的主数据源，理解估计方法的假设与局限性 |
| **L17** | Elgin et al. (2021) "Understanding informality" *SEPS* (被引 160+) | §3 DGE 法与 MIMIC 法的比较 | 备选数据源，理解不同估计方法的差异 |
| **L18** | Appiah-Otoo et al. (2023) "Modelling the impact of renewable energy investment on global CO₂ emissions" *Energy Reports* (被引 10) | §3-4 能源投资→CO₂的实证设计 | 对标能源域实验设计；确认变量构造和预期方向一致 |
| **L19** | Sheng (2025) "Technological change, capital deepening, and agricultural TFP growth" *AEPP* (被引 3) | §2-4 资本深化→TFP 的异质效应 | 经济域实验假说的直接支撑 |

---

## 八、Step 8 — 消融实验 E5（第 9–10 周；synthetic 核心消融已完成首版）

### 产出物
- `experiments/run_ablation.py`
- 消融对比表（论文 Table 2）

### 执行状态更新（2026-04-18）
- `experiments/run_ablation.py` 已实现 `no_ac_encoder`、`uniform_lag` 与 `no_recon_regularization` 三个变体，并已在 synthetic formal_target 下完成 linear / nonlinear 两个场景的首版验证。
- 当前 notebook formal_target 首版 comparison 使用单 seed（42）；脚本已支持多 seed 聚合，但 `5 seed + Wilcoxon` 仍保留到论文表 2 定稿阶段。

### 工作内容

1. **消融变体 A: 无 AC 编码器**
   - 所有实体共享单一 $\omega$（输入 $z_i$ 替换为全局常数）
   - 预期: ω 分布退化为同质，k* 方差显著下降

2. **消融变体 B: 固定均匀滞后**
   - $\omega_k = 1/K$，AC-Gate 无效化
   - 预期: 预测精度下降，且无法输出有意义的 k*

3. **消融变体 C: 无代理重构正则**
   - $\lambda_r = 0$，$z_i$ 无信息约束
   - 当前证据更新: 在 synthetic formal_target 下，该变体与完整 CMDL 的 k* / z / proxy 指标几乎重合。
   - 因此它更适合作为“主要增益不来自 reconstruction regularization”的证据，而不是预设为必然崩塌的失败基线。

4. **统计检验**
   - 每个变体 5 次随机种子运行
   - 当前代码已经支持 `seeds` 多值输入与聚合；formal_target 首版仅验证 seed=42，后续再补 `mean ± std` 与 Wilcoxon signed-rank test

### 使用的库
- 同 Step 4
- `scipy.stats.wilcoxon`

### � 可复用开源代码

| 来源 | 文件 / 类 | 复用方式 | 用于 |
|---|---|---|---|
| **pythae** | [`vae_model.py → VAE`](https://github.com/clementchadebec/benchmark_VAE/blob/main/src/pythae/models/vae/vae_model.py) | **复制核心片段并改造**。用于消融变体“VAE AC Encoder”：(1) `_sample_gauss(mu, std)` 重参数化（2 行）；(2) KL 散度 `KLD = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))`（1 行）；(3) mu/logvar 双头结构 | `model/ac_encoder.py`（VAE 变体分支） |
| **pythae** | [`vae_config.py → VAEConfig`](https://github.com/clementchadebec/benchmark_VAE/blob/main/src/pythae/models/vae/vae_config.py) | **参考 config 结构**。`latent_dim`、`reconstruction_loss` 枚举等可借鉴到 `CMDLConfig` | `config/cmdl_config.py`（可选） |

**从 pythae VAE → AC Encoder VAE 变体的改造指南：**
```python
# pythae VAE 原版：编码图像 → mu, log_var → z [B, latent_dim]
# 你的 VAE AC Encoder 改造：
# 1. 输入 = p_i [B, n_proxies]（而非图像）
# 2. latent_dim = 1（z_i 是标量 AC 得分）
# 3. 网络: Linear(n_proxies, 32) → GELU → Linear(32, 16)
#    → mu_head: Linear(16, 1)
#    → logvar_head: Linear(16, 1)
# 4. 重参数化：z_i = mu + std * eps（复制 pythae._sample_gauss）
# 5. 额外 KL 项加入 loss：
#    L = L_task + λ_r * L_recon + β(t) * λ_kl * KLD
#    β(t) = min(1, t/T_warmup)  ← KL annealing
```

### �📖 文献阅读

> 此时无新增必读文献，专注实验执行。如消融结果异常，回看 L4 (门控机制) 和 L7 (面板DNN固定效应)。

---

## 九、Step 9 — 论文撰写（第 10–13 周）

### 产出物
- Workshop 论文初稿（8 页，LaTeX）
- 所有图表的 publication-ready 版本

### 工作内容

1. **论文骨架（第 10 周）**
   ```
   1. Introduction              (1 页)
   2. Related Work               (0.75 页)
   3. Method: AC-Gate            (2 页)
   4. Experiments                (3 页)
   5. Discussion & Future Work   (0.75 页)
   6. Conclusion                 (0.5 页)
   ```

2. **第 10 周: Introduction + Related Work**
   - 问题动机: 正规—非正规经济时滞 + 制度异质性导致不同传导速度
   - Related Work 四条线（回应老师邮件要求）:
     - 含滞后的计算模型: DLM/ARDL, delay DE, LSTM, Granger causality (L1, L13)
     - 经济过程间时滞的经济理论: 政策传导滞后、制度质量与调整速度 (L15, L17)
     - 社会经济过程中特定时滞的实证研究 (L16, L18, L19)
     - 非正规经济规模与社会经济过程的关系 (L15, L16, L17)
   - **必须讨论的竞品**: Babii et al. (2020), Zhou et al. (2025)

3. **第 11 周: Method 章节**
   - 形式化 CMDL 任务定义
   - AC-GATE 的完整公式需使用当前改进版：
     - entity proxy encoder: $z_i=f_\phi(p_i)$, $\hat p_i=r_\psi(z_i)$；
     - lag logits: $a_i=g_\theta(z_i)+b_{lag}$；
     - lag distribution: $\omega_i=T_\tau(a_i)$, $T\in\{\mathrm{softmax},\mathrm{sparsemax}\}$；
     - effective lag: $k_i^*=\sum_{k=1}^{K} k\omega_{i,k}$；
     - lag context: $c_{i,t}=\sum_{k=1}^{K}\omega_{i,k}X_{i,t-k}$；
     - objective: $\mathcal L=\mathcal L_{task}+\lambda_r\mathcal L_{recon}+\lambda_\omega\mathcal L_{entropy}+\lambda_z\mathcal L_{z-anchor}$，其中后两项默认关闭。
   - 损失函数与训练细节

4. **第 11–12 周: Experiments 章节**
   - 从 E1-E4 提取关键表格和图
   - 写法: "为什么需要神经网络" ← **正面回答 N=180 足以支撑的理由**
     - 核心论点: 模型的价值在于发现异质滞后结构（ω/k*），而非仅压缩 MSE
     - 传统方法无法输出实体级滞后分布

5. **第 12–13 周: Discussion + 修改**
   - 局限性: 观测数据非因果、proxy 信息泄漏风险
   - Future Work: 因果扩展、VAE 编码器、多时间粒度
   - 全文 polish + 请导师/同行 review

### 使用的库
- LaTeX（推荐 Overleaf 或 VS Code + LaTeX Workshop）

### 📖 文献阅读（写作时补充）

| # | 文献 | 读什么 | 为什么现在读 |
|---|---|---|---|
| **L18** | Zhou et al. (2025) "Covariate-Adjusted Deep Causal Learning for Heterogeneous Panel Data Models" *arXiv* (被引1) | §2-3 异质处理效应的深度学习方法 | Related Work 中因果方向的竞品定位 |
| **L19** | Cerqua et al. (2025) "On the (mis)use of machine learning with panel data" *OBES* (被引14) | §3-4 ML面板应用的常见陷阱 | 避免审稿人已知的方法论错误；强化 Discussion 的自我批判 |
| **L20** | Thayasivam et al. (2025) "A Comprehensive Survey on Statistical and Deep Learning Models for Panel Data Analysis" *KAIS* | 全文浏览 | 最新综述——确保 Related Work 不遗漏最新进展 |

---

## 十、Step 10 — 投稿 + 申博材料（第 13–16 周）

### 产出物
- 最终论文 PDF
- Research Proposal（2-3 页，用于博士申请）

### 工作内容

1. **论文定稿与投稿**
   - 目标 Venue: KDD MiLeTS Workshop (通常 5-6 月截稿) 或 AAAI Workshop (通常 8-9 月截稿)
   - 如时间不赶上 KDD，转投 AAAI Workshop 或 NeurIPS Workshop (9 月)
   - 代码整理 + README，准备 GitHub repo（审稿人可能要求）

2. **申博 Research Proposal**
   - 将 AC-Gate 定位在更大议程中: "Conditional Heterogeneous Panel Learning"
   - Phase 2 蓝图: 企业级面板 + VAE + 因果扩展
   - 准备退化定理的非正式证明（面试可能被问）:
     > 当 $z_i = c$ (常数) 时, $\omega(k|c)$ 对所有实体相同 → AC-Gate 退化为 homogeneous DLM

3. **面试准备**
   - 准备 5 分钟 presentation slides
   - 预备问题: "为什么不用因果推断?" → 指向 Discussion section
   - 预备问题: "样本量足够吗?" → 指向合成数据验证 + 模型参数量分析

---

## 附录 A：文献阅读总索引

> 按阅读时间排序，共 20 篇。前 3 篇为开始编码前必读。

| 阶段 | 编号 | 文献简称 | 类型 |
|---|---|---|---|
| **Step 1 (开始前)** | L1 | Lim et al. 2021 (TFT) | 方法参考 |
| | L2 | Schweikl & Obermaier 2020 (IT Productivity Survey) | 领域综述 |
| | L3 | Cohen & Levinthal 1990 (Absorptive Capacity) | 理论基础 |
| **Step 2 (编码时)** | L4 | Xue et al. 2020 (Gated Attention, AAAI) | 门控机制 |
| | L5 | Yang & Zheng 2020 (Interpretable Panel NN) | 面板NN设计 |
| **Step 3 (组装时)** | L6 | tft-torch 源码 GRN | 实现参考 |
| | L7 | Chronopoulos et al. 2023 (DNN Panel) | 面板DNN |
| **Step 4 (合成实验)** | L8 | Babii et al. 2020 (ML Panel Regression) | 计量竞品 |
| | L9 | 黄睿珍 2023 (Meta-Learning Panel) | ML竞品 |
| **Step 5 (数据处理)** | L10 | Akram et al. 2020 (Energy NARDL) | 能源域参考 |
| | L11 | Mirziyoyeva 2022 (RE & CO₂) | 能源域参考 |
| | L12 | Feenstra et al. 2015 (PWT Guide) | PWT使用指南 |
| **Step 6 (Baseline)** | L13 | Pesaran & Smith 1995 (PMG) | 理论基础 |
| | L14 | pytorch-forecasting TFT Tutorial | 代码教程 |
| **Step 7 (真实实验)** | L15 | Schneider & Enste 2000 (影子经济综述) | 影子经济基础 |
| | L16 | Medina & Schneider 2018 (IMF WP) | 主数据源 |
| | L17 | Elgin et al. 2021 (DGE法) | 备选数据源 |
| | L18 | Appiah-Otoo 2023 (RE Investment CO₂) | 能源域对标 |
| | L19 | Sheng 2025 (Capital Deepening TFP) | 经济域对标 |
| **Step 9 (写作)** | L20 | Zhou et al. 2025 (CoDEAL) | 竞品定位 |
| | L21 | Cerqua et al. 2025 (ML Panel Pitfalls) | 方法论警示 |
| | L22 | Thayasivam et al. 2025 (Panel DL Survey) | 最新综述 |

---

## 附录 B：甘特图总览

```
Week  1  2  3  4  5  6  7  8  9  10  11  12  13  14  15  16
Step1 ██
Step2    ████
Step3       ████
Step4          ████
Step5             ████
Step6                ████
Step7                   ██████
Step8                         ████
Step9                            ████████
Step10                                     ████████

📖 L1-L3  ██
📖 L4-L5     ████
📖 L6-L7        ████
📖 L8-L9           ████
📖 L10-12              ████
📖 L13-14                 ████
📖 L15-17                    ██████
📖 L18-20                            ████████
```

---

## 附录 C：开源项目可复用文件索引

> 本附录将各阶段提及的可复用代码精确到**源文件和类名**，方便直接定位。

### C.1 tft-torch（⭐ 核心参考，Step 2–3）

| 源文件 | 类/函数 | 代码量 | 你的用法 | 目标文件 |
|---|---|---|---|---|
| `tft_torch/tft.py` | `GatedLinearUnit` | ~15 行 | **直接复制**。双 Linear + sigmoid 门控 | `model/lag_gate.py` |
| `tft_torch/tft.py` | `GatedResidualNetwork` | ~40 行 | **复制 + 改造**。替换 output_dim→K，加温度 softmax + 位置偏置 | `model/lag_gate.py` |
| `tft_torch/tft.py` | `GateAddNorm` | ~20 行 | **直接复制**。Dropout→GLU→Residual→LayerNorm | `model/backbone.py` |
| `tft_torch/tft.py` | `InputChannelEmbedding` | ~50 行 | **参考设计**。每变量独立 Linear(1, d) 后拼接 | `model/cmdl_model.py` |
| `tft_torch/tft.py` | `NumericInputTransformation` | ~20 行 | **参考设计**。可简化为单个 Linear | `model/cmdl_model.py` |
| `tft_torch/tft.py` | `TemporalFusionTransformer.__init__` 中 `past_lstm` | ~5 行 | **参考初始化**。nn.LSTM 参数配置 | `model/backbone.py` |
| `tft_torch/base_blocks.py` | `TimeDistributed` | ~25 行 | **可选复制**。batch+time 维度合并后共享 Linear | `model/backbone.py` |

### C.2 pythae（Step 3 参考 / Step 8 消融用）

| 源文件 | 类/函数 | 代码量 | 你的用法 | 目标文件 |
|---|---|---|---|---|
| `src/pythae/models/vae/vae_model.py` | `VAE.forward` | ~15 行 | **参考流程**。encoder → mu/logvar → reparameterize → decode → loss | `model/ac_encoder.py`（VAE 变体） |
| 同上 | `VAE._sample_gauss` | 2 行 | **直接复制**。`mu + eps * std` | `model/ac_encoder.py` |
| 同上 | `VAE.loss_function` | ~10 行 | **复制 KL 项**。`KLD = -0.5 * sum(1 + log_var - mu² - exp(log_var))`；**参考返回模式** `(total, recon, reg)` | `model/loss.py` |
| `src/pythae/models/vae/vae_config.py` | `VAEConfig` | ~20 行 | **参考 config 字段**。`latent_dim`, `reconstruction_loss` | `config/cmdl_config.py` |

### C.3 pytorch-forecasting（Step 5–6 直接使用）

| 模块 | 类 | 你的用法 | 目标文件 |
|---|---|---|---|
| `pytorch_forecasting` | `TimeSeriesDataSet` | **直接实例化**。面板数据→DataLoader，自动处理窗口和归一化 | 各域 loader（`data/shadow/`、`data/energy/`、`data/economics/`） |
| 同上 | `TimeSeriesDataSet.from_dataset()` | **直接调用**。验证集构造 | 各域 loader |
| 同上 | `TemporalFusionTransformer` | **直接调用 `.from_dataset()`**。TFT baseline | `baselines/tft_baseline.py` |

### C.4 linearmodels（Step 6 直接使用）

| 模块 | 类 | 你的用法 | 目标文件 |
|---|---|---|---|
| `linearmodels.panel` | `PanelOLS` | **直接调用**。Panel OLS + entity effects baseline | `baselines/panel_ols.py` |

### C.5 neuralforecast（Step 6 备选）

| 模块 | 类 | 你的用法 | 目标文件 |
|---|---|---|---|
| `neuralforecast.models` | `TFT` | 备选 TFT baseline（接口极简，~5 行） | `baselines/tft_baseline.py` |
| `neuralforecast.models` | `LSTM` | 备选 LSTM baseline（替代自写） | `baselines/lstm_baseline.py` |

---

*文档版本：2026-04-18 v4。补充 Step 4 已完成状态与当前 formal_target 口径；新增 Step 4.5 合成 baseline / ablation / comparison 首版；修正 λ_r 默认值、Step 6 baseline 现状与 Step 8 无重构正则的预期表述。与 readme.md v5.0 Core/Expansion 结构对齐。*
