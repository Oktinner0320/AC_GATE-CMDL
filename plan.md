# CMDL 开发计划

> 面向 Phase 1（Core）的逐步实施指南，含依赖库、开源项目引用与文献阅读节点。
> 预算：3.5–4 个月（16 周）。每个 Step 标注预计耗时、产出物与阅读任务。
> 依赖库与安装命令见 [requirements.md](requirements.md)。

---

## 一、Step 1 — 环境搭建 + 合成数据生成器（第 1 周）

### 产出物
- `config/cmdl_config.py`
- `data/synthetic/generate.py`
- 验证脚本能生成 200×30 合成面板并可视化

### 工作内容

1. **初始化项目结构**（按 readme.md §七 的 Core 文件树创建空文件）
2. **编写 `CMDLConfig` 数据类**
   - 三个预设：`synthetic` / `energy` / `economics`
   - 核心参数：`max_lag=10`, `d_model=64`, `n_proxies`, `lambda_r=0.1`
3. **编写合成数据生成器** `generate.py`
   - 线性 ground truth: $k^*(z) = \text{round}(3 + 7(1-z))$, $z \sim U[0,1]$
   - 输出: `(X_it, p_i, s_i, Y_it, z_true, kstar_true)`
   - **增加非线性场景**: $k^*(z) = \text{round}(10 \cdot (1-z)^2)$（验证 MLP 非线性能力）
4. **可视化验证**：画 $z$ vs $k^*$ 散点图，确认 ground truth 正确

### 使用的库
- `torch`, `numpy`, `matplotlib`（仅基础依赖）

### 📖 文献阅读（开始前读）

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
   Input: z_i [B, 1]
   Gate: Linear(1, K) → 加相对位置偏置 → 温度缩放 softmax
   Output: ω [B, K] 概率分布
   Context: Σ ω_k · X_{t-k} → [B, d_model]
   k*: Σ k · ω_k → [B]
   ```
3. **单元测试**
   - 验证 ω 求和为 1
   - 验证 k* 范围在 [1, K]
   - 验证梯度可回传到 z_i

### 使用的库
- `torch`（`nn.Module`, `nn.Linear`, `F.softmax`）

### 📖 文献阅读（编码时读）

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
   L = L_task (MSE) + λ_r * L_recon (proxy 重构 MSE)
   仅两项，λ_r 默认 0.1
   ```
3. **组装 `CMDLModel`**
   - 输入适配层 → AC Encoder → Lag Gate → Backbone → RegressionHead
   - 输出: ŷ, ω, z_i, p̂_i, k*
4. **端到端 smoke test**：合成数据 1 个 batch，loss 能下降

### 使用的库
- `torch`（`nn.LSTM`, `nn.Embedding`）

### 📖 文献阅读（组装时参考）

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

### 工作内容

1. **训练循环实现**
   - Adam, lr=1e-3, 200 epochs, early stopping (patience=20)
   - MLflow 记录每轮 loss / k* MAE / proxy recon R²
2. **E1a: 机制验证**
   - 线性 $k^*(z)$: 训练后比对 predicted k* vs ground truth
   - 合格线: k* MAE < 1.0, Spearman $\rho_s > 0.8$
3. **E1b: $z_i$ 识别性验证**
   - 用学到的 $z_i$ 反向预测各 proxy: R² > 0.5
   - $z_i$ vs $z_{true}$ 的 Spearman > 0.8
4. **E1c: 非线性场景**
   - 切换到 $k^*(z) = 10(1-z)^2$，验证 MLP 非线性拟合
5. **生成关键图表**
   - ω 热力图（x=lag k, y=实体按 z 排序, color=ω_k）
   - k* 散点图（predicted vs true）

### 使用的库
- `torch`, `mlflow`, `scipy.stats.spearmanr`, `sklearn.metrics.r2_score`
- `matplotlib`, `seaborn`

### 📖 文献阅读（跑实验时读）

| # | 文献 | 读什么 | 为什么现在读 |
|---|---|---|---|
| **L8** | Babii et al. (2020) "Machine Learning Panel Data Regressions with an Application to Nowcasting Price Earnings Ratios" *arXiv* (被引14) | §2-3 组级滞后选择方法 | 最接近的计量经济学竞品——理解 CMDL 必须超越的 baseline 方法论 |
| **L9** | 黄睿珍 (2023) "Time and Entity Adaptation on Panel Data Forecasting Via Meta Learning" 首尔大学硕士论文 | §3-4 元学习处理实体异质性 | 思路相近但机制不同的竞品；确认 Related Work 中的定位差异 |

---

## 五、Step 5 — 数据下载 + 预处理管线（第 5–6 周）

### 产出物
- `scripts/download_owid.py`
- `scripts/download_pwt.py`
- `data/energy_loader.py`
- `data/economics_loader.py`
- `data/preprocessing.py`
- 两个域的清洗后 DataFrame，存为 parquet

### 工作内容

1. **OWID 能源数据下载与清洗**
   ```python
   # 一行下载
   df = pd.read_csv("https://raw.githubusercontent.com/owid/energy-data/master/owid-energy-data.csv")
   ```
   - 筛选列: `country`, `year`, `renewables_share_energy`, `co2_per_unit_energy`, `gdp`, `population`
   - 排除聚合实体（"World", "OECD", "EU" 等）
   - 处理缺失值：国家级缺失率 >30% 的列/行删除，其余线性插值
   - 时间窗口：1990–2024（可再生份额在 1990 前大部分国家为 0）

2. **OWID CO₂ 数据合并**
   ```python
   co2 = pd.read_csv("https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv")
   ```
   - 提取 `co2_per_gdp`（CO₂强度，作为 $Y$）

3. **World Bank WDI 拉取 AC Proxy**
   ```python
   import wbgapi as wb
   # 治理指数 + 人力资本相关
   indicators = ['SE.XPD.TOTL.GD.ZS',  # 教育支出占GDP比
                 'GB.XPD.RSDV.GD.ZS',  # R&D占GDP比
                 'SP.POP.SCIE.RD.P6']  # 研究人员密度
   ```
   - 取时间均值作为实体级静态 AC proxy

4. **PWT 经济数据下载与清洗**
   ```python
   pwt = pd.read_csv("https://www.rug.nl/ggdc/docs/pwt1001.csv", encoding='utf-8')
   ```
   - 核心变量: `countrycode`, `year`, `ctfp`(TFP), `rtfpna`(相对TFP), `hc`(人力资本), `ck`(资本), `rgdpna`(实际GDP)
   - 计算资本深化: `cap_deepening = ck / rgdpna`
   - AC proxy: `hc`（人力资本指数，PWT 内置）
   - 增加 `rtfpna` 做稳健性检查

5. **统一面板构建** `preprocessing.py`
   - 标准化：时序变量实体内 z-score，proxy 变量跨实体 z-score
   - 构造滑动窗口：`(X_{t-K:t}, p_i, s_i) → Y_{t+1}`
   - 输出 `TimeSeriesDataSet` 兼容格式（或自定义 `torch.utils.data.Dataset`）
   - 训练/验证/测试按时间切分（70/15/15），**不按实体切分**

### 使用的库
- `pandas`, `numpy`, `wbgapi`, `sklearn.preprocessing.StandardScaler`
- `pytorch-forecasting`（`TimeSeriesDataSet`，可选）

### 📖 文献阅读（数据处理时读）

| # | 文献 | 读什么 | 为什么现在读 |
|---|---|---|---|
| **L10** | Akram et al. (2020) "Asymmetric effects of energy efficiency and renewable energy on carbon emissions of BRICS" *ESPR* (被引184) | §3 Panel NARDL 变量构造 + 数据来源 | 学习能源域面板变量的标准构造方式；你的变量选择需与此类文献对齐 |
| **L11** | Mirziyoyeva & Salahodjaev (2022) "Renewable energy and CO₂ emissions intensity in the top carbon intense countries" *Renewable Energy* (被引140) | §2 数据 + §3 面板方法 | 直接竞品的数据处理流程；确认你的 OWID 变量选择合理 |
| **L12** | Feenstra et al. (2015) "The Next Generation of the Penn World Table" *AER* | §2-4 PWT 变量定义 | 正确理解 `ctfp`, `hc`, `ck` 的含义——**误用 PWT 变量是经济学实证论文被拒的常见原因** |

---

## 六、Step 6 — Baseline 实现（第 6–7 周）

### 产出物
- `baselines/panel_ols.py`
- `baselines/tft_baseline.py`
- `baselines/grouped_ardl.py`（新增强 baseline）
- 三个 baseline 在能源域上的初步结果

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
   按 proxy 均值将国家分为 4 组（Q1-Q4）
   每组独立拟合 Panel ARDL → 得到 4 组不同滞后系数
   从系数反推组级 k*: argmax(|cumulative_coef|)
   ```
   - 这是 "人工版 AC-Gate"——如果 AC-Gate 无法显著超越，说服力会大打折扣
   - 使用 `linearmodels`

### 使用的库
- `linearmodels`（PanelOLS, BetweenOLS）
- `pytorch-forecasting`（TemporalFusionTransformer）或 `neuralforecast`
- `statsmodels`（ADF 检验、滞后选择 AIC/BIC）

### 📖 文献阅读（实现 baseline 时读）

| # | 文献 | 读什么 | 为什么现在读 |
|---|---|---|---|
| **L13** | Pesaran & Smith (1995) "Estimating Long-Run Relationships from Dynamic Heterogeneous Panels" *Journal of Econometrics* (经典) | §2-3 异质系数面板模型 | 理解 Pooled Mean Group (PMG) 估计——分组 ARDL baseline 的理论基础 |
| **L14** | pytorch-forecasting 官方教程 "Demand Forecasting with the Temporal Fusion Transformer" | 代码示例 | 快速上手 TFT baseline 的 `TimeSeriesDataSet` 构造 |

---

## 七、Step 7 — 真实数据实验 E2 + E3（第 7–9 周）

### 产出物
- `experiments/run_energy.py`
- `experiments/run_economics.py`
- E2/E3 结果表 + 核心图表

### 工作内容

1. **E2: OWID 能源面板实验**
   - 训练 AC-Gate 模型 + 3 个 baseline
   - 超参搜索：仅调 `lambda_r` ∈ {0.01, 0.05, 0.1, 0.2}，其余锁死
   - 评估指标: MSE / MAE / R² (主指标) + k* 分布分析 (副产品)
   - 关键图表:
     - ω 热力图（按 z_i 分位数分 4 组）
     - k* 跨国分布箱型图
     - z_i 与已知 proxy 的 Spearman 相关

2. **E3: PWT 经济增长面板实验**
   - 同 E2 流程，treatment = 资本深化，outcome = TFP (ctfp)
   - AC proxy = 人力资本指数 (hc)
   - 额外: 用 `rtfpna` 替代 `ctfp` 做稳健性检查
   - 关键对比: E2 与 E3 的 k* 分布模式是否一致（高 AC 实体滞后更短）

3. **跨域对比分析**
   - 合并 E2/E3 的 k* 结果，按国家匹配
   - 散点图: 能源域 k* vs 经济域 k*（预期正相关但非完美线性）
   - 这是论文跨域泛化性的核心证据

### 使用的库
- 同 Step 4 + Step 6 的全部库

### 📖 文献阅读（分析结果时读）

| # | 文献 | 读什么 | 为什么现在读 |
|---|---|---|---|
| **L15** | Appiah-Otoo et al. (2023) "Modelling the impact of renewable energy investment on global CO₂ emissions" *Energy Reports* (被引10) | §3-4 能源投资→CO₂的实证设计 | 对标你的能源域实验设计；确认变量构造和预期方向一致 |
| **L16** | Sheng (2025) "Technological change, capital deepening, and agricultural TFP growth: Cross-country comparison" *AEPP* (被引3) | §2-4 资本深化→TFP 的异质效应 | 最新的 capital deepening→TFP 异质性文献；Direct support for your hypothesis |
| **L17** | Fuentes & Mies (2021) "Technological Absorptive Capacity and Development Stage" *Macroeconomic Dynamics* (被引12) | §3-4 AC与TFP的跨国面板分析 | 你的经济域实验假说的直接理论支撑 |

---

## 八、Step 8 — 消融实验 E4（第 9–10 周）

### 产出物
- `experiments/run_ablation.py`
- 消融对比表（论文 Table 2）

### 工作内容

1. **消融变体 A: 无 AC 编码器**
   - 所有实体共享单一 $\omega$（输入 $z_i$ 替换为全局常数）
   - 预期: ω 分布退化为同质，k* 方差显著下降

2. **消融变体 B: 固定均匀滞后**
   - $\omega_k = 1/K$，AC-Gate 无效化
   - 预期: 预测精度下降，且无法输出有意义的 k*

3. **消融变体 C: 无代理重构正则**
   - $\lambda_r = 0$，$z_i$ 无信息约束
   - 预期: z_i 退化为无语义噪声，proxy 重构 R² 崩塌

4. **统计检验**
   - 每个变体 5 次随机种子运行
   - 报告 mean ± std，Wilcoxon signed-rank test 对比显著性

### 使用的库
- 同 Step 4
- `scipy.stats.wilcoxon`

### 📖 文献阅读

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
   - 问题动机: 异质滞后 + Solow 悖论 + 吸收能力
   - Related Work 三条线:
     - 经典 DLM/ARDL 与面板扩展（L13）
     - 深度学习面板时序模型 TFT/Panel-LSTM（L1, L5, L7）
     - 实体异质性建模：元学习、条件化模型（L9, L8）
   - **必须讨论的竞品**: Babii et al. (2020), 黄 (2023), Zhou et al. (2025)

3. **第 11 周: Method 章节**
   - 形式化 CMDL 任务定义
   - AC-Gate 的完整公式（readme.md §6.3）
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
| **Step 7 (真实实验)** | L15 | Appiah-Otoo 2023 (RE Investment CO₂) | 能源域对标 |
| | L16 | Sheng 2025 (Capital Deepening TFP) | 经济域对标 |
| | L17 | Fuentes & Mies 2021 (AC & TFP) | 理论支撑 |
| **Step 9 (写作)** | L18 | Zhou et al. 2025 (CoDEAL) | 竞品定位 |
| | L19 | Cerqua et al. 2025 (ML Panel Pitfalls) | 方法论警示 |
| | L20 | Thayasivam et al. 2025 (Panel DL Survey) | 最新综述 |

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

## 附录 C：开源项目参考（不安装，仅阅读源码）

| 项目 | 链接 | 参考内容 |
|---|---|---|
| **tft-torch** | https://github.com/PlaytikaOSS/tft-torch | `GatedResidualNetwork` 结构——AC-Gate 的门控设计参考 |
| **pythae** | https://github.com/clementchadebec/benchmark_VAE | VAE 变体实现——消融实验中 VAE 编码器参考 |
| **neuralforecast** | https://github.com/Nixtla/neuralforecast | NHITS/TFT/LSTM baseline 的统一接口（可选替代自写） |

---

*文档版本：2026-04-12 v1。与 readme.md v4 Core/Expansion 结构对齐。*
