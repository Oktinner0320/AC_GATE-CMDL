# CMDL 项目总览文档

> AC-Gate：面向面板时序数据的条件调节型分布式滞后模型

---

## 〇、项目定位

> **首要目标**：完成一篇可发表的 Workshop / Applied Track 论文，支撑博士申请。
> **执行者**：研究生（申博阶段），时间预算 3–4 个月。
> **核心策略**：先跑通最小可发表集合（Phase 1），再按需扩展至主会投稿。

本文档采用 **Core / Expansion 双层结构**：
- **Core（Phase 1）**：合成数据 + 影子经济域（主验证）+ 能源域 + 经济域（泛化验证），支撑 Workshop 投稿与博士申请
- **Expansion（Phase 2+）**：教育域、IT 专项数据、高级模型变体，供博士入学后延续

标记说明：
- 无标记 = Core 内容，必须完成
- 🔮 = Expansion 内容，Phase 1 不做

---

## 一、项目目标

### 核心研究问题

**How can temporal delays between formal and informal economic processes be modeled?**

正规经济政策与指标（如税收负担、监管强度、GDP 增长）对非正规经济（影子经济）规模的影响存在显著时滞，且不同国家因制度环境差异，时滞长度呈现系统性异质性。本项目提出 **AC-Gate**（Adaptive Conditioning Gated Lag Network）——一种以实体级调节变量为条件、自适应学习滞后权重分布的神经网络机制，解决面板时序数据中**条件调节型分布式滞后预测（CMDL）**问题。

### 方法论目标

- 形式化定义 CMDL 任务：在实体级调节变量 $Z_i$ 存在的条件下，联合学习调节表示、条件滞后权重分布 $\omega(k|z_i)$ 和下游结果预测，端到端优化
- 展示现有模型（ARDL、TFT、Panel-LSTM）的结构性局限：假设同质滞后，无法捕捉实体间异质性
- 提供可解释副产品：最优滞后期 $k^* = \sum_k k \cdot \omega_k$，无需直接监督
- 通过代理重构和合成数据 ground truth 比对，验证 $z_i$ 的语义可解释性

### 应用目标

**主验证域（影子经济）**：正规经济指标（税负、监管）向非正规经济规模的传导存在显著时滞，且时滞长度与国家制度质量（治理水平、执法效率、金融包容度）呈系统性关联。预期高制度质量国家的 $k^*$ 显著短于低制度质量国家。

**泛化验证域**：在能源转型（可再生投资 → CO₂ 强度）和经济增长（资本深化 → TFP）两个独立域上验证模型的跨领域泛化能力。

> **重要声明**：本项目聚焦于发现和量化 **预测性异质滞后模式（predictive heterogeneous lag pattern）**，而非严格因果推断。因果语义的讨论作为论文 Discussion 部分的开放问题处理。

### 发表目标

以领域问题（正规—非正规经济时滞建模）为驱动，方法论贡献为核心，跨域泛化验证为加分项。

| 投稿策略 | 目标 Venue | 说明 | 阶段 |
|---|---|---|---|
| **一稳** | KDD Workshop (MiLeTS) / AAAI Workshop | Phase 1 核心实验即可支撑 | Core |
| 一冲 | KDD Applied Data Science Track | 强调跨域应用价值 | Core + 补充实验 |
| 跨界 | AAAI AI for Social Impact / ICAIF | 影子经济 + AI 交叉领域 | Core |
| 🔮 跨界 | ICML CML Workshop / NeurIPS CML Workshop | 因果机器学习社区 | Expansion |

---

## 二、研究对象

### 研究单元结构

```
面板数据结构：实体 i × 时间 t
  实体维度：国家（Core）/ 行业 / 企业（Expansion）
  时间维度：年（Core）/ 月 / 小时（Expansion）
  面板类型：平衡（Core）/ 不平衡 / 不规则（Expansion）
```

### Core 验证域（Phase 1 必须完成）

| 域 | 角色 | 数据集 | 实体 | 时间跨度 | 规模 | 核心问题 |
|---|---|---|---|---|---|---|
| **合成数据** | 机制验证 | 代码生成 | 200 实体 | 30 期 | 6,000 | 有 ground truth，机制验证 |
| **影子经济** | 🔴 主验证域 | Medina & Schneider + WGI | 158 国家 | 1991–2015（~25 年） | ~3,950 | 正规经济指标 → 影子经济规模滞后 |
| **能源转型** | 🟢 泛化域 1 | OWID energy + CO₂ | 180+ 国家 | 1965–2024（~60 年） | ~10,800 | 可再生投资 → CO₂强度下降滞后 |
| **经济增长** | 🟢 泛化域 2 | Penn World Table 10 | 150+ 国家 | 1950–2019（~70 年） | ~10,500 | 资本深化 → TFP 增长滞后 |

> **主验证域选择理由**：影子经济域直接回应核心 RQ——“如何建模正规与非正规经济过程之间的时间延迟”。Medina & Schneider (2018, IMF WP/18/17) 提供 158 国 MIMIC 法估计的影子经济 GDP 占比，单 CSV 即可加载。治理质量指标（WGI）作为 $Z_i$ 代理，反映制度环境对时滞传导速度的调节作用。
>
> **跨域泛化策略**：能源域与经济域的实体重合度高（均为国家级面板），但 treatment/outcome 完全不同。若 AC-Gate 在三个独立的 input→outcome 关系上均能学到异质滞后模式，则泛化性强于仅用双域验证。

### 🔮 Expansion 验证域（Phase 2+）

| 域 | 数据集 | 实体 | 时间粒度 | 主要问题 | 额外成本 |
|---|---|---|---|---|---|
| 🔮 IT 生产率 | OECD STAN + MSTI | 38 国 | 年 | IT 资本 → TFP（原 Solow 悖论主题） | 需 OECD 账号拼表，N=38 需轻量化配置 |
| 🔮 教育人力资本 | PISA + EdStats | 80 国 | 3 年 | 教育支出 → 测评成绩滞后 | PISA 仅 6 轮，面板极短 |
| 🔮 企业级面板 | Compustat / ORBIS | 5000+ 企业 | 季度 | R&D → 营收增长滞后 | 需付费数据库，但 N 大适合 VAE |

---

## 三、变量体系

> 以下为泛化指标定义，括号内为各域的具体实例

---

### 3.1 时序输入变量 $X_t$（Input Treatment Sequence）

**类型**：连续型时序变量，随时间变化，为模型输入核心序列

> **术语说明**：本项目使用 "treatment" 指代感兴趣的输入变量，不隐含因果声明。在观测数据设置中，$X_t$ 与 $Y_{t+k}$ 的关系为预测性关联，而非因果关系。

| 泛化指标 | 类型 | 影子经济域示例 | 经济域示例 | 能源域示例 |
|---|---|---|---|---|
| 主输入强度 | 连续 / 比率 | 税收负担率（税收/GDP） | IT 资本支出占 GDP 比 | 可再生能源份额 |
| 输入分类1 | 连续 | 监管强度指数 | 软件投资额 | 太阳能装机容量 |
| 输入分类2 | 连续 | 劳动市场管制指数 | 硬件投资额 | 风能装机容量 |
| 历史输入积累 | 连续（滚动和） | 5年平均税负 | 5年 IT 存量 | 历史累计投资 |
| 输入增长率 | 连续（差分） | 税负同比变化 | IT 投资同比增长 | 可再生增速 |

---

### 3.2 实体调节变量代理指标 $\mathbf{p}_i$（AC Proxies）

**类型**：连续型静态或缓变变量，反映实体的"转化能力"，用于编码潜在 $Z_i$

| 泛化指标 | 类型 | 影子经济域示例 | 经济域示例 | 能源域示例 |
|---|---|---|---|---|
| 知识积累能力 | 连续 / 比率 | 执法效率指数 | R&D 支出占 GDP 比 | 绿色 R&D 投入 |
| 人力资本质量 | 连续（指数） | 受教育年限 | STEM 毕业生比例 / 人力资本指数 | 工程师密度 |
| 组织学习强度 | 连续 | 金融包容度指数 | 企业培训投入强度 | 能源管理体系普及率 |
| 历史技术积累 | 连续 | 制度历史稳定性 | 历史 IT 存量（5年均值） | 历史电网覆盖率 |
| 制度质量 | 有序 / 连续 | WGI 治理指数（核心 proxy） | WB 治理指数 | 监管质量指数 |

> **注**：代理指标数量 $M$ 由 `CMDLConfig.n_proxies` 配置，模型自适应处理 $M \in [2, 8]$

---

### 3.3 实体静态特征 $\mathbf{s}_i$（Static Entity Features）

**类型**：连续 / 类别型静态变量，作为实体嵌入的输入

| 泛化指标 | 类型 | 说明 |
|---|---|---|
| 实体规模 | 连续（对数） | 人口 / 员工数 / 床位数 |
| 经济发展水平 | 连续 | 人均 GDP（PPP） |
| 地理 / 气候分组 | 类别 | 区域虚拟变量 |
| 实体 ID 嵌入 | 整数索引 | 学习固定效应嵌入向量 |

---

### 3.4 宏观控制变量 $\mathbf{c}_t$（Time-Varying Controls）

**类型**：连续型时变变量，控制混淆因素

| 泛化指标 | 类型 | 经济域示例 |
|---|---|---|
| 总体经济状态 | 连续 | GDP 增长率 |
| 价格水平 | 连续 | CPI / PPI |
| 对外开放程度 | 连续 | 贸易开放度 |
| 金融市场条件 | 连续 | 实际利率 |

---

### 3.5 结果变量 $Y_{t+k}$（Outcome）

**类型**：连续值回归（Core）；其他类型为 Expansion

| 输出头类型 | 适用结果 | 示例 | 损失函数 | 阶段 |
|---|---|---|---|---|
| `RegressionHead` | 连续值 | 影子经济占比 / TFP 增长率 / CO₂强度 | MSE | **Core** |
| 🔮 `BinaryHead` | 二元 | 是否达标 | BCE | Expansion |
| 🔮 `SurvivalHead` | 时间到事件 | ICU 时长 | Cox 偏似然 | Expansion |
| 🔮 `CountHead` | 计数（零膨胀） | 专利数 | ZIP 负对数似然 | Expansion |
| 🔮 `MultiHorizonHead` | 多步同时预测 | $t+1 \ldots t+K$ | MSE（多输出） | Expansion |

---

### 3.6 潜在变量与模型内生输出（Latent & Derived）

| 变量 | 类型 | 含义 | 来源 |
|---|---|---|---|
| $z_i$ | 连续潜在变量（1维） | 实体吸收能力得分 | VAE 编码器输出 |
| $\omega_k$ | 概率分布（K维） | 各滞后期权重 | AC-Gate softmax |
| $k^*$ | 连续标量 | 期望最优滞后期 | $\sum_k k \cdot \omega_k$（推断后计算，非训练目标） |
| $k^*_\text{rel}$ | $[0,1]$ 归一化 | 跨域可比的相对滞后 | $k^* / K$ |

**$z_i$ 可识别性验证输出：**

| 验证指标 | 方法 | 合格阈值 | 阶段 |
|---|---|---|---|
| 代理重构误差 | 用 $z_i$ 反向预测各 proxy 指标，报告 $R^2$ | $R^2 > 0.5$ | **Core** |
| 秩相关一致性 | $z_i$ 与参考 proxy 的 Spearman 相关 | $\rho_s > 0.7$ | **Core** |
| 合成数据 ground truth | $z_i$ 与真实 AC 的 Spearman 相关 | $\rho_s > 0.8$ | **Core** |
| 消融实验 | 移除 AC Encoder，观察 $\omega$ 分布是否退化为同质 | $\omega$ 方差显著下降 | **Core** |
| 🔮 解耦评估 | MIG / DCI 指标（需多维 $z_i$） | MIG > 0.3 | Expansion |

---

## 四、推荐开源模型

> 仅列出 PyTorch 生态实现，排除 TensorFlow。标注 Core / Expansion。

---

### 4.1 数据处理层

| 项目 | 链接 | 用途 | 阶段 |
|---|---|---|---|
| **pytorch-forecasting** | https://github.com/sktime/pytorch-forecasting | `TimeSeriesDataSet` 面板数据组织，静态/时变协变量分离 | **Core** |
| 🔮 **nowcast_lstm** | https://github.com/dhopp1/nowcast_lstm | 宏观面板不规则滞后处理，ragged data 核心逻辑 | Expansion |
| 🔮 **tsai** | https://github.com/timeseriesAI/tsai | 多变量时序数据管道，基于 PyTorch + fastai | Expansion |

### 4.2 AC Encoder 基础

| 项目 | 链接 | 用途 | 阶段 |
|---|---|---|---|
| **pythae** | https://github.com/clementchadebec/benchmark_VAE | 20+ VAE 变体，消融实验中 VAE 变体参考 | **Core**（消融用） |
| 🔮 **nflows** | https://github.com/bayesiains/nflows | 归一化流，VAE 效果不足时的升级替代方案 | Expansion |
| 🔮 **disentanglement_lib** | https://github.com/google-research/disentanglement_lib | 解耦评估指标（MIG、DCI） | Expansion |

### 4.3 注意力机制 / 门控设计（AC-Gate 实现参考）

| 项目 | 链接 | 用途 | 阶段 |
|---|---|---|---|
| **tft-torch** | https://github.com/PlaytikaOSS/tft-torch | `GatedResidualNetwork` 结构参考 + TFT baseline | **Core** |
| 🔮 **x-transformers** | https://github.com/lucidrains/x-transformers | `CrossAttention`、条件注意力变体工具库 | Expansion |
| 🔮 **flash-attention** | https://github.com/Dao-AILab/flash-attention | 大规模面板时内存优化（N>1000 实体时使用） | Expansion |

### 4.4 Baseline 与评估

| 项目 | 链接 | 用途 | 阶段 |
|---|---|---|---|
| **linearmodels** | https://github.com/bashtage/linearmodels | Panel OLS baseline（计量经济学对照） | **Core** |
| **neuralforecast** | https://github.com/Nixtla/neuralforecast | 统一接口运行 NHITS / TFT / LSTM baseline | **Core** |
| **mlflow** | https://github.com/mlflow/mlflow | 消融实验记录，超参数与指标管理 | **Core** |
| 🔮 **EconML** | https://github.com/py-why/EconML | Double ML、因果森林（因果扩展对比） | Expansion |
| 🔮 **causalnex** | https://github.com/mckinsey/causalnex | NOTEARS DAG 学习 | Expansion |
| 🔮 **shap** | https://github.com/slundberg/shap | `DeepExplainer` 特征归因 | Expansion |
| 🔮 **captum** | https://github.com/pytorch/captum | PyTorch 原生归因 | Expansion |

---

## 五、开源数据集

> 均无需资质认证，可直接下载或通过 API 获取。标注 Core / Expansion。

---

### 5.1 合成基准数据集（Core，优先）

| 名称 | 获取方式 | CMDL 映射 |
|---|---|---|
| **自生成合成数据** | 代码生成（无需下载） | 唯一有 $k^*$ ground truth 的方式，验证机制正确性 |

```python
# 示例：generate_cmdl_synthetic(n_entities=200, T=30, max_lag=10)
# 真实 k*(z) = round(3 + 7*(1-z)), z ~ U[0,1]
```

---

### 5.2 影子经济面板（Core，主验证域）

| 名称 | 链接 | 覆盖 | CMDL 映射 |
|---|---|---|---|
| **Medina & Schneider (2018)** | IMF Working Paper 18/17 | 158国 × 1991-2015 | $X_t$: 税收负担率、监管强度；$Y$: 影子经济占 GDP 比 |
| **World Bank WGI** | https://info.worldbank.org/governance/wgi/ | 200国 × 1996-2024 | $Z_i$ 代理来源（治理质量、执法效率、监管质量）|
| **Elgin et al. (2021)** | DGE-based 估计 | 161国 × 1950-2018 | 备选影子经济度量（更长时间跨度） |

> **影子经济数据说明**：Medina & Schneider (2018) 使用 MIMIC (Multiple Indicators Multiple Causes) 方法估计影子经济规模，是该领域最广泛引用的数据源。WGI 治理指数包含 6 个维度（言论自由、政治稳定、政府效能、监管质量、法治、腐败控制），可用作 $Z_i$ 的多维代理指标。

---

### 5.3 能源转型面板（Core，泛化验证域 1）

| 名称 | 链接 | 覆盖 | CMDL 映射 |
|---|---|---|---|
| **OWID 能源数据** | https://github.com/owid/energy-data | 200国 × 1965-2024 | $X_t$: 可再生份额；$Y$: CO₂强度 |
| **OWID CO₂数据** | https://github.com/owid/co2-data | 200国 × 1750-2024 | 精细碳排放分项，与能源数据合并使用 |
| **World Bank WDI** | https://data.worldbank.org （API: `wbgapi`） | 200国 × 1960-2024 | $Z_i$ 代理来源（治理指数、人力资本）|

```python
# OWID 能源数据一行加载
df = pd.read_csv("https://raw.githubusercontent.com/owid/energy-data/master/owid-energy-data.csv")

# World Bank API
import wbgapi as wb
df = wb.data.DataFrame(['EG.ELC.RNEW.ZS', 'EN.ATM.CO2E.PC', 'NY.GDP.PCAP.PP.KD'], economy='all')
```

---

### 5.4 经济增长面板（Core，泛化验证域 2）

| 名称 | 链接 | 覆盖 | CMDL 映射 |
|---|---|---|---|
| **Penn World Table 10** | https://www.rug.nl/ggdc/docs/pwt1001.csv | 183国 × 1950-2019 | $X_t$: 资本深化（`ck`/`rgdpna`）；$Y$: TFP（`ctfp`）；$Z_i$: 人力资本（`hc`） |

> **PWT 的优势**：一个 CSV 文件同时包含 treatment ($X$)、outcome ($Y$) 和 AC proxy ($Z$)，无需拼表。这是成本最低的泛化验证域。

```python
# PWT 一行加载
df = pd.read_csv("https://www.rug.nl/ggdc/docs/pwt1001.csv", encoding='utf-8')
# 核心变量: countrycode, year, ctfp, hc, ck, rgdpna
```

---

### 🔮 5.5 Expansion 数据集（Phase 2+）

| 名称 | 链接 | 覆盖 | CMDL 映射 | 额外成本 |
|---|---|---|---|---|
| 🔮 OECD Data Explorer | https://data-explorer.oecd.org | 38国 × 1970-2023 | IT 资本、劳动生产率 | 需免费账号，拼表复杂 |
| 🔮 OWID 教育/R&D | https://ourworldindata.org/grapher/ | 150国 × 1970-2022 | 受教育年限、R&D 支出占 GDP 比 | 低 |
| 🔮 PISA 国际测评 | https://www.oecd.org/pisa/data/ | 80国 × 6轮（2000-2022） | 教育支出 → 测评分 | PISA 仅 6 轮，面板极短 |
| 🔮 World Bank EdStats | https://data.worldbank.org/topic/education | 200国 × 1970-2024 | 中高等教育入学率 | 低 |

---

## 六、泛化模型设计结构

### 6.1 核心设计原则

**Core 版本追求最小复杂度**：默认使用确定性 MLP 编码器 + LSTM backbone，仅保留 `RegressionHead`。VAE 编码器和 Transformer backbone 作为消融/Expansion 变体。

```python
# Core: 两行启动
cfg = CMDLConfig.from_domain("shadow")  # 或 "energy" / "economics" / "synthetic"
model = CMDLModel(cfg)

# 🔮 Expansion: 多域切换
# cfg = CMDLConfig.from_domain("education")
# cfg = CMDLConfig.from_domain("clinical")
```

---

### 6.2 模块架构

> `★` = Core 核心贡献　`🔮` = Expansion 变体

```
┌─────────────────────────────────────────────────────────┐
│                    输入适配层（薄）                        │
│  UniversalInputAdapter                                   │
│  ├── seq_encoder      时序特征 → d_model                 │
│  │   ├── Core: Linear + LayerNorm + GELU                │
│  │   └── 🔮 不规则时序: GRU-D                           │
│  ├── static_encoder   静态特征 → d_model/2              │
│  └── proxy_normalizer AC代理跨实体z-score标准化          │
└───────────────────────┬─────────────────────────────────┘
                        │ [B, T, d_model] + [B, d_model/2]
┌───────────────────────▼─────────────────────────────────┐
│                  CMDL 核心引擎（厚）                      │
│                                                          │
│  AdaptiveACEncoder ★                                    │
│  ├── 输入: [B, n_proxies]（n_proxies 由配置决定）        │
│  ├── Core 默认: MLP(proxies → z_i)  ← 确定性映射        │
│  │   网络: Linear→LayerNorm→GELU→Linear→GELU→z_i [B,1] │
│  ├── 🔮 VAE 变体: mu_head / logvar_head → 重参数化采样   │
│  ├── ★ 代理重构头: z_i → p̂_i [B, n_proxies]            │
│  │   （确保 z_i 保留 proxy 信息）                       │
│  └── 🔮 KL Annealing: β 从 0→1（仅 VAE 变体）          │
│                                                          │
│  ScaleInvariantLagGate ★★（核心贡献）                  │
│  ├── 时间单位嵌入: Embedding(4种粒度, 8维)              │
│  ├── 门控网络: [z_i; time_unit_emb] → logits[B,K]      │
│  ├── 相对位置偏置: rel_pos = k/K ∈ [0,1]               │
│  ├── 温度缩放softmax → ω [B, K]                        │
│  ├── 加权历史聚合: context = Σ ω_k · X_{t-k}           │
│  └── k* 输出: abs（步数）& rel（归一化，跨域可比）      │
│                                                          │
│  UniversalPanelBackbone                                 │
│  ├── 实体嵌入层: Embedding(n_entities, 8)               │
│  ├── Core 默认: LSTM 2层，hidden=d_model，LayerNorm     │
│  ├── 🔮 Transformer: 4头自注意力（大规模场景）          │
│  └── 输入 = concat[context, entity_emb, static, macro]  │
│                                                          │
│  DomainAgnosticLoss ★                                   │
│  ├── L_task   主预测损失（MSE）                          │
│  ├── L_recon  ★ 代理重构损失（z_i → p̂_i 的 MSE）       │
│  ├── L_total  = L_task + λ_r·L_recon  ← Core 仅两项     │
│  ├── 🔮 + λ_a·L_anchor（AC锚定，可选）                  │
│  └── 🔮 + β(t)·λ_kl·L_KL（VAE 变体）                   │
└───────────────────────┬─────────────────────────────────┘
                        │ hidden [B, d_model]
┌───────────────────────▼─────────────────────────────────┐
│                 输出头层（薄·可插拔）                     │
│  ├── ★ RegressionHead   连续值（TFP / CO₂）            │
│  ├── 🔮 BinaryHead      二元分类                        │
│  ├── 🔮 SurvivalHead    时间到事件                      │
│  ├── 🔮 CountHead       计数（零膨胀泊松）              │
│  └── 🔮 MultiHorizonHead 多步预测                       │
│                                                          │
│  共享副产品输出（所有域）：                              │
│  ├── ω 分布 [B, K]     →  论文 Figure 1（热力图）       │
│  ├── k* 估计 [B]       →  论文 Figure 2（跨实体分布）   │
│  ├── z_i 得分 [B]      →  论文 Figure 3（AC排名）       │
│  └── p̂_i 重构 [B, M]  →  论文 Figure 4（识别性验证）   │
└─────────────────────────────────────────────────────────┘
```

---

### 6.3 核心公式

$$\omega(k \mid z_i) = \text{Softmax}\!\left(\frac{f_\theta(z_i) - \lambda \cdot \tilde{k}}{\tau}\right)$$

$$\hat{Y}_{i,t+k} = g_\phi\!\left(\sum_{k=1}^{K} \omega_k(z_i) \cdot h(X_{i,t-k}),\; \mathbf{s}_i,\; \mathbf{c}_t\right)$$

$$k^* = \sum_{k=1}^{K} k \cdot \omega_k(z_i) \qquad \text{（推断后计算，非训练目标）}$$

**Core 损失函数（仅两项）：**

$$\mathcal{L} = \underbrace{\mathcal{L}_\text{task}}_{\text{MSE}} + \lambda_r \underbrace{\mathcal{L}_\text{recon}}_{\text{代理重构}}$$

$$\mathcal{L}_\text{recon} = \|\hat{\mathbf{p}}_i - \mathbf{p}_i\|^2 \qquad \text{（确保 $z_i$ 保留 proxy 信息）}$$

> 超参数仅 1 个：$\lambda_r$（代理重构权重），默认值 0.1

**🔮 Expansion 损失函数（VAE 变体，增加 KL 和锚定项）：**

$$\mathcal{L}_\text{full} = \mathcal{L}_\text{task} + \lambda_a \mathcal{L}_\text{anchor} + \lambda_r \mathcal{L}_\text{recon} + \beta(t) \cdot \lambda_{kl} \mathcal{L}_\text{KL}$$

$$\beta(t) = \min\!\left(1,\; \frac{t}{T_{\text{warmup}}}\right) \qquad \text{（KL Annealing，缓解后验退化）}$$

---

## 七、文件结构

### Core 文件结构（Phase 1 实现范围）

```
CMDL/
│
├── README.md                        项目说明（本文档）
├── requirements.md                  依赖清单与安装命令
├── plan.md                          开发步骤、文献阅读节点与甘特图
├── worklog.md                       我的工作日志
│
├── config/
│   └── cmdl_config.py               CMDLConfig 数据类，含 synthetic / shadow / energy / economics 四个预设
│
├── data/
│   ├── synthetic/
│   │   └── generate.py              合成数据生成（线性 k* 函数）
│   ├── shadow/
│   │   ├── download.py              下载影子经济 + WGI 数据
│   │   └── shadow_loader.py         Medina & Schneider 影子经济 + WGI 加载与预处理
│   ├── energy/
│   │   ├── download.py              一键下载 OWID 数据集
│   │   └── energy_loader.py         OWID energy + CO₂ 加载与预处理
│   └── economics/
│       ├── download.py              下载 Penn World Table
│       └── economics_loader.py      PWT 加载与预处理
│
├── model/
│   ├── ac_encoder.py                MLP 编码器（默认）+ VAE 编码器（消融用）
│   ├── lag_gate.py                  ScaleInvariantLagGate（AC-Gate，核心贡献）
│   ├── backbone.py                  LSTM backbone
│   ├── loss.py                      L_task + L_recon（两项）
│   └── cmdl_model.py                CMDLModel 主类
│
├── baselines/
│   ├── panel_ols.py                 Panel OLS + 固定滞后（linearmodels）
│   ├── lstm_baseline.py             标准 LSTM（无 Gate）
│   └── tft_baseline.py              标准 TFT（tft-torch）
│
├── experiments/
│   ├── run_synthetic.py             E1: 合成数据验证
│   ├── run_shadow.py                E2: 影子经济面板（主验证）
│   ├── run_energy.py                E3: OWID 能源面板（泛化）
│   ├── run_economics.py             E4: PWT 经济增长面板（泛化）
│   └── run_ablation.py              E5: 消融实验（3 变体）
│
├── evaluation/
│   ├── metrics.py                   MSE / MAE / R² / Spearman
│   └── kstar_eval.py                k* 恢复评估
│
├── visualization/
│   ├── omega_heatmap.py             ω 分布热力图（按 AC 分位数分组）
│   └── kstar_distribution.py        k* 跨实体分布图
│
└── notebooks/
    ├── 01_synthetic_verify.ipynb    合成数据机制验证
    └── 02_real_data_results.ipynb   真实数据结果 + 论文图表
```

> Core 文件数：**~23 个**

### 🔮 Expansion 扩展文件（Phase 2+ 按需添加）

```
├── 🔮 data/
│   ├── loaders/
│   │   ├── base_loader.py           抽象基类
│   │   ├── education_loader.py      PISA + EdStats
│   │   └── oecd_it_loader.py        OECD IT 专项数据
│   └── preprocessing/
│       └── ragged_handler.py        不规则面板处理（GRU-D 配套）
│
├── 🔮 model/
│   ├── adapters/
│   │   └── output_heads.py          Binary / Survival / Count / MultiHorizon
│   └── core/
│       └── transformer_backbone.py  Transformer 平行 backbone
│
├── 🔮 baselines/
│   └── econml_baseline.py           Double ML / 因果森林
│
├── 🔮 experiments/
│   ├── E5_education.py
│   ├── E6_oecd_it.py
│   └── E7_full_ablation.py          5 变体 × 全部数据集
│
├── 🔮 evaluation/
│   ├── zi_identification.py         MIG / DCI 解耦评估
│   └── counterfactual.py            预测性敏感度分析
│
└── 🔮 notebooks/
    ├── 03_education.ipynb
    └── 04_extended_figures.ipynb
```

---

## 八、快速启动

> 完整的分步开发指南（含每步产出物、使用的库、文献阅读节点）见 **[plan.md](plan.md)**。
> 依赖安装命令见 **[requirements.md](requirements.md)**。

```bash
# 1. 安装依赖（见 requirements.md）
pip install torch pandas numpy scikit-learn scipy matplotlib seaborn mlflow jupyter
pip install pytorch-forecasting linearmodels wbgapi

# 2. 下载数据
python data/shadow/download.py        # 影子经济 + WGI 治理指数
python data/energy/download.py        # OWID 能源 + CO₂
python data/economics/download.py     # Penn World Table（一个 CSV）

# === Phase 1：Core 实验（对应 plan.md Step 1–10） ===

# 3. 合成数据验证（plan.md Step 1–4）
python experiments/run_synthetic.py

# 4. 影子经济主实验（plan.md Step 5–7）
python experiments/run_shadow.py

# 5. 能源域泛化验证（plan.md Step 7）
python experiments/run_energy.py

# 6. 经济域泛化验证（plan.md Step 7）
python experiments/run_economics.py

# 7. 消融实验（plan.md Step 8）
python experiments/run_ablation.py

# 7. 生成论文图表
jupyter notebook notebooks/01_synthetic_verify.ipynb
jupyter notebook notebooks/02_real_data_results.ipynb

# === 🔮 Phase 2+：Expansion（博士入学后） ===
# pip install pythae causalnex econml shap captum
# python experiments/E5_education.py
# python experiments/E6_oecd_it.py
```

---

## 九、论文实验规划（分阶段）

> 每个实验的详细实施步骤、所需库与文献阅读节点见 **[plan.md](plan.md)**。

### Phase 1：Core 实验（支撑 Workshop 投稿 + 博士申请，3-4 个月）

| 实验 | 数据集 | 目的 | 对应论文章节 | 月份 |
|---|---|---|---|---|
| **E1a** | 合成（线性 $k^*$） | 验证 AC-Gate 机制正确性，有 ground truth | §4.1 | 第 1 月 |
| **E1b** | 合成（$z_i$ 识别性） | 验证 proxy 重构 $R^2$、Spearman 达标 | §4.1 | 第 1 月 |
| **E2** | 影子经济面板 | 主验证域（158 国 × 25 年，回应核心 RQ） | §4.2 | 第 2 月 |
| **E3** | OWID 能源面板 | 泛化验证域 1（180 国 × 60 年） | §4.3 | 第 2 月 |
| **E4** | PWT 经济增长面板 | 泛化验证域 2（183 国 × 70 年） | §4.3 | 第 2 月 |
| **E5** | 消融实验（3 核心变体） | 各组件贡献量化 | §4.4 | 第 3 月 |
| — | 论文撰写 + 投稿 | Workshop 8 页论文 | 全文 | 第 3-4 月 |

**Core 消融变体（3 个）：**
1. 无 AC 编码器（$\omega$ 退化为同质，所有实体共享滞后权重）
2. 固定均匀滞后（$\omega_k = 1/K$，AC-Gate 无效化）
3. 无代理重构正则（$\lambda_r = 0$，$z_i$ 无约束）

**Core Baseline（3 个）：**

| Baseline | 工具 | 代表什么 |
|---|---|---|
| Panel OLS + 固定滞后 | `linearmodels` | 传统计量经济学 |
| 标准 LSTM（无 Gate） | 自写 | 深度学习无 AC 条件 |
| TFT（原版） | `tft-torch` / `neuralforecast` | 当前 SOTA 面板时序模型 |

### 🔮 Phase 2：Expansion 实验（博士入学后或主会投稿补充）

| 实验 | 数据集 | 目的 | 优先级 |
|---|---|---|---|
| 🔮 E6 | OECD 38 国 IT 专项 | Solow 悖论原始假设验证（小样本，轻量化配置） | P1 |
| 🔮 E7 | PISA 教育面板 | 跨域泛化（异构时间粒度） | P2 |
| 🔮 E8 | 企业级面板（Compustat） | 大 N 场景，VAE 编码器主场 | P2 |
| 🔮 E9 | 扩展消融（5 变体 × 全数据集） | +两阶段训练 / +无KL annealing | P1 |

**🔮 Expansion 消融增量（在 Core 3 变体基础上）：**
4. 两阶段训练（先 VAE 后 Gate，对比端到端）
5. 无 KL annealing（$\beta$ 恒为 1）

### 核心评估指标体系

| 维度 | 指标 | 说明 | 阶段 |
|---|---|---|---|
| 预测精度 | MSE / MAE / $R^2$ | 标准回归指标 | **Core** |
| 滞后恢复 | $k^*$ MAE / Spearman-$\rho$ | 合成数据独有，核心贡献指标 | **Core** |
| $z_i$ 质量 | Proxy 重构 $R^2$ / Spearman | 潜在表示可识别性 | **Core** |
| 异质性捕获 | $\omega$ 方差（跨实体） | AC-Gate 的核心价值验证 | **Core** |
| 🔮 解耦质量 | MIG / DCI | 多维潜变量解耦评估 | Expansion |

### Core 论文结构（Workshop 8 页）

```
1. Introduction              (1 页)    正规—非正规经济时滞问题 + 动机
2. Related Work               (0.75 页)  延迟建模方法、经济时滞理论、影子经济实证、实体异质性
3. Method: AC-Gate            (2 页)    问题定义 + 模型 + 损失函数
4. Experiments                (3 页)
   4.1 合成数据验证            k* 恢复 MAE + ω 热力图
   4.2 影子经济面板          预测精度 + k* 与制度质量的关联
   4.3 泛化验证（能源 + 经济） 双域均有效 → 方法通用性
   4.4 消融实验                3 变体对比表
5. Discussion & Future Work   (0.75 页)  局限性 + 因果扩展
6. Conclusion                 (0.5 页)
```

### 时间线总览

> 按周拆解的甘特图见 [plan.md 附录 B](plan.md#附录-b甘特图总览)。

| 月份 | 里程碑 | 交付物 | 对应 plan.md |
|---|---|---|---|
| **第 1 月** | 合成数据 + 核心模型实现 | `generate.py` + `cmdl_model.py` 跑通，$k^*$ MAE < 1.0 | Step 1–4 |
| **第 2 月** | 三域真实数据实验 + baseline | E2（影子经济）+ E3（能源）+ E4（经济）完成，均优于 3 个 baseline | Step 5–7 |
| **第 3 月** | 消融实验 + 论文初稿 | 3 消融表 + Workshop 论文 8 页 | Step 8–9 |
| **第 4 月（弹性）** | 论文修改 + 投稿 + 申博材料 | 投 KDD MiLeTS 或 AAAI Workshop | Step 10 |

---

## 十、已识别风险与缓解策略

### 10.1 $z_i$ 可识别性风险

**风险**：潜在 AC 得分 $z_i$ 仅通过 proxy 间接学习，可能编码了非 AC 的混淆因素。

**Core 缓解措施**：
1. **代理重构损失** $\mathcal{L}_\text{recon}$：强制 $z_i$ 能反向预测原始 proxy 指标
2. **合成数据验证**：在 ground truth 已知的设置下，验证 $z_i$ 与真实 AC 的秩相关（目标 $\rho_s > 0.8$）
3. **消融实验**：移除 AC 编码器后，观察 $\omega$ 分布退化为同质的程度
4. **论文写作**：明确声明 $z_i$ 是 "learned entity-level moderator score"，避免过度解释

**🔮 Expansion 缓解**：MIG / DCI 解耦评估（需多维 $z_i$，Phase 2）

### 10.2 因果声明合法性

**风险**：论文隐含因果假说，但纯观测数据不支持严格因果推断。

**Core 缓解措施**：
1. 全文使用 **"predictive heterogeneous lag pattern"** 替代 "causal lag effect"
2. 在 Related Work 中显式讨论观测数据的因果局限性
3. 合成数据中因果关系已知，可在该设置下做有限因果声明
4. 在 Discussion 中将因果分析作为 future work 处理

### 10.3 模型复杂度与训练稳定性

**风险**：端到端模型可能训练不稳定。

**Core 缓解措施（已通过架构简化大幅降低此风险）**：
1. **默认 MLP 编码器**（确定性映射），消除 KL 坍缩 / 后验退化风险
2. **损失函数仅两项**（$\mathcal{L}_\text{task} + \lambda_r \mathcal{L}_\text{recon}$），超参数仅 1 个
3. **LSTM backbone 固定**，无 Transformer 的注意力头数/层数调参
4. VAE 变体仅在消融实验中出现，结论预期："VAE 在大样本上略优，小样本无显著差异"

### 10.4 跨域泛化说服力

**风险**：仅一个真实域可能被认为泛化性不足。

**Core 缓解措施**：
1. **三域验证**：影子经济（正规指标 → 影子经济规模）+ 能源（可再生投资 → CO₂）+ 经济（资本深化 → TFP），treatment/outcome 完全不同
2. 三个域的实体集重合但机制不同，若 AC-Gate 均有效则泛化性强于双域验证
3. PWT 经济域的边际实现成本极低（同一个 CSV 文件包含全部所需变量）

### 10.5 与传统方法的公平比较

**风险**：Panel OLS 在简单场景下可能表现不差。

**Core 缓解措施**：
1. 核心评估增加 **$k^*$ 恢复能力** 和 **$\omega$ 异质性方差** 指标——传统方法无此输出
2. 明确 AC-Gate 的核心价值是"可解释的异质滞后发现"，而非单纯预测精度提升
3. 在合成数据上展示传统方法无法恢复实体级 $k^*$

### 10.6 项目范围与执行风险

**风险**：工作量超出申博时间预算。

**Core 缓解措施（已大幅压缩）**：
1. Core 实验矩阵：**5 实验 × 3 baseline × 3 真实域** = 可控规模
2. 代码文件 ~23 个（原 ~20 个）
3. PWT 经济域无需拼表（一个 CSV 即用）
4. 影子经济域同样仅需单 CSV + WGI API
5. 所有 Expansion 内容已标记 🔮，不影响 Phase 1 交付

### 🔮 10.7 理论深度不足（Expansion 风险）

**风险**：当前设计偏"组合创新"，对主会投稿竞争力不足。

**Expansion 缓解**：
1. 补充退化定理：证明 AC-Gate 在 $z_i$ 为常数时退化为 heterogeneous ARDL 特例
2. 补充一致性实验：$N \to \infty$ 时 $k^*$ MAE → 0
3. Phase 1 投稿策略以 Workshop / Applied Track 为主，避开理论要求高的 venue

---

*文档版本：2026-04 v5.0。调整核心 RQ 为正规—非正规经济时滞建模；新增影子经济主验证域；能源+经济域降为泛化验证。Core / Expansion 双层结构。开发步骤与文献阅读节点已迁移至 [plan.md](plan.md)，依赖清单见 [requirements.md](requirements.md)。*
