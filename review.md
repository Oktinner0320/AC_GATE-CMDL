# 实验公平性 / 可复现性审查报告

> 依据：[reproducibility_requirements.md](reproducibility_requirements.md) 第三节"实验设计层面的公平性要求"，以及与之耦合的 §2（代码层）、§4（NeurIPS Checklist）、§5（GenAI / 匿名化）。
> 范围：当前仓库 `experiments/`、`baselines/`、`data/`、`evaluation/`、`config/`、根目录元文件。
> 评级：✅ 合规 / ⚠️ 部分合规需补 / ❌ 不合规需修。
> 行动原则：所有代码改动给出**最小补丁方案**，不重写既有模块。

---

## 0. 总览

| 维度 | 评级 | 关键依据 |
|---|---|---|
| 3.1 基线公平对比 | ✅ + ⚠️ | 同 loader / 同 seeds / 同骨干超参；缺超参搜索声明 |
| 3.2 数据处理透明性 | ✅ + ⚠️ | 训练窗口统计已严格隔离；缺数据快照元数据 |
| 3.3 评估协议严谨性 | ✅ + ❌ | 20 seeds 均值 + std 已有；**缺配对显著性检验** |
| §2.1 代码工程包装 | ❌ | 缺 `requirements.txt` / `LICENSE` / 复现命令段 |
| §4 NeurIPS Checklist | ⚠️ | 缺硬件 / 训练时长 / 确定性算法标记 |
| §5 GenAI / 匿名化 | ❌ | 缺 `GenAI Usage Disclosure`，未做匿名化清理 |

风险排序（影响录用红线）：**§3.3 显著性检验 > §5 GenAI > §2.1 依赖锁定 > §3.2 数据元数据 > 其余**。

---

## 1. 3.1 基线（Baseline）公平对比

### 1.1 ✅ 相同实验条件

- 入口 [experiments/run_complete_20seed_suite.py](experiments/run_complete_20seed_suite.py) 通过 `economics_common_args` / `energy_common_args` / `synthetic_common_args` 给所有方法（CMDL / Plain LSTM / Grouped ARDL / Ablations）传同一份数据路径、`train_end_year`、`val_end_year`、`feature_bundle`、`max_missing_share`、`epochs`、`patience`、20 seeds（`SEEDS = list(range(20))`）。
- 随机种子在每次 run 入口处由 [run_economics.py#L172-L182](experiments/run_economics.py#L172-L182) `set_seed()` 统一固定 Python / NumPy / PyTorch (CPU+CUDA)。

**无需改动。**

### 1.2 ⚠️ 超参搜索一致性

**问题**：仓库内**没有任何超参搜索脚本**（grid / random / Optuna 均未出现）。所有方法用 `*_common_args` 中的固定默认值；CMDL 与基线骨干超参（`d_model=64`, `lstm_layers=2`, `dropout=0.05`, `lr=1e-3`, `epochs`, `patience`, `grad_clip=1.0`）已经一致。但论文若不显式声明"未做超参调优"，审稿人无法判别公平性。

**最小修改方案**：
1. 在 [readme.md](readme.md) 新增一节 `## Hyperparameter Protocol`，明确：
   - 所有方法共享 `experiments/run_complete_20seed_suite.py` 中的 `*_common_args`，未做超参搜索；
   - CMDL 特有项（`lambda_r`, `temperature`, `omega_*`, `lag_bias_strength`）来自合成域 Step 4 的固定默认；
   - 基线（Plain LSTM, Grouped ARDL）骨干超参直接继承 `CMDLConfig`。
2. 论文 Implementation Details 节复制此段。

**不改代码**。

### 1.3 ⚠️ 自实现基线 / 官方实现声明

**问题**：
- [baselines/lstm_baseline.py](baselines/lstm_baseline.py)、[baselines/panel_ols.py](baselines/panel_ols.py) 均为自实现；
- [baselines/tft_baseline.py](baselines/tft_baseline.py) 存在但**未被 [run_complete_20seed_suite.py](experiments/run_complete_20seed_suite.py) 调用**，[readme.md](readme.md) 仍把它当作活跃基线。

**最小修改方案**：
1. 在 [readme.md](readme.md) 基线章节增加表格，列出每个基线的：实现方式（self-implemented / official）、参考文献、是否参与 20-seed 套件。
2. 把 TFT 标注为 "implemented but not included in the 20-seed evaluation; reserved for future work" 或从 readme 基线清单中移除，避免审稿人质疑选择性比较。
3. 论文 Baselines 节同步声明。

**代码无需改动**（`tft_baseline.py` 保留作为后续扩展即可）。

### 1.4 ✅ 公平的数据划分

- [data/economics/economics_loader.py](data/economics/economics_loader.py) `build_temporal_splits` 与 [data/energy/energy_loader.py](data/energy/energy_loader.py) 同形式，对所有方法返回同一 train/val/test；
- 测试集仅在 `evaluate()` 入口被消费，超参未基于 test 选择。

**无需改动。**

---

## 2. 3.2 数据处理透明性

### 2.1 ✅ 预处理步骤完整记录

- Economics: [data/economics/prepare.py](data/economics/prepare.py) 导出 `economics_cleaned_long.csv`，含 `row_was_missing` 等插值标志；
- Energy: [data/energy/download.py](data/energy/download.py) 输出标准化合并表 `energy_wgi_merged.csv`；
- 所有方法共享同一 loader，预处理一致。

**无需改动。**

### 2.2 ✅ 数据泄露检查

- [economics_loader.py#L511-L580](data/economics/economics_loader.py#L511-L580)、energy_loader 同模式：`stats_end_year` 强制 per-entity X/Y 标准化、proxy/static 聚合、cross-entity z-score 仅用训练窗口；
- [run_economics.py](experiments/run_economics.py) 与 [run_economics_lstm_baseline.py](experiments/run_economics_lstm_baseline.py) 都以 `train_end_year` 触发。

**无需改动。**

### 2.3 ⚠️ 数据集版本锁定

**问题**：原始数据通过下载脚本拉取后缓存为 `data/economics/raw/pwt110.csv`、`data/energy/raw/energy_wgi_merged.csv`，但**未记录下载日期、源 URL 或 sha256**。Zenodo 归档与审稿复现都需要稳定版本号。

**最小修改方案**（每个 download 脚本仅加约 15 行）：

在 [data/economics/download.py](data/economics/download.py) 与 [data/energy/download.py](data/energy/download.py) 下载完成后追加：

```python
import hashlib, json, datetime
def _write_meta(cache_path: Path, source_url: str) -> None:
    digest = hashlib.sha256(cache_path.read_bytes()).hexdigest()
    meta = {
        "source_url": source_url,
        "downloaded_at_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "sha256": digest,
        "bytes": cache_path.stat().st_size,
    }
    cache_path.with_suffix(cache_path.suffix + ".meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
```

并在原下载落盘后调用 `_write_meta(cache_path, SOURCE_URL)`。生成 `pwt110.csv.meta.json` 与 `energy_wgi_merged.csv.meta.json`，提交进 git。

### 2.4 ✅ 专有数据

全部为公开数据（PWT 11.0 Dataverse / OWID-energy / WGI），自动满足 ECML PKDD "至少部分公开"。

---

## 3. 3.3 评估协议严谨性

### 3.1 ✅ 多次独立运行

- 20 seeds × 全部方法 × 三域，已聚合 mean / std / positive seed share，见 [experiment_results_20seed.md](experiment_results_20seed.md) 与 `outputs/notebook_*/complete_20seed/comparison/*.csv`。

**无需改动。**

### 3.2 ❌ 配对显著性检验（关键缺失）

**问题**：全仓 grep 无 `wilcoxon` / `ttest_rel` / `bootstrap` 命中。当前 comparison CSV 只输出 `mean ± std`，未做配对检验。Economics/Energy 的 test_R² 差距非常小（~0.05），不附显著性论文将无法主张差异有意义——这是 §3.3 明文要求。

**最小修改方案**（仅在 evaluation 层加一个新函数，调用方按需启用，不破坏既有汇总）：

新增 `evaluation/significance.py`：

```python
"""Paired significance tests across seeds for compact comparison tables."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

def paired_wilcoxon(per_seed: pd.DataFrame, metric: str,
                    method_col: str = "model", seed_col: str = "seed",
                    reference: str = "CMDL") -> pd.DataFrame:
    """Return Wilcoxon signed-rank p-values comparing each method to `reference`
    on `metric`, paired by seed. Drops seeds missing either side.
    """
    pivot = per_seed.pivot(index=seed_col, columns=method_col, values=metric)
    if reference not in pivot.columns:
        raise ValueError(f"reference {reference!r} not in {list(pivot.columns)}")
    rows = []
    ref = pivot[reference]
    for method in pivot.columns:
        if method == reference:
            continue
        joined = pd.concat([ref, pivot[method]], axis=1).dropna()
        if len(joined) < 5:
            rows.append({"method": method, "n_pairs": len(joined),
                         "median_diff": np.nan, "wilcoxon_p": np.nan})
            continue
        diff = joined[reference] - joined[method]
        try:
            stat = wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
            p = float(stat.pvalue)
        except ValueError:
            p = np.nan
        rows.append({"method": method, "n_pairs": int(len(joined)),
                     "median_diff": float(np.median(diff)), "wilcoxon_p": p})
    return pd.DataFrame(rows)
```

并在三域 comparison 脚本（[evaluation/economics_comparison.py](evaluation/economics_comparison.py)、[evaluation/energy_comparison.py](evaluation/energy_comparison.py)、[evaluation/synthetic_comparison.py](evaluation/synthetic_comparison.py)）的"按 seed 长表"导出位置增加一行：

```python
from evaluation.significance import paired_wilcoxon
sig_table = paired_wilcoxon(per_seed_long, metric="test_r2", reference="CMDL")
sig_table.to_csv(out_dir / "significance_test_r2.csv", index=False)
```

对关键指标各跑一份：`test_r2`（任务侧）、`anchor_adjusted_rho`（机制侧）、`kstar_mae`（synthetic）。

**一次新增文件 + 每个 comparison 模块加 3 行调用**，不动既有数值。

### 3.3 ✅ 评估指标对齐

- 任务侧 MSE/MAE/R²；
- 机制侧 anchor-adjusted Spearman ρ + p-value、k* MAE/std、Ω 熵、proxy R²、lag sensitivity；
- 由 [evaluation/realdata_diagnostics.py](evaluation/realdata_diagnostics.py) 统一输出，对所有方法（含 ablation）一致。

**无需改动。**

### 3.4 ✅ Ablation Study 完整性

- 三个变体（`no_ac_encoder` / `uniform_lag` / `no_recon_regularization`）× 20 seeds × 三域，结果已落盘。

**无需改动。**

### 3.5 ⚠️ Ablation 的"匹配初始化"对照

**问题**：[run_economics_ablation.py#L165-L181](experiments/run_economics_ablation.py#L165-L181) 已显式区分：
- `no_recon_regularization`：复用 `setup.model` → `matched_init_to_full_cmdl=True`；
- `no_ac_encoder` / `uniform_lag`：模型结构改变后**未重新设置 seed**，与 full CMDL 不可视为匹配初始化对照。
- 这一情况已写入每个 run 的 `summary.json`（`causal_ablation_validity` 字段），属于"已诚实披露的局限"。

**最小修改方案**（任选其一）：

- **方案 A（不改代码）**：在论文 Ablation 节引用 summary 字段，明说"架构差异变体不是匹配初始化对照，仅作机制必要性证据"；
- **方案 B（小改）**：在 [run_economics_ablation.py](experiments/run_economics_ablation.py) 与 [run_energy_ablation.py](experiments/run_energy_ablation.py) `prepare_variant_setup` 构造完 `model` 后立即调用一次 `set_seed(seed)`，使变体模型在与 full CMDL 同种子下被 PyTorch 初始化（仅几行）。这能把 `causal_ablation_validity` 提升为更强的对照。

推荐 **方案 B**（最小改：每个 ablation runner 加 1 行 `set_seed(int(seed))` 在 `model = ...` 之前）。

---

## 4. §2.1 代码工程包装

### 4.1 ❌ `requirements.txt` 版本锁定

**问题**：仅有 [requirements.md](requirements.md) 文档，使用 `≥X.Y` 软约束；无 `requirements.txt` / `environment.yml`。

**最小修改方案**：在 PTenv 中执行：

```powershell
conda activate PTenv
pip freeze | Select-String -NotMatch "^(-e |# )" > requirements.txt
```

提交 `requirements.txt`。可选：再导出 `environment.yml`：

```powershell
conda env export --no-builds | Out-File -Encoding utf8 environment.yml
```

不动任何源代码。

### 4.2 ❌ `LICENSE`

**问题**：根目录无 LICENSE 文件。

**最小修改方案**：新增一个 `LICENSE`，建议 MIT 或 Apache-2.0（与依赖库主流许可兼容）。无代码改动。

### 4.3 ⚠️ README 复现指引

**问题**：[readme.md](readme.md) 列出实验脚本路径，但**没有"如何复现论文表/图"的具体命令段**；20-seed 套件入口未在 README 显式标注。

**最小修改方案**：在 [readme.md](readme.md) 末尾新增一节 `## Reproducing paper tables and figures`：

```markdown
## Reproducing paper tables and figures

```bash
# 1. 完整复现 20-seed 实验套件（会跳过已存在的 summary.json）
python experiments/run_complete_20seed_suite.py --domain all

# 2. 或按域分片
python experiments/run_complete_20seed_suite.py --domain synthetic --seeds 0 1 2 3 4 5 6 7 8 9
python experiments/run_complete_20seed_suite.py --domain synthetic --seeds 10 11 12 13 14 15 16 17 18 19

# 3. 重新生成 comparison CSV / 图
jupyter nbconvert --to notebook --execute notebooks/01_synthetic_verify.ipynb
jupyter nbconvert --to notebook --execute notebooks/02_economics_results.ipynb
jupyter nbconvert --to notebook --execute notebooks/03_energy_results.ipynb
```

| Paper artifact | Source file |
|---|---|
| Table: Synthetic main result | `outputs/notebook_synthetic/complete_20seed/comparison/synthetic_multiseed_summary.csv` |
| Table: Economics main result | `outputs/notebook_economics/complete_20seed/comparison/economics_compact_summary.csv` |
| Table: Energy main result | `outputs/notebook_energy/complete_20seed/comparison/energy_compact_summary.csv` |
| Figure: ω heatmap | `visualization/omega_heatmap.py` |
| Figure: k* distribution | `visualization/kstar_distribution.py` |
```

无代码改动。

---

## 5. §4 NeurIPS Reproducibility Checklist 缺口

### 5.1 ⚠️ 硬件 / 训练时长记录

**问题**：每次 run 写出 `args.json` + `summary.json`，但**未记录 GPU 型号、CUDA 版本、训练时长**。

**最小修改方案**：在 [run_economics.py](experiments/run_economics.py) `setup_experiment` 末尾或 `summarize_run` 入口处，把以下字段并入 summary：

```python
import time, platform
runtime_meta = {
    "device_name": (torch.cuda.get_device_name(setup.device)
                    if setup.device.type == "cuda" else platform.processor()),
    "cuda_version": torch.version.cuda,
    "torch_version": torch.__version__,
    "python_version": platform.python_version(),
    "os": platform.platform(),
    "wall_time_seconds": float(time.perf_counter() - t0),  # t0 在训练循环前记录
}
summary["runtime"] = runtime_meta
```

energy / lstm / ablation runner 同步加（可抽到 `experiments/_runtime_meta.py` 一个单文件 helper，3 处调用）。

### 5.2 ⚠️ 确定性算法（可选）

**问题**：`set_seed` 未设 cuDNN / `torch.use_deterministic_algorithms`。GPU 上 LSTM 默认非完全确定。

**最小修改方案**（在 `set_seed` 末尾增加 2 行，仅当用户设置了环境变量时才启用，避免影响速度）：

```python
if os.environ.get("CMDL_DETERMINISTIC", "0") == "1":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

文档化在 README 复现节提示 `CMDL_DETERMINISTIC=1`。

---

## 6. §5 GenAI / 匿名化

### 6.1 ❌ GenAI Usage Disclosure（CIKM 强制）

**问题**：仓库与论文均无该章节。

**最小修改方案**：新增根目录文件 `GenAI_Usage_Disclosure.md`，包含：
- 工具与版本（如 GitHub Copilot Chat / Claude / GPT-4 等实际使用工具）；
- 用途分布：代码生成 / 代码审查 / 文档润色 / 实验结果分析；
- 未用于生成实验数据、未代写实验结论；
- 所有 GenAI 输出均经作者人工审查。

论文同步插入独立章节（CIKM 不计入页数）。

### 6.2 ⚠️ 投稿期匿名化（ICDM 三盲）

**问题**：未审计 `git log` / commit message / 注释中是否含个人信息。当前路径已无个人化绝对路径（仅工作区相对路径）。

**最小修改方案**（仅在投稿前执行，不立刻改仓库）：
1. 创建匿名分支：`git checkout --orphan anonymous && git commit -m "Initial anonymous snapshot"`；
2. 上传至 [Anonymous GitHub](https://anonymous.4open.science/)；
3. Checklist：
   - [ ] README 中无作者姓名、邮箱、机构；
   - [ ] 代码文件头无个人信息；
   - [ ] `LICENSE` 与 `GenAI_Usage_Disclosure.md` 中作者字段匿名；
   - [ ] notebook metadata（`kernelspec`、用户名）已清理；
   - [ ] outputs/checkpoints 中无 mlflow 用户名（mlflow 默认会记录 OS user）。

---

## 7. 行动清单（按风险高 → 低）

| # | 行动 | 类型 | 文件 | 是否改代码 |
|---|---|---|---|---|
| 1 | 加配对 Wilcoxon 显著性检验 | 新增 | `evaluation/significance.py` + 三个 comparison 模块各 3 行 | ✅ 是（最小） |
| 2 | 写 `GenAI_Usage_Disclosure.md` | 新增 | 根目录 | ❌ |
| 3 | 导出 `requirements.txt` (+ `environment.yml`) | 命令 | 根目录 | ❌ |
| 4 | 加 `LICENSE` | 新增 | 根目录 | ❌ |
| 5 | 数据快照 `*.meta.json`（sha256 + URL + 日期） | 新增 + 小补丁 | `data/economics/download.py`, `data/energy/download.py` | ✅ 是（每文件 ~15 行） |
| 6 | summary.json 加 runtime meta（GPU/CUDA/wall time） | 小补丁 | `experiments/_runtime_meta.py`（新）+ 各 runner 1-2 行 | ✅ 是（最小） |
| 7 | Ablation runner 在改造模型后重设 seed | 小补丁 | `experiments/run_economics_ablation.py`, `experiments/run_energy_ablation.py` 各 1 行 | ✅ 是（1 行） |
| 8 | README 增 `Hyperparameter Protocol` + `Reproducing paper tables` 段 | 文档 | `readme.md` | ❌ |
| 9 | README 基线表，TFT 标注为未纳入 | 文档 | `readme.md` | ❌ |
| 10 | `set_seed` 加可选 `CMDL_DETERMINISTIC` 分支 | 小补丁 | `experiments/run_economics.py`（共享函数） | ✅ 是（2 行） |
| 11 | 投稿前匿名化分支 + Anonymous GitHub 上传 | 流程 | — | ❌ |

红线项（不做会被拒）：**1, 2**。
强烈建议补的工程项：**3, 4, 5, 6**。
论文层面声明项（零代码）：**8, 9, 11**。
