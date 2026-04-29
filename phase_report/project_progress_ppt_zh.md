# 中文版项目进度 PPT（5页）

## PPT 生成提示词（不计入页数）

请根据下列“第1页”到“第5页”生成一份 16:9 中文研究进度 PPT。风格要求：学术汇报、简洁、留白充足，每页只保留 2-4 个要点，不扩写成长段落。

需要插入的已有图表如下：

- 第2页插入机制流程图：`workflow_overview.png`
- 第4页插入合成实验分布图：`synthetic_kstar_mae_seed_distribution.png`
- 第5页并排插入真实数据机制图：`economics_structured_mechanism_seed_distribution.png` 与 `energy_structured_mechanism_seed_distribution.png`

可参考的数据表：

- `synthetic_main_table.csv`
- `realdata_forecast_table.csv`
- `economics_stratified_main_table.csv`
- `energy_stratified_main_table.csv`
- `verdict_matrix.csv`

---

## 第1页：机制目的与用途

**AC-GATE / CMDL：实体条件异质滞后发现机制**

- 目的：发现不同国家/实体对同一时间序列信号的响应滞后差异
- 核心输出：每个实体的有效滞后期 `k*` 和滞后权重分布 `omega`
- 可以用于：经济、能源、治理等面板时序中的滞后审计与机制比较
- 定位：机制发现优先，预测结果作为校准与边界检验

---

## 第2页：当前模型机制已闭合

![AC-GATE workflow](workflow_overview.png)

- 代理变量 `proxy` 被编码为吸收能力表示 `z_i`
- `z_i` 条件化生成实体专属滞后分布 `omega(k | z_i)`
- 滞后分布汇总历史输入，得到可解释的 `k*`
- 已完成模型、数据管线、基线、消融和 20-seed 输出

---

## 第3页：实验体系与当前进度

**三类数据，统一协议，20 个随机种子**

- Synthetic：有真实 `k*`，用于验证机制恢复能力
- Economics：PWT 11.0，目标为 `ctfp`，用于真实经济面板审计
- Energy：OWID-energy × WGI，目标为 `co2_per_unit_energy`
- 对照方法：Plain LSTM、Grouped ARDL、No AC Encoder、Uniform Lag、No Recon Reg

---

## 第4页：合成实验结果

![Synthetic kstar distribution](synthetic_kstar_mae_seed_distribution.png)

- Linear：CMDL `k*` Spearman rho = 0.945，`k*` MAE = 1.159
- Nonlinear：CMDL `k*` Spearman rho = 0.907，`k*` MAE = 1.467
- Plain LSTM 和退化消融在 `k*` 恢复上明显更弱
- 结论：AC encoder + lag gate 是机制恢复的关键结构

---

## 第5页：真实数据实验情况

![Economics mechanism](economics_structured_mechanism_seed_distribution.png)
![Energy mechanism](energy_structured_mechanism_seed_distribution.png)

- Economics：`k*` 与人力资本等发展指标显著结构化，Fisher p < 1e-24
- Energy：`k*` 与治理指标结构更强，Fisher p < 1e-76
- 预测层不主张最优：Economics 与 Energy 均有更强预测基线
- 当前结论：真实域支持“结构化异质滞后”，但方向性机制仍需谨慎表述