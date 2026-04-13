# Worklog

## 2026-04-13

### config/cmdl_config.py

#### 完成清单
- 实现 CMDLConfig 数据类，统一管理 Core 阶段的模型超参数和合成数据参数。
- 实现 synthetic、energy、economics 三个 domain preset，并支持 overrides 覆盖默认值。
- 增加基础参数校验，覆盖 domain、scenario、lag 长度、特征维度、噪声强度等约束。
- 增补中英文模块、类、方法注释，明确配置字段含义与 preset 用途。

#### 质量验证
- 静态错误检查通过，无语法或类型错误。
- 最小生成验证中成功通过 CMDLConfig.from_domain("synthetic") 构造默认配置。
- synthetic 默认配置与计划保持一致：max_lag=10，d_model=64，lambda_r=0.1，n_entities=200，seq_length=30。

#### 模块依赖
- 作为后续 model、data、experiments 模块的统一参数入口。
- 被 data/synthetic/generate.py 直接依赖，用于控制场景、实体规模、噪声与维度设置。

### data/synthetic/generate.py

#### 完成清单
- 实现 SyntheticPanel 数据容器，统一封装 X_it、p_i、s_i、Y_it、z_true、kstar_true、entity_ids、time_index 与 metadata。
- 实现 linear 与 nonlinear 两种 ground truth 的 kstar 生成逻辑。
- 实现 proxy、static feature、AR 风格输入序列、lag 权重和目标序列生成逻辑。
- 增加 summarize_synthetic_data 和 plot_z_vs_kstar 两个辅助接口，支持快速检查与可视化。
- 增补中英文模块、类、函数注释，并在 lag 权重归一化与 target 合成处补充关键中文说明。

#### 质量验证
- 静态错误检查通过，无语法或类型错误。
- 已完成最小生成验证：默认 synthetic 配置可成功生成形状为 (200, 30, 1) 的 X_it、(200, 3) 的 p_i、(200, 2) 的 s_i 和 (200, 30) 的 Y_it。
- 线性场景下 kstar 范围为 3 到 10；非线性场景下 kstar 范围为 1 到 10。
- summarize_synthetic_data 可返回 z 与 kstar 的范围统计，满足 Step 1 的基础 sanity check 需求。

#### 模块依赖
- 作为 Step 1 合成数据入口，供后续 experiments/run_synthetic.py 和 notebooks 验证流程复用。
- 返回结构已对齐后续 LSTM 的 batch_first 布局，减少 Step 2 和 Step 3 的接口返工。
