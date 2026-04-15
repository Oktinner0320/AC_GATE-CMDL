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

---

## 2026-04-15

### config/cmdl_config.py

#### 完成清单
- 为 Step 2 新增 `lag_bias_strength` 配置项，用于控制 Lag Gate 中相对位置偏置的强度。
- 在配置校验逻辑中补充 `lag_bias_strength >= 0` 约束，避免无效超参数进入训练流程。
- 将三个 domain preset 同步更新，保证 synthetic、energy、economics 三类配置都能直接构造 Step 2 所需参数。

#### 质量验证
- 静态错误检查通过，无语法或类型错误。
- `CMDLConfig.from_domain("synthetic")` 可直接返回包含 `temperature` 与 `lag_bias_strength` 的完整 Step 2 配置。

#### 模块依赖
- 被 `model/lag_gate.py` 直接消费，用于 Lag Gate 的 softmax 温度与位置偏置控制。
- 作为 `tests/test_step2_modules.py` 的 synthetic 集成测试配置入口。

### model/ac_encoder.py

#### 完成清单
- 实现 `ACEncoderOutput` 数据结构，统一封装 `z_i` 与 `p_hat_i`。
- 实现 `AdaptiveACEncoder`，结构对齐 Step 2 计划：`Linear(n_proxies, 32) -> LayerNorm -> GELU -> Linear(32, 16) -> GELU -> Linear(16, 1)`。
- 增加 `proxy_reconstructor`，支持从 `z_i` 重构代理变量，供后续 reconstruction loss 使用。
- 补充中英文模块、类、函数与关键步骤注释，说明编码、压缩与重构各阶段职责。

#### 质量验证
- 静态错误检查通过，无语法或类型错误。
- 单元测试中已验证编码器输出 `z_i` 形状为 `[B, 1]`，重构输出 `p_hat_i` 形状与输入 proxy 宽度一致。
- 与 Lag Gate 的集成测试已验证梯度可从下游 loss 回传到编码器参数。

#### 模块依赖
- 被 `tests/test_step2_modules.py` 直接调用，用于 Step 2 单元测试与集成测试。
- 后续将作为 `model/cmdl_model.py` 中 AC 表示学习的入口模块。

### model/lag_gate.py

#### 完成清单
- 清理并保留可复用的 `GatedLinearUnit` 与时序版 `GatedResidualNetwork`，修复早期 `TimeDistributed` 未导入问题。
- 新增 `LagGateOutput` 数据结构，统一封装 `omega`、`k_star`、`lag_context` 与原始 `logits`。
- 实现 `ScalarGatedResidualNetwork`，将 TFT 风格 GRN 骨架适配到 Step 2 的标量 `z_i` 条件化场景。
- 实现 `ScaleInvariantLagGate`，支持相对位置偏置、温度缩放 softmax、`k_star` 计算与 `lag_context` 聚合。
- 补充中英文模块、类、函数与关键步骤注释，明确 time-distributed GRN 与标量 Gate 的职责边界。

#### 质量验证
- 静态错误检查通过，无语法或类型错误。
- 单元测试中已验证 `omega` 按 lag 维归一化到 1，`k_star` 始终落在 `[1, K]`。
- 梯度测试已验证 `loss.backward()` 后 `z_i.grad` 非空且梯度和大于 0，说明门控链路可训练。

#### 模块依赖
- 依赖 `model/backbone.py` 中的 `TimeDistributed` 以保留 Step 3 时序 GRN 复用路径。
- 被 `tests/test_step2_modules.py` 直接使用，作为 Step 2 核心门控模块的验证对象。

### tests/test_step2_modules.py

#### 完成清单
- 新建 Step 2 测试脚本，覆盖 AC Encoder 输出形状、Lag Gate 概率分布、梯度回传与 synthetic 集成链路共 4 项测试。
- 在脚本开头显式将工作区根目录加入 `sys.path`，使测试文件可以从 `tests` 目录直接执行。
- 增加 `KMP_DUPLICATE_LIB_OK` 环境变量设置，绕开 Windows + PyTorch 下重复 OpenMP runtime 的环境性报错。
- 补充中英文模块说明、测试类说明、测试函数说明与关键断言注释，明确每项测试的意图与覆盖范围。

#### 质量验证
- 直接执行文件 `python .\\tests\\test_step2_modules.py` 可通过，输出为 `Ran 4 tests ... OK`。
- 以 unittest 模块路径执行 `python -m unittest tests.test_step2_modules` 可通过，输出为 `Ran 4 tests ... OK`。
- 集成测试已验证编码器和 Lag Gate 在 synthetic batch 上可完成一次前向与反向传播。

#### 模块依赖
- 依赖 `config/cmdl_config.py`、`data/synthetic/generate.py`、`model/ac_encoder.py` 和 `model/lag_gate.py`。
- 当前作为 Step 2 的最小质量门槛，后续可扩展为 Step 3 的 smoke test 基础。

### .vscode/settings.json

#### 完成清单
- 启用 `python.testing.unittestEnabled`，关闭 pytest 自动发现，统一测试框架为 unittest。
- 配置 `python.testing.unittestArgs` 为 `-v -s tests -p test_*.py`，固定测试发现规则。
- 设置 `python.testing.cwd` 为工作区根目录，并开启保存时自动发现测试。

#### 质量验证
- VS Code Testing 面板可按 unittest 规则发现 `tests/test_step2_modules.py`。
- 当前测试文件已支持编辑器内直接运行、终端直接运行和 unittest 模块方式运行三种入口。

#### 模块依赖
- 服务于所有后续 Python 测试文件，不局限于 Step 2。
- 与 `tests/test_step2_modules.py` 的路径修正逻辑配合，提升“直接点击运行测试”的稳定性。
