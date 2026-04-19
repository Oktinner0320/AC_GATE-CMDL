#import "@preview/charged-ieee:0.1.4": ieee

#show: ieee.with(
  title: [CMDL/AC-GATE 阶段报告：机制验证进展、已有结果与当前瓶颈],
)

#set text(
  lang: "zh",
  font: ("Georgia", "SimSun", "STSong"),
)
#set par(justify: true)
#show heading: set text(font: ("Georgia", "SimSun", "STSong"))

= 摘要

本项目关注面板时序中的异质滞后建模问题。核心目标不是只做一个更强的黑盒预测器，而是在实体间制度、能力或治理条件存在差异时，显式学习不同实体对应的滞后分布，并据此解释“为什么同样的输入冲击会在不同主体上表现出不同的传导速度”。围绕这一目标，我提出了 AC-GATE 机制，并已完成 synthetic、economics、energy-co2 三个数据域上的 baseline 对比与核心消融。当前阶段的主要收获是：机制本身已经在合成数据上被较强地验证；但在真实域上，这种结构优势尚未稳定转化为显著优于 plain LSTM 的预测性能，且机制解释证据仍有不稳定之处。下面主要汇报项目背景、提出的机制、已有成绩，以及目前最困扰我的问题。

= 当前项目

本项目以条件调节型分布式滞后预测为核心任务，英文上可理解为 conditionally modulated distributed lag prediction，内部统一记为 CMDL。研究对象是实体 $i$ 与时间 $t$ 共同构成的面板数据：输入序列 $X_(i,t)$ 会通过若干滞后期影响输出 $Y_(i,t)$，但不同实体的有效滞后长度并不相同，而是受其制度环境、组织能力或结构条件影响。传统 panel-LSTM、ARDL 或 TFT 可以拟合动态关系，但通常不直接给出“实体级异质滞后分布”这一中间结构，因此很难把预测结果转化为清晰的滞后解释。

基于这一判断，我当前把项目分成三个验证层次。第一层是 synthetic 合成数据，用来验证机制是否真的学到了设定好的异质滞后规律；第二层是 economics 域，用资本深化到 TFP 的传导关系做真实数据检验；第三层是 energy-co2 域，用可再生能源占比到单位能源 CO2 强度的关系做跨域泛化。到目前为止，这三条实验链都已经跑通了 full CMDL、plain LSTM baseline 和关键 ablation，因此当前问题已经不再是“工程上是否能跑起来”，而是“现有结果是否足以支撑一条清晰、可信的研究叙事”。

= 我提出的机制

AC-GATE 的直觉很简单：如果不同实体的滞后模式确实受某类慢变量调节，那么模型就不应为所有实体共享同一套固定 lag 权重，而应当先抽取实体级条件表示，再用这个条件表示去控制 lag 分布。具体而言，我把机制拆成三步。

第一步，用 AC encoder 从实体的 proxy 集合 $p_i$ 中编码出一个低维条件表示 $z_i$；第二步，用 $z_i$ 生成条件滞后分布 $omega_i(k)$，表示第 $k$ 个历史时刻对当前预测的权重；第三步，再将加权后的历史输入与静态特征一起送入下游预测头，从而得到最终输出。若写成更完整的形式，AC-GATE 至少包含下面三层量：
\
\
\

$
H_(i,t) = sum_(k=1)^K omega_i(k) X_(i,t-k), quad
omega_i(k) >= 0, quad sum_(k=1)^K omega_i(k) = 1,
$

$
hat(Y)_(i,t) = f(H_(i,t), s_i), quad
k_i^* = sum_(k=1)^K k omega_i(k)
$

其中各符号含义如下：\
- $i$ 表示实体索引，在本项目中通常对应一个国家或地区。\
- $t$ 表示当前预测时刻。\
- $k$ 表示滞后阶数，即向前追溯的历史步长。\
- $K$ 表示预先设定的最大滞后窗口长度。\
- $p_i$ 表示实体 $i$ 的 proxy 向量，用于刻画治理、能力或结构条件。\
- $z_i$ 表示由 AC encoder 提取出的实体级条件表示。\
- $omega_i(k)$ 表示实体 $i$ 在第 $k$ 个滞后位置上的权重，决定该历史时刻对预测的贡献大小。\
- $X_(i,t-k)$ 表示实体 $i$ 在时刻 $t-k$ 的输入观测值。\
- $H_(i,t)$ 表示对历史输入按条件滞后分布加权后得到的聚合表示。\
- $s_i$ 表示实体 $i$ 的静态特征或固定背景信息。\
- $f(·)$ 表示下游预测映射，可以理解为由主干网络实现的回归函数。\
- $hat(Y)_(i,t)$ 表示模型对时刻 $t$ 输出变量的预测值。\
- $Y_(i,t)$ 表示实体 $i$ 在时刻 $t$ 的真实观测输出，用于监督训练和最终评估。\
- $k_i^*$ 表示实体 $i$ 的期望最优滞后，用于概括其平均传导延迟。\

从解释角度看，$H_(i,t)$ 负责把历史信息压缩成一个条件加权后的有效输入，$omega_i(k)$ 负责回答模型更关注哪几个滞后期，而 $k_i^*$ 则把整条滞后分布进一步压缩成一个可直接比较的标量指标。

与 plain LSTM 相比，这个机制的关键区别不在于“更深”或“参数更多”，而在于它把 lag learning 单独显式化了。plain LSTM 只能在隐状态里混合历史信息，后续即使做 post-hoc lag occlusion，也只是事后解释；而 AC-GATE 直接把异质滞后分布作为模型中间产物输出，这使得后续的恢复实验、消融实验和跨域解释都具有更明确的目标。

= 已有成绩

目前最实在的成绩首先来自 synthetic。合成数据上的三条 formal target 验收链已经全部通过：在线性恢复任务中，$k^*$ 的 MAE 已下降到 0.9229，Spearman 相关达到 0.9805；在 latent identification 任务中，proxy 重构 $R^2$ 达到 0.9439，$z$ 与真实条件变量的秩相关达到 0.9892；在非线性恢复任务中，$k^*$ 的 MAE 进一步下降到 0.5348，相关保持在 0.9622。这说明 AC-GATE 并不只是“能跑”，而是已经在有 ground truth 的环境下较强地学到了目标结构。

如果与 plain LSTM 的 post-hoc lag occlusion 结果直接对比，这个优势会更直观：linear 场景下，baseline 的 $k^*$ MAE 大约在 1.60 到 1.67 之间，而 CMDL 已降到 0.92；nonlinear 场景下，baseline 的 $k^*$ MAE 大约在 2.51 到 2.68 之间，而 CMDL 已降到 0.53。对应的秩相关也从 baseline 的约 0.4 左右提升到 0.96 以上。换句话说，synthetic 上的结论已经不只是“机制存在”，而是“引入 AC-GATE 之后，结构恢复能力出现了明显的量级提升”。

#figure(
  image("img/synthetic_recovery.png", width: 100%),
  caption: [Synthetic 合成数据的恢复对比。CMDL 在两类场景下都明显优于 plain LSTM，说明异质滞后机制本身已经被验证。],
)

第二个已经比较明确的成绩，是 Step 4.5 的 baseline 与 ablation 证据已经闭合。当前结果表明，no_ac_encoder 与 uniform_lag 会系统性破坏 $k^*$ 的排序恢复，而 no_recon_regularization 与 full CMDL 基本重合。这一点对我很重要，因为它说明 synthetic 上真正带来提升的，是 AC conditioning 与 adaptive lag gate 本身，而不是某个附加技巧。换句话说，我提出的核心机制不是装饰性的，而是必要模块。

第三个成绩是，economics 和 energy 两个真实域的 full pipeline 已经打通，并完成了与 plain LSTM 的可比实验。虽然结果还不够漂亮，但并非毫无增益。例如在 economics 的 formal target 中，CMDL 的平均 test $R^2$ 为 -0.079，而 plain LSTM 为 -0.094；平均 MAE 为 1.604，而 plain LSTM 为 1.619。energy-co2 域也有类似现象：CMDL 的平均 test $R^2$ 为 -0.028，而 plain LSTM 为 -0.033；平均 MAE 为 16.83，而 plain LSTM 为 16.98。也就是说，真实域里已经能看到局部 forecast 收益，只是这种收益目前还比较弱，不足以直接支撑强结论。

= 当前遇到的问题

我目前最困扰的并不是 synthetic，因为 synthetic 反而是最清楚的一部分。真正的困难来自真实域。先看 economics：formal target 下，CMDL 的平均 test $R^2$ 虽然略优于 plain LSTM，但 lag-proxy 相关的均值大约是 -0.926，而 plain LSTM 约为 0.062；同时 CMDL 的 effective $k^*$ 标准差只有 0.110 左右，而 plain LSTM 的 post-hoc spread 约为 1.097。这个结果非常矛盾，因为它意味着模型在 forecast 上并非完全失效，但在机制评价上却表现出明显的方向错位或异质性展开不足。

更进一步地看 richer feature bundle，问题并没有自动消失。在 growth-aware 套件中，CMDL 的平均 test $R^2$ 约为 0.021，反而低于 plain LSTM 的 0.057；其 lag-proxy rho 的均值约为 -0.376，而且 3 个 seeds 中只有 1 个是正号。在 effective-labor-aware 套件中，CMDL 的平均 test $R^2$ 提升到 0.063，但仍低于 plain LSTM 的 0.099；同时其机制相关指标的 dominant sign 仍然是 negative。也就是说，即便换了更贴近经济含义的 proxy 设计，当前也只能改善 forecast，不能稳定修复 mechanism sign。

#figure(
  image("img/economics_diagnostics.png", width: 100%),
  caption: [Economics 域的 lag-proxy diagnostics。当前瓶颈不是机制完全失效，而是机制方向与评价 anchor 之间仍存在明显张力。],
)

energy-co2 域的问题与 economics 有相似之处，但证据更弱。当前 CMDL 对 plain LSTM 只有边际预测优势，而 lag-proxy rho 的均值大约为 -0.235，标准差却接近 0.972，说明不同 seeds 之间的符号和强度都很不稳定；同时 CMDL 的 effective $k^*$ 标准差只有 0.106 左右，而 plain LSTM 的 post-hoc spread 约为 1.357。我的直观感受是：energy 目前更像一个“可运行的跨域泛化样本”，而不是一个已经足以支撑方法论主张的强验证域。如果在这个阶段继续在 energy 上堆复杂度，收益可能并不高。

除此之外，我还越来越意识到一个更根本的问题：真实域里的 diagnostics 可能和当前训练目标并没有完全对齐。以 economics 为例，训练时往往同时重构多个 proxies，但最后在 notebook 里判断机制是否成立时，又经常只盯住一个 anchor proxy；再加上当前 proxy reconstruction 分支对 $z_i$ 的使用是 detached 的，很多真实域指标其实更像是“冻结后的 latent 可读性诊断”，而不是严格意义上的端到端机制学习效果。这使得我现在不太愿意把负的 lag-proxy rho 直接解释为“AC-GATE 机制失败”，更可能的解释是 objective 与 evaluation anchor 之间存在错位。

基于目前的结果，我更倾向于把当前研究口径写成这样：AC-GATE 已经在 synthetic 上强力证明了异质滞后机制是可学习、可解释且优于 plain LSTM 的；economics 与 energy 则展示了这套机制在真实数据上的潜力，但真实域中的结构收益尚未稳定兑现为显著的 forecast 优势。也正因为如此，我下一步更想优先做 matched-init 对照、clip control 和 anchor-aligned diagnostics，而不是马上继续堆更复杂的模型。对我来说，当前最想向老师请教的，其实不是“这个机制还要不要继续做”，而是“在 synthetic 已经证明机制有效的前提下，是否应该把真实域暂时定位为 generalization / feasibility evidence，而不是强机制验证域”。