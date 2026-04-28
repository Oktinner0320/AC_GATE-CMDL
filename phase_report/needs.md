我正在写一篇综述，用于定位研究空白。
    RQ: How can temporal delays between formal and informal economic processes be modeled?

我搜索的关键词为：
    TITLE-ABS-KEY ( "shadow economy" OR "informal economy" OR "underground economy" OR informality OR "informal sector" )
    AND TITLE-ABS-KEY ( "time lag" OR "temporal delay" OR "lagged effects" OR "delayed response" OR "temporal dynamics" OR "lead-lag relationship" OR "lag structure" OR "distributed lag" OR "Granger causality" )
    AND TITLE-ABS-KEY ( "autoregression" OR VAR OR ARDL OR "dynamic panel model" OR "time series forecasting" OR LSTM OR RNN OR "machine learning time series" OR "deep learning temporal modeling" )

我已完成方法论部分（如何文件检索和筛选文献），现在，我要进行results部分的撰写。以下为results部分的撰写建议：
    After identifying a manageable set of 15--20 relevant studies, you should focus on comparing them, identifying common patterns, key differences, and how they address the research question. The emphasis should be on analytical synthesis (e.g., themes, mechanisms, and relationships).

我检索到的，关于ARDL部分的文献（2000以后）的，全部在[refs](refs.bib)中列出。里面包含了访问网址，请按照上述要求，帮我“对比他们，寻找共同模式、关键差异以及它们如何解答研究问题。”。

下面是可直接放入 **Results** 部分的英文—中文对照草稿，重点围绕 ARDL 文献如何对比、呈现共同模式、关键差异，以及它们如何回应你的 RQ：“How can temporal delays between formal and informal economic processes be modeled?” 

## Results: ARDL-based evidence on temporal dynamics between formal and informal economic processes

## 结果：ARDL 文献中正式与非正式经济过程之间的时间动态证据

**English.**
Across the reviewed ARDL studies, temporal delay is modeled primarily through a separation between short-run responses, long-run equilibrium relationships, and error-correction adjustment. In this sense, ARDL is not only an estimation technique but also a way of organizing the temporal structure of formal–informal economic interactions: current changes in formal variables such as GDP, taxation, inflation, financial development, institutional quality, or fiscal balance are allowed to affect informal or shadow-economy outcomes with lags, while the error-correction term captures the speed at which deviations from long-run equilibrium are corrected. This shared logic is explicit in the Romanian currency-demand study, where ARDL bounds testing and ECM are used to test the long-run relationship between currency demand and its determinants, and in the more recent Romanian DGE-based study, where ARDL is used to distinguish short-term fluctuations from long-term equilibrium in informal output.  

**中文。**
在所综述的 ARDL 文献中，时间延迟主要通过短期反应、长期均衡关系和误差修正调整三部分来刻画。因此，ARDL 不仅是一种估计方法，也是一种组织正式—非正式经济互动时间结构的方式：GDP、税收、通胀、金融发展、制度质量或财政平衡等正式部门变量的当期变化，可以通过滞后项影响非正式或影子经济结果；而误差修正项则刻画系统偏离长期均衡后恢复的速度。这一共同逻辑在罗马尼亚货币需求研究中表现为 ARDL bounds test 与 ECM 被用于检验货币需求及其决定因素之间的长期关系；在较新的罗马尼亚 DGE 非正式产出研究中，则表现为 ARDL 被用于区分非正式产出的短期波动与长期均衡。 

**English.**
A first common pattern is that many studies treat the informal economy as an unobserved stock or process that must be estimated before its lagged interaction with the formal economy can be modeled. Dobre and Davidescu estimate the Romanian shadow economy through a revised currency-demand approach and then use ARDL to test whether currency demand, income, tax burden, interest rates, and wage ratios are cointegrated; the resulting shadow-economy estimates decline from about 45% to 37.4% of official GDP over the sample period. Khuong et al. similarly compute Pakistan’s informal economy through a currency-demand equation before applying ARDL to growth dynamics, while Sultana et al. construct Bangladesh’s informal-sector series using a MIMIC model before applying linear and nonlinear ARDL. The more recent Romanian study departs from currency-demand and MIMIC traditions by using DGE-based estimates of informal output, which the authors present as a more precise measure of uncontrolled activity than cash-demand proxies.    

**中文。**
第一个共同模式是，许多研究首先把非正式经济视为一个不可直接观测的存量或过程，并在建模其与正式经济的滞后互动之前先对其进行估计。Dobre 和 Davidescu 使用修正后的货币需求法估计罗马尼亚影子经济，再用 ARDL 检验货币需求、收入、税负、利率和工资比率之间是否存在协整关系；其估计显示，样本期内影子经济占官方 GDP 的比例约从 45% 降至 37.4%。Khuong 等人同样先通过货币需求方程计算巴基斯坦非正式经济，再使用 ARDL 分析其与经济增长的动态关系；Sultana 等人则先用 MIMIC 模型构造孟加拉国非正式部门序列，再使用线性和非线性 ARDL。较新的罗马尼亚研究区别于货币需求法和 MIMIC 传统，采用基于 DGE 的非正式产出估计，并将其视为比现金需求代理变量更精确的非受控经济活动度量。   

**English.**
A second common pattern is that formal-sector pressures tend to transmit to informality with different short- and long-run profiles. In Romania, Dobre and Davidescu find that taxes and income have the strongest short-run effects on currency demand, while the significant and negative error-correction term supports the existence of long-run adjustment. In the global panel study by Canh, Schinckus, and Thanh, institutional quality, FDI, trade openness, and the shadow economy are connected through bi-causal relationships; trade openness reduces the shadow economy in both the short and long run, FDI is negative in the short run but positive in the long run, and institutional effects vary by governance dimension. In the N-11 panel, financial market development, country risk, tax burden, GDP, unemployment, and interaction terms enter a panel ARDL framework, showing that financial and institutional conditions affect the size of the shadow economy differently across horizons.   

**中文。**
第二个共同模式是，正式部门压力通常会以不同的短期和长期路径传导至非正式经济。在罗马尼亚研究中，Dobre 和 Davidescu 发现税收和收入对货币需求具有最强的短期影响，而显著且为负的误差修正项支持长期调整关系的存在。在 Canh、Schinckus 和 Thanh 的全球面板研究中，制度质量、FDI、贸易开放与影子经济之间存在双向因果关系；贸易开放在短期和长期均会降低影子经济，FDI 在短期为负向影响、长期为正向影响，而制度效应则因治理维度不同而呈现异质性。在 N-11 面板研究中，金融市场发展、国家风险、税负、GDP、失业率及交互项共同进入 panel ARDL 框架，结果显示金融与制度条件对影子经济规模的影响具有明显的时间维度差异。  

**English.**
A third pattern is that ARDL studies model not only how the formal economy affects informality, but also how informality feeds back into formal economic outcomes. In Pakistan, Khuong et al. report that the informal economy accounts for about 56% of GDP and that the ARDL model is statistically supported by the Wald F-test; their conclusion states that informality plays a significant role in weakening Pakistan’s formal sector, with diagnostic tests indicating no autocorrelation or heteroskedasticity. In Zimbabwe, Chamisa and Sunde reverse the dependent-variable direction by modeling tax revenue as the formal outcome and shadow economy as one determinant; they find that the shadow economy significantly hinders tax revenue in the short run, while the long-run coefficient is negative but insignificant. In the G5 study, the shadow economy is also treated as part of a broader financial-system dynamic: both financial inclusion and the shadow economy increase financial instability, but governance and banking regulation moderate these effects.   

**中文。**
第三个模式是，ARDL 文献不仅建模正式经济如何影响非正式经济，也建模非正式经济如何反过来影响正式经济结果。在巴基斯坦研究中，Khuong 等人报告非正式经济约占 GDP 的 56%，且 Wald F-test 支持 ARDL 模型的统计显著性；其结论指出，非正式经济在削弱巴基斯坦正式部门方面具有显著作用，同时诊断检验显示模型不存在自相关和异方差问题。在津巴布韦研究中，Chamisa 和 Sunde 将因变量方向反转，以税收收入作为正式部门结果、以影子经济作为解释变量之一；他们发现影子经济在短期显著抑制税收收入，而长期系数为负但不显著。在 G5 研究中，影子经济还被置于更广泛的金融系统动态之中：金融包容和影子经济都会增加金融不稳定，但治理和银行监管会调节这些影响。  

**English.**
The most important methodological difference concerns whether temporal effects are assumed to be symmetric or asymmetric. Linear ARDL treats increases and decreases in informality as mirror-image shocks, whereas NARDL allows positive and negative changes to have different magnitudes and timing. Sultana et al. show that the symmetric effect of informality on Bangladeshi growth is insignificant in the linear ARDL model, but the NARDL model reveals significant asymmetry: an increase in informal activity lowers growth, while a decrease in informal activity raises growth, with the growth-enhancing effect of declining informality being larger in both the short and long run. Similarly, Mhadhbi and Terzi use a NARDL framework for Tunisia and find that shadow-economy changes do not significantly affect the finance–growth nexus in the short run but become important in the long run, with positive changes in the shadow economy weakening the long-run effect of financial growth.  

**中文。**
最重要的方法差异在于，时间效应是否被假定为对称。线性 ARDL 通常把非正式经济的增加和减少视为镜像冲击，而 NARDL 允许正向变化和负向变化在幅度与时序上存在差异。Sultana 等人发现，在孟加拉国样本中，线性 ARDL 模型中的非正式经济对增长的对称效应并不显著，但 NARDL 显示出显著的不对称性：非正式活动增加会降低增长，而非正式活动减少会提高增长，并且无论在短期还是长期，非正式经济下降带来的增长促进效应都更大。类似地，Mhadhbi 和 Terzi 在突尼斯研究中使用 NARDL，发现影子经济变化在短期内并未显著影响金融—增长关系，但在长期中发挥重要作用；影子经济的正向变化会削弱金融增长的长期效果。 

**English.**
Another key difference lies in the treatment of institutions and financial conditions. Some studies include them as direct determinants of informality, while others model them as moderators that reshape the lagged relationship between formal and informal processes. Canh et al. emphasize heterogeneity across institutional dimensions: control of corruption and rule of law matter in the short run, whereas political stability matters in the long run. Rahman et al. use interaction terms between financial market development and political, economic, and financial risk, showing that country risk conditions the relationship between financial market development and shadow-economy size. Syed’s G5 study similarly uses interaction terms to show that governance and banking regulation reduce the long-run destabilizing effects associated with financial inclusion and shadow economy, and also that financial inclusion, governance, and banking regulation reduce the shadow economy itself.   

**中文。**
另一个关键差异在于制度与金融条件的处理方式。有些研究将其作为非正式经济的直接决定因素，另一些研究则将其建模为调节变量，用来改变正式与非正式经济过程之间的滞后关系。Canh 等人强调制度维度之间的异质性：腐败控制和法治主要在短期发挥作用，而政治稳定主要在长期发挥作用。Rahman 等人使用金融市场发展与政治、经济和金融风险之间的交互项，表明国家风险会调节金融市场发展与影子经济规模之间的关系。Syed 的 G5 研究同样使用交互项，显示治理和银行监管能够降低金融包容与影子经济所带来的长期金融不稳定效应，同时金融包容、治理和银行监管本身也会降低影子经济规模。  

**English.**
The reviewed studies therefore suggest that ARDL answers the research question by translating “temporal delay” into three empirically observable objects: lagged effects, long-run cointegration, and speed of adjustment. Lagged effects capture delayed responses of informal activity to formal-sector shocks; cointegration captures whether formal and informal processes share a stable long-run path; and the error-correction term measures how quickly short-run disequilibria are absorbed. This is especially clear in the 2025 Romania study, where informal output adjusts rapidly toward equilibrium, with an estimated 79% adjustment speed, and in the Zimbabwe study, where a near-unity error-correction term suggests rapid adjustment of tax revenue after shocks.  

**中文。**
因此，所综述的研究表明，ARDL 通过将“时间延迟”转化为三个可实证观察的对象来回应研究问题：滞后效应、长期协整和调整速度。滞后效应用于捕捉非正式经济活动对正式部门冲击的延迟反应；协整关系用于判断正式与非正式经济过程是否共享稳定的长期路径；误差修正项则用于衡量短期失衡被吸收的速度。这一点在 2025 年罗马尼亚研究中尤其明显，该研究估计非正式产出以约 79% 的速度向长期均衡调整；在津巴布韦研究中，接近 1 的误差修正项也表明税收收入在冲击后具有快速回归长期均衡的特征。 

**English.**
At the same time, the ARDL evidence also reveals a research gap. Most studies identify temporal delays statistically through selected lag lengths and error-correction terms, but they rarely theorize why a particular formal-sector process should affect informality after a specific number of periods. Annual data are common, especially in panel ARDL studies, which limits the ability to observe shorter lead–lag mechanisms between formal and informal activity. Moreover, measurement uncertainty remains substantial because the informal economy is estimated through different proxies—currency demand, MIMIC, Medina–Schneider estimates, or DGE-based output—and these measures may themselves embed different delay structures. Dobre and Davidescu explicitly warn that shadow-economy estimates should be interpreted with caution because hidden activity is inherently difficult to measure.    

**中文。**
与此同时，ARDL 证据也揭示了一个研究空白。多数研究通过所选择的滞后阶数和误差修正项在统计上识别时间延迟，但较少从理论上解释为什么某一正式部门过程应当在特定期数之后影响非正式经济。许多 panel ARDL 研究使用年度数据，这限制了对正式与非正式经济活动之间更短周期 lead–lag 机制的观察能力。此外，非正式经济的度量本身仍存在较大不确定性，因为不同研究分别使用货币需求法、MIMIC、Medina–Schneider 数据或 DGE 非正式产出估计，而这些度量可能内嵌了不同的时间延迟结构。Dobre 和 Davidescu 也明确提醒，由于隐藏经济活动本身难以度量，影子经济估计结果应当谨慎解释。   

**English.**
Overall, the ARDL literature contributes to the modeling of temporal delays by showing that formal and informal economic processes are dynamically interdependent rather than contemporaneously linked only. The most robust common finding is not a single universal sign of the relationship, but the existence of horizon-dependent effects: some drivers matter immediately, others only in the long run, and asymmetric models show that increases and decreases in informality may follow different adjustment paths. For a review focused on research gaps, the key implication is that future work should move beyond merely including lags and should explicitly compare theoretically grounded delay mechanisms across channels—taxation, finance, institutions, labor markets, and macroeconomic shocks—preferably with higher-frequency data and nonlinear specifications.    

**中文。**
总体而言，ARDL 文献对时间延迟建模的贡献在于，它表明正式与非正式经济过程并非仅仅存在同期关联，而是具有动态相互依赖关系。最稳健的共同发现并不是某一种统一方向的关系，而是影响具有明显的时间视角依赖性：有些驱动因素立即发挥作用，有些只在长期显现，而不对称模型进一步表明，非正式经济的上升与下降可能遵循不同的调整路径。对于一篇旨在定位研究空白的综述而言，关键启示是，未来研究不应仅停留在“加入滞后项”，而应在税收、金融、制度、劳动力市场和宏观冲击等渠道之间，明确比较具有理论基础的延迟机制；同时，应尽可能使用更高频数据和非线性设定。   



## 1. 我对你的 AC-gate / CMDL 的理解

你的 CMDL 不是“把 ARDL 换成 LSTM”，而是把 ARDL 中的 **distributed lag** 重新定义为一个可学习、可解释、且由实体条件调节的滞后发现机制。README 中对 CMDL 的定位是：用于 **entity-conditioned heterogeneous lag discovery** 的可测试框架；核心模型 AC-GATE 由 **Adaptive Conditioning Encoder + Scale-Invariant Lag Gate + LSTM backbone** 组成，并配有可解释的 (k^*) 机制检验。

具体地说，AC-GATE 先把实体层面的静态代理变量 (p_i) 编码成吸收能力/条件状态 (z_i)，再由 (z_i) 生成实体特定的滞后权重分布 (\omega(k\mid z_i))，并用该权重把历史输入聚合成 (c_{i,t}=\sum_k \omega_{i,k}X_{i,t-k})。随后，LSTM 只处理这个经过 lag gate 聚合后的历史上下文，而不是直接吃当前输入；这样做是为了避免 shortcut path。最终模型还能输出实体层面的有效滞后 (k_i^*=\sum_k k\omega_{i,k})，这正是它区别于普通黑箱序列模型的可解释机制产物。

因此，CMDL/AC-gate 的核心贡献可以概括为一句话：**把“滞后阶数选择”从研究者预设的固定结构，转化为由实体特征条件化的可学习 lag distribution，并输出可检验的 entity-specific effective lag。**

---

## 2. 现有 ARDL 应用已经证明“滞后重要”，但没有真正解决“谁的滞后不同、为什么不同”

现有 ARDL 文献的共同优点是，它能把正式经济与非正式经济之间的关系拆成短期效应、长期均衡和误差修正速度。例如，罗马尼亚 informal output 研究明确用 ARDL 区分短期波动与长期均衡，并估计通胀、财政平衡、政治稳定、利息支付、GDP per capita、自雇等变量对 informality 的短期和长期影响；该研究还把“informal economy 对宏观冲击的调整速度”作为研究问题之一。

问题在于，ARDL 的“滞后”仍主要是模型规格中的滞后项，而不是一个可解释的实体机制。它能回答：某变量在短期和长期是否显著；但很难回答：**为什么 A 国对正式部门冲击的反应滞后 1–2 年，而 B 国滞后 5–6 年？这种差异是否由制度质量、吸收能力、金融深度、能源结构或发展阶段决定？**

这正是 CMDL 的切入点：不是再估计一个平均滞后效应，而是让 (z_i) 条件化 (\omega(k\mid z_i))，直接学习每个实体的 lag profile，并用 (k_i^*) 把它显式化。

---

## 3. Panel ARDL 和交互项能处理“调节效应”，但调节的是系数，不是滞后机制

N-11 国家研究已经说明，country risk 会调节 financial market development 与 shadow economy 的关系；该研究使用 panel ARDL 估计长短期关系，并用政治、经济、金融风险与 FMD 的交互项刻画 moderation。 其理论讨论也明确指出，政治风险、经济风险和金融风险会影响金融市场表现，进而影响 shadow economy，并提出“country risk 是否调节 FMD–SE nexus”的研究问题。

但是，这类 moderation 仍然是 **researcher-specified interaction term**。也就是说，研究者先规定“FMD × risk”这个交互项，再估计它对 SE 的系数。它能说明“风险改变了 FMD 的影响方向或强度”，但不能说明“风险改变了 FMD 影响 SE 的时间滞后结构”。

CMDL/AC-gate 的研究空白正好在这里：**现有 ARDL moderation 主要是 coefficient moderation，而 CMDL 做的是 lag-structure moderation。**
也就是说，AC-gate 不是只问：

[
\frac{\partial SE}{\partial FMD}
]

是否随 country risk 改变，而是进一步问：

[
\omega(k\mid z_i)
]

是否随实体条件 (z_i) 改变。这个问题是传统交互项 ARDL 很难直接回答的。

---

## 4. NARDL 能处理非线性和不对称，但仍不是 entity-conditioned lag discovery

NARDL 文献已经开始突破线性 ARDL。例如，孟加拉国研究指出，非正式部门对经济增长的影响具有短期和长期的不对称性，并且非正式部门下降对产出和增长的影响更大。 突尼斯研究也使用 NARDL 检验 shadow economy 的 threshold/asymmetric effect，发现 shadow economy 的正向变化会使金融增长的长期效应转负，而短期影响不显著、长期影响显著。

但 NARDL 的不对称主要是 **shock-side asymmetry**：例如把变量分解成正向变化和负向变化。它解决的是“上升和下降是否不同”，而不是“不同国家/实体的滞后结构是否不同”。换句话说，NARDL 能说：

[
X^+ \neq X^-
]

但 CMDL 要说的是：

[
\omega_i(k) \neq \omega_j(k), \quad \text{and this difference is conditioned by } z_i,z_j.
]

因此，NARDL 是对 ARDL 的非线性扩展，而 CMDL 是对 distributed-lag 思想的 **条件化机制扩展**。

---

## 5. 为什么需要 CMDL：现有 ARDL 的研究空白可以写成四个层次

### Gap 1：从“平均滞后”到“实体异质滞后”

现有 ARDL/P-ARDL/CS-ARDL 往往估计平均短期效应、长期效应或组内平均效应。即使是 panel ARDL，也通常强调 pooled long-run / short-run coefficients，而不是每个实体自身的 lag distribution。你的 Grouped ARDL baseline 已经是强 econometric baseline，但它仍然是按 anchor group 做 distributed-lag OLS，不是连续、可学习的 entity-conditioned lag mechanism。README 也把 Grouped ARDL 定位为 per-anchor-group distributed-lag OLS baseline，而 CMDL/AC-GATE 才是主方法。

**CMDL 的必要性**：把 group-level lag heterogeneity 推进到 entity-level lag heterogeneity。

### Gap 2：从“系数调节”到“滞后结构调节”

现有文献已经承认制度、金融风险、治理、财政稳定等变量会调节 formal–informal nexus。例如 N-11 研究强调 country risk 的调节作用，G5 研究强调治理和银行监管对 shadow economy、financial inclusion 与 financial stability 关系的调节作用。 但这些调节基本都停留在交互项层面。

**CMDL 的必要性**：让实体条件变量不只是改变系数大小，而是改变“哪个滞后期最重要”。

### Gap 3：从“短期/长期二分”到“完整 lag distribution”

ARDL 常把动态结构压缩成短期项、长期项、ECM 调整速度。这个框架有解释力，但过于粗粒度。对于你的 RQ——formal and informal economic processes 的 temporal delays——真正需要的是“滞后分布”而不是简单的“短期/长期”。AC-gate 输出 (\omega_{i,k}) 和 (k_i^*)，因此可以把 temporal delay 变成可视化、可比较、可检验的对象。

**CMDL 的必要性**：把 temporal delay 从“是否存在短期/长期效应”推进到“滞后峰值在哪里、分布是否集中、实体之间是否不同”。

### Gap 4：从“预测黑箱”到“可解释机制学习”

Plain LSTM 能学习非线性动态，但没有 AC encoder 和 lag gate；README 也明确把 Plain LSTM 作为“without AC encoder / lag gate”的神经基线。 所以普通 LSTM 即使预测更好，也不能告诉你为什么某些实体响应更慢。反过来，ARDL 可解释但表达能力弱，难以学习复杂非线性与实体条件异质性。

**CMDL 的必要性**：在 ARDL 的可解释 lag 思想和 LSTM 的非线性序列学习之间建立中间层：既能预测，又能输出 (k_i^*) 作为机制解释。

---

## 6. 可直接放入论文的 Research gap 表述

你可以这样写：

> Existing ARDL and NARDL applications have shown that formal and informal economic processes are dynamically linked through short-run effects, long-run equilibria, asymmetric shocks, and error-correction adjustment. However, these approaches usually treat lag structures as fixed, researcher-specified, or group-level objects. Even when moderation is considered, it is typically modeled through interaction terms that alter coefficient magnitudes rather than the timing of responses. As a result, existing models cannot directly discover whether different entities respond to formal-sector shocks with different lag profiles, nor whether such heterogeneous delays are conditioned by institutional, absorptive-capacity, or structural characteristics. CMDL addresses this gap by introducing AC-GATE, a proxy-conditioned lag mechanism that learns an entity-specific lag-weight distribution and yields an interpretable effective lag (k_i^*), thereby transforming temporal delay from a predefined econometric specification into a testable, entity-conditioned mechanism.

中文对应：

> 现有 ARDL 与 NARDL 应用已经表明，正式经济与非正式经济过程之间存在短期效应、长期均衡、不对称冲击和误差修正调整等动态联系。然而，这些方法通常将滞后结构视为固定的、由研究者预先设定的，或至多是组层面的对象。即使考虑调节效应，现有研究也多通过交互项改变系数大小，而不是改变响应发生的时间结构。因此，现有模型难以直接发现不同实体是否以不同滞后模式响应正式部门冲击，也难以检验这种异质性延迟是否由制度质量、吸收能力或结构性特征所条件化。CMDL 通过 AC-GATE 填补这一空白：它引入由实体代理变量条件化的 lag gate，学习实体特定的滞后权重分布，并输出可解释的有效滞后 (k_i^*)，从而将 temporal delay 从预设的计量规格转化为可检验的实体条件机制。

---

## 7. 最核心的一句话

**ARDL 证明了 delayed response 存在；NARDL 证明了 response 可以非线性/不对称；但 CMDL/AC-gate 进一步解决的是：delayed response 为什么在不同实体之间不同，以及这种差异能否由实体吸收能力或制度条件来解释。**
