# References

集中维护项目所引用的文献、数据来源与开源软件，方便论文 BibTeX 与 README 直接复用。所有条目按主题分组；末尾给出投稿稿件可直接复用的 BibTeX 模板。

---

## 1. 方法 / 模型参考（Method & Model References）

| Key | Citation | 引用位置 | 用途 |
|---|---|---|---|
| `lim2021tft` | Lim, B., Arık, S. Ö., Loeff, N., & Pfister, T. (2021). Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting. *International Journal of Forecasting*, 37(4), 1748–1764. | Related Work, Method | Gated Residual Network 设计参考；说明 TFT 假设同质滞后，本文不复制其主体（仅作对比参考） |
| `xue2020gated` | Xue, F., Hong, R., He, X., Wang, J., Qian, S., & Xu, C. (2020). Not All Attention Is Needed: Gated Attention Network for Sequence Data. *AAAI*. | Method | 启发 AC-GATE 的门控稀疏化设计 |
| `yang2020panelnn` | Yang, Y., & Zheng, X. (2020). Interpretable Neural Networks for Panel Data Analysis in Economics. *arXiv:2010.05311*. | Related Work | 面板神经网络可解释性参考 |
| `chronopoulos2023dnnpanel` | Chronopoulos, I., Chrysikou, K., & Kapetanios, G. (2023). Deep Neural Network Estimation in Panel Data Models. *arXiv:2305.05083*. | Method, Discussion | 面板 DNN 中实体固定效应的处理 |
| `pesaran1995pmg` | Pesaran, M. H., & Smith, R. (1995). Estimating Long-Run Relationships from Dynamic Heterogeneous Panels. *Journal of Econometrics*, 68(1), 79–113. | Baseline | Pooled Mean Group / 异质系数面板模型——Grouped ARDL 基线的理论基础 |
| `babii2020mlpanel` | Babii, A., Ball, R. T., Ghysels, E., & Striaukas, J. (2020). Machine Learning Panel Data Regressions with an Application to Nowcasting Price–Earnings Ratios. *arXiv:2008.03600*. | Related Work | 组级滞后选择方法——计量经济学竞品 |
| `zhou2025codeal` | Zhou, J., Lin, X., Cao, Y., Lin, Z., & Ong, Y.-S. (2025). Covariate-Adjusted Deep Causal Learning for Heterogeneous Panel Data Models. *arXiv:2502.xxxxx*. | Related Work | 异质处理效应的深度学习方法——因果方向的竞品 |
| `cerqua2025mlpanel` | Cerqua, A., Letta, M., & Menchetti, F. (2025). On the (mis)use of machine learning with panel data. *Oxford Bulletin of Economics and Statistics* (forthcoming). | Discussion | ML 面板应用的常见陷阱——加强自我批判 |
| `thayasivam2025panelsurvey` | Thayasivam, U., et al. (2025). A Comprehensive Survey on Statistical and Deep Learning Models for Panel Data Analysis. *Knowledge and Information Systems* (forthcoming). | Related Work | 最新综述——确保 Related Work 不遗漏最新进展 |

---

## 2. 领域 / 实证文献（Domain & Empirical References）

| Key | Citation | 用途 |
|---|---|---|
| `cohen1990absorptive` | Cohen, W. M., & Levinthal, D. A. (1990). Absorptive Capacity: A New Perspective on Learning and Innovation. *Administrative Science Quarterly*, 35(1), 128–152. | $z_i$ 概念的理论根基——entity-level absorption capacity |
| `schweikl2020it` | Schweikl, S., & Obermaier, R. (2020). Lessons from Three Decades of IT Productivity Research: Towards a Better Understanding of IT-induced Productivity Effects. *Management Review Quarterly*, 70, 461–509. | Solow 悖论 + 滞后效应的领域综述 |
| `feenstra2015pwt` | Feenstra, R. C., Inklaar, R., & Timmer, M. P. (2015). The Next Generation of the Penn World Table. *American Economic Review*, 105(10), 3150–3182. | PWT 11.0 变量定义（`ctfp`, `hc`, `ck`, `rgdpna`, `emp`, `avh`） |
| `akram2020energy` | Akram, R., Chen, F., Khalid, F., Ye, Z., & Majeed, M. T. (2020). Heterogeneous effects of energy efficiency and renewable energy on carbon emissions: Evidence from developing countries. *Journal of Cleaner Production*, 247, 119122. | 能源域 Panel NARDL 变量构造对标 |
| `mirziyoyeva2022re` | Mirziyoyeva, Z., & Salahodjaev, R. (2022). Renewable energy and CO₂ emissions intensity in the top carbon intense countries. *Renewable Energy*, 192, 507–512. | 能源域直接竞品 |
| `appiah2023renewable` | Appiah-Otoo, I., Acheampong, A. O., Song, N., & Obeng, C. K. (2023). Modelling the impact of renewable energy investment on global CO₂ emissions. *Energy Reports*, 9, 5159–5170. | 能源域实证设计参考 |
| `kaufmann2010wgi` | Kaufmann, D., Kraay, A., & Mastruzzi, M. (2010). The Worldwide Governance Indicators: Methodology and Analytical Issues. *World Bank Policy Research Working Paper 5430*. | WGI 治理指标方法论；能源域 stratifier 的官方文档 |

> Medina & Schneider (2018) 与 Elgin et al. (2021) 等影子经济文献保留在 `plan.md` 历史记录中；当前 20-seed 实验体系并未启用影子经济域，故未列入本表。如后续重启该域，请补回。

---

## 3. 数据集（Datasets）

| Key | 数据集 | 引用 / 链接 | 用途 |
|---|---|---|---|
| `pwt110` | Penn World Table 11.0 | Feenstra, Inklaar & Timmer (2015); <https://www.rug.nl/ggdc/productivity/pwt/> | Economics 域 target = `ctfp`；feature_bundle = `effective_labor_aware` |
| `owid_energy` | Our World in Data — Energy | Ritchie, H., Roser, M., & Rosado, P. (2024). "Energy". OurWorldInData.org. <https://github.com/owid/energy-data> | Energy 域 treatment = `renewables_share_energy`，target = `co2_per_unit_energy` |
| `wgi` | World Bank Worldwide Governance Indicators (2024 update) | Kaufmann, Kraay & Mastruzzi (2010); <https://info.worldbank.org/governance/wgi/> | Energy 域 stratifier：`rule_of_law`、`government_effectiveness` |

数据快照与 SHA-256 校验值见 `data/economics/raw/pwt110.csv.meta.json` 与 `data/energy/raw/energy_wgi_merged.csv.meta.json`。

---

## 4. 开源软件（Software & Libraries）

| 名称 | 链接 | 项目内位置 |
|---|---|---|
| PyTorch | <https://pytorch.org/> | 全部模型实现 |
| pandas / numpy / scipy | — | 数据处理与统计检验（`evaluation/significance.py` 使用 `scipy.stats.wilcoxon` 与 `combine_pvalues`） |
| linearmodels | <https://github.com/bashtage/linearmodels> | `baselines/grouped_ardl.py` 与 `baselines/panel_ols.py` |
| matplotlib | <https://matplotlib.org/> | 所有 notebook 可视化 |
| Jupyter | <https://jupyter.org/> | `notebooks/01_synthetic_verify.ipynb` 等 |
| tft-torch *(参考)* | <https://github.com/PlaytikaOSS/tft-torch> | Step 2/3 设计阶段曾参考其 `GatedResidualNetwork` 形式；当前实现未直接依赖该包，TFT 也不在 20-seed baseline 集合内 |
| pytorch-forecasting *(参考)* | <https://github.com/sktime/pytorch-forecasting> | 早期 loader 设计参考；当前数据加载已自实现 |

---

## 5. BibTeX 模板（Submission-ready）

```bibtex
@article{lim2021tft,
  author  = {Lim, Bryan and Ar{\i}k, Sercan {\"O}. and Loeff, Nicolas and Pfister, Tomas},
  title   = {Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting},
  journal = {International Journal of Forecasting},
  volume  = {37}, number = {4}, pages = {1748--1764}, year = {2021}
}

@inproceedings{xue2020gated,
  author    = {Xue, Fei and Hong, Richang and He, Xiangnan and Wang, Jianjun and Qian, Shengsheng and Xu, Changsheng},
  title     = {Not All Attention Is Needed: Gated Attention Network for Sequence Data},
  booktitle = {AAAI}, year = {2020}
}

@article{cohen1990absorptive,
  author  = {Cohen, Wesley M. and Levinthal, Daniel A.},
  title   = {Absorptive Capacity: A New Perspective on Learning and Innovation},
  journal = {Administrative Science Quarterly}, volume = {35}, number = {1}, pages = {128--152}, year = {1990}
}

@article{pesaran1995pmg,
  author  = {Pesaran, M. Hashem and Smith, Ron},
  title   = {Estimating Long-Run Relationships from Dynamic Heterogeneous Panels},
  journal = {Journal of Econometrics}, volume = {68}, number = {1}, pages = {79--113}, year = {1995}
}

@article{feenstra2015pwt,
  author  = {Feenstra, Robert C. and Inklaar, Robert and Timmer, Marcel P.},
  title   = {The Next Generation of the Penn World Table},
  journal = {American Economic Review}, volume = {105}, number = {10}, pages = {3150--3182}, year = {2015}
}

@techreport{kaufmann2010wgi,
  author      = {Kaufmann, Daniel and Kraay, Aart and Mastruzzi, Massimo},
  title       = {The Worldwide Governance Indicators: Methodology and Analytical Issues},
  institution = {World Bank}, type = {Policy Research Working Paper}, number = {5430}, year = {2010}
}

@misc{owid_energy,
  author       = {Ritchie, Hannah and Roser, Max and Rosado, Pablo},
  title        = {Energy},
  howpublished = {OurWorldInData.org}, year = {2024},
  url          = {https://ourworldindata.org/energy}
}
```

> 投稿前请按目标 venue 的格式（IEEE / ACM / Springer LNCS）转换；ICDM 用 IEEEtran，CIKM 用 ACM `acmart`，ECML PKDD 用 Springer `llncs`。
