# 参考文献（候補）

※ 記憶ベースで列挙したもの。巻号・ページは **docs/ で bib 化する前に必ず原典で確認** すること。
※ 2026-11-24：1 章で引用したものは原典（Crossref・出版社・JMLR/PMLR/NeurIPS・arXiv・CiNii）で確認し `docs/references.bib` に登録済み（項目末尾に【bib済】）。

## SVモデル・MCMC
- Taylor, S. J. (1986). *Modelling Financial Time Series*. Wiley. — SV モデルの古典 【bib済】
- Jacquier, E., Polson, N. G., & Rossi, P. E. (1994). Bayesian analysis of stochastic volatility models. *Journal of Business & Economic Statistics*, 12(4), 371–389. 【bib済】
- Harvey, A., Ruiz, E., & Shephard, N. (1994). Multivariate stochastic variance models. *Review of Economic Studies*, 61(2), 247–264. — QML（線形化＋カルマンフィルタ） 【bib済】
- Kim, S., Shephard, N., & Chib, S. (1998). Stochastic volatility: Likelihood inference and comparison with ARCH models. *Review of Economic Studies*, 65(3), 361–393. — 混合正規近似による MCMC、事前分布の標準 【bib済】
- Omori, Y., Chib, S., Shephard, N., & Nakajima, J. (2007). Stochastic volatility with leverage: Fast and efficient likelihood inference. *Journal of Econometrics*, 140(2), 425–449. 【bib済】
- Kastner, G., & Frühwirth-Schnatter, S. (2014). Ancillarity-sufficiency interweaving strategy (ASIS) for boosting MCMC estimation of stochastic volatility models. *Computational Statistics & Data Analysis*, 76, 408–423. — R `stochvol` 【bib済】
- Chan, J. C. C., & Jeliazkov, I. (2009). Efficient simulation and integrated likelihood estimation in state space models. *International Journal of Mathematical Modelling and Numerical Optimisation*, 1(1/2), 101–120. — 帯精度行列による状態サンプリング 【bib済】
- 渡部敏明 (2000).『ボラティリティ変動モデル』朝倉書店. — 日本語の定番解説書 【bib済】
- Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity. *Journal of Econometrics*, 31(3), 307–327. — GARCH（比較対象） 【bib済】

## 変分推論
- Blei, D. M., Kucukelbir, A., & McAuliffe, J. D. (2017). Variational inference: A review for statisticians. *Journal of the American Statistical Association*, 112(518), 859–877. 【bib済】
- Kucukelbir, A., Tran, D., Ranganath, R., Gelman, A., & Blei, D. M. (2017). Automatic differentiation variational inference. *Journal of Machine Learning Research*, 18(14), 1–45. 【bib済】
- Kingma, D. P., & Welling, M. (2014). Auto-encoding variational Bayes. *ICLR*. — 再パラメータ化トリック 【bib済】
- Ormerod, J. T., & Wand, M. P. (2010). Explaining variational approximations. *The American Statistician*, 64(2), 140–153.
- Tan, L. S. L., & Nott, D. J. (2018). Gaussian variational approximation with sparse precision matrices. *Statistics and Computing*, 28, 259–275. — **構造化ガイドの直接の理論的背景** 【bib済】
- Archer, E., Park, I. M., Buesing, L., Cunningham, J., & Paninski, L. (2015). Black box variational inference for state space models. arXiv:1511.07367. 【bib済】
- Johnson, M. J., Duvenaud, D., Wiltschko, A. B., Datta, S. R., & Adams, R. P. (2016). Composing graphical models with neural networks for structured representations and fast inference. *NeurIPS*. — 擬似観測（共役な因子）による構造化 VI。条件付き構造化族の考え方の背景 【bib済】
- Krishnan, R. G., Shalit, U., & Sontag, D. (2017). Structured inference networks for nonlinear state space models. *AAAI*.
- Gunawan, D., Kohn, R., & Nott, D. (2021). Variational Bayes approximation of factor stochastic volatility models. *International Journal of Forecasting*, 37(4), 1355–1375. — SV × VI の先行研究。逐次的な変分更新も 【bib済】
- Yao, Y., Vehtari, A., Simpson, D., & Gelman, A. (2018). Yes, but did it work?: Evaluating variational inference. *ICML*. — PSIS $\hat k$ による VI 診断

## MCMC・ソフトウェア
- Hoffman, M. D., & Gelman, A. (2014). The No-U-Turn sampler. *Journal of Machine Learning Research*, 15, 1593–1623. 【bib済】
- Phan, D., Pradhan, N., & Jankowiak, M. (2019). Composable effects for flexible and accelerated probabilistic programming in NumPyro. arXiv:1912.11554. 【bib済】
- Bradbury, J., et al. (2018). JAX: composable transformations of Python+NumPy programs. http://github.com/jax-ml/jax

## VaR バックテスト
- Kupiec, P. H. (1995). Techniques for verifying the accuracy of risk measurement models. *Journal of Derivatives*, 3(2), 73–84. 【bib済】
- Christoffersen, P. F. (1998). Evaluating interval forecasts. *International Economic Review*, 39(4), 841–862. 【bib済】

## 2026-11-24 の調査で追加（1 章の関連研究。いずれも原典で確認し bib 済み）
- Tan, L. S. L., Bhaskaran, A., & Nott, D. J. (2020). Conditionally structured variational Gaussian approximation with importance weights. *Statistics and Computing*, 30(5), 1255–1272. — 局所変数の変分分布を大域パラメータに条件付ける。**提案族に最も近い先行研究**
- Loaiza-Maya, R., Smith, M. S., Nott, D. J., & Danaher, P. J. (2022). Fast and accurate variational inference for models with many latent variables. *Journal of Econometrics*, 230(2), 339–362. — q(θ)·p(z|θ,y)（潜在変数は正確な条件付き事後）
- Frazier, D. T., Loaiza-Maya, R., & Martin, G. M. (2023). Variational Bayes in state space models: Inferential and predictive accuracy. *JCGS*, 32(3), 793–804. — 状態を近似しない方法ほどパラメータ推定が正確、予測の差は短期では小さい（R2 の平均場の結果と照らして 6 章で議論できる）
- Chan, J. C. C., & Yu, X. (2022). Fast and accurate variational inference for large Bayesian VARs with stochastic volatility. *JEDC*, 143, 104505.
- Quiroz, M., Nott, D. J., & Kohn, R. (2023). Gaussian variational approximations for high-dimensional state space models. *Bayesian Analysis*, 18(3), 989–1016. — 例に Wishart 型多変量 SV
- Clark, P. K. (1973). A subordinated stochastic process model with finite variance for speculative prices. *Econometrica*, 41(1), 135–155. — 混合分布仮説
- Basel Committee on Banking Supervision (2019). Minimum capital requirements for market risk (d457). — FRTB（99% VaR → 97.5% ES）

## 2026-11-24 の調査で追加（2 章。Crossref で確認し bib 済み）
- Harvey, A. C., & Shephard, N. (1996). Estimation of an asymmetric stochastic volatility model for asset returns. *JBES*, 14(4), 429–434. — レバレッジの時点の取り方
- Yu, J. (2005). On leverage in a stochastic volatility model. *Journal of Econometrics*, 127(2), 165–178. — 上の時点の取り方を支持
- Jacquier, E., Polson, N. G., & Rossi, P. E. (2004). Bayesian analysis of stochastic volatility models with fat-tails and correlated errors. *Journal of Econometrics*, 122(1), 185–212.
- Chib, S., Nardari, F., & Shephard, N. (2002). Markov chain Monte Carlo methods for stochastic volatility models. *Journal of Econometrics*, 108(2), 281–316. — SV-t
- Juárez, M. A., & Steel, M. F. J. (2010). Model-based clustering of non-Gaussian panel data based on skew-t distributions. *JBES*, 28(1), 52–66. — ν の Gamma(2, 0.1) 事前
- Christie, A. A. (1982). The stochastic behavior of common stock variances. *Journal of Financial Economics*, 10(4), 407–432. — レバレッジ効果
- Papaspiliopoulos, O., Roberts, G. O., & Sköld, M. (2007). A general framework for the parametrization of hierarchical models. *Statistical Science*, 22(1), 59–73. — 中心化・非中心化

## 2026-11-25 の調査で追加（3 章。Crossref・JMLR・PMLR・arXiv・JAX 公式で確認し bib 済み）
- Metropolis ら (1953) J. Chem. Phys. 21(6) 1087–1092 / Hastings (1970) Biometrika 57(1) 97–109 / Duane ら (1987) Phys. Lett. B 195(2) 216–222
- Neal (2011) MCMC using Hamiltonian dynamics. Handbook of MCMC, 113–162 / Betancourt (2017) arXiv:1701.02434
- Gelman & Rubin (1992) Stat. Sci. 7(4) 457–472 / Gelman ら (2013) BDA3 / Vehtari ら (2021) 改良 R̂, Bayesian Analysis 16(2) 667–718
- Bishop (2006) PRML（平均場の分散過小評価）/ **Turner & Sahani (2011)** Two problems with variational EM for time series models（パラメータが近似の正確な方向に偏る。本稿の φ 崩壊と同じ現象）
- Robbins & Monro (1951) / Kingma & Ba (2015) Adam / Vehtari ら (2024) PSIS, JMLR 25(72) / Bradbury ら (2018) JAX

## 追加調査したいもの
- SV × VI の最近の先行研究（2020 年以降）のサーベイ → 本稿の新規性の位置付けに必要
- 日本の株価指数での SV 推定例（大森・渡部らの実証研究）

## 6 章で追加（2026-11-26、Crossref で確認）
- Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. *JBES*, 13(3), 253–263. doi:10.1080/07350015.1995.10524599 【bib済】
- Newey, W. K., & West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica*, 55(3), 703–708. 【bib済】
- Gneiting, T., & Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation. *JASA*, 102(477), 359–378. 【bib済】
- Sheppard, K. et al. arch（Python パッケージ v9.0.0、2026-11-11 公開、doi:10.5281/zenodo.593254）【bib済】
- J-Quants API（株式会社 JPX 総研、https://jpx-jquants.com/）【bib済】。サイトは自動アクセスに 403 を返すため URL の到達は未確認（memo/07 の調査時に公式で確認済み）

## 7 章で追加（2026-11-27）
- 金融庁（2021）「モデル・リスク管理に関する原則」2021-11-12 公表。https://www.fsa.go.jp/news/r3/ginkou/20211112.html（ページの表題で確認）【bib済】

## 8 章で追加（2026-11-27、Crossref・出版社ページで確認）【すべて bib済】
- Rezende & Mohamed (2015) Variational inference with normalizing flows. ICML, PMLR 37, 1530–1538
- Kingma, Salimans, Jozefowicz, Chen, Sutskever & Welling (2016) Improved variational inference with inverse autoregressive flow. NeurIPS 29
- Chib, Nardari & Shephard (2006) Analysis of high dimensional multivariate SV models. J. Econometrics 134(2), 341–371. doi:10.1016/j.jeconom.2005.06.026
- Asai, McAleer & Yu (2006) Multivariate SV: a review. Econometric Reviews 25(2–3), 145–175
- Eraker, Johannes & Polson (2003) The impact of jumps in volatility and returns. J. Finance 58(3), 1269–1300
- Takahashi, Omori & Watanabe (2009) Estimating SV models using daily returns and realized volatility simultaneously. CSDA 53(6), 2404–2426. doi:10.1016/j.csda.2008.07.039
- Glosten, Jagannathan & Runkle (1993) J. Finance 48(5), 1779–1801 / Nelson (1991) Econometrica 59(2), 347–370
- Krishnan, Shalit & Sontag (2017) Structured inference networks for nonlinear state space models. AAAI 31(1)
- 注：最初に推測した Chib ら 2006・Takahashi ら 2009 の DOI は別の論文だった。題名検索で正しい DOI を確認して登録
