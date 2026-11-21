# 参考文献（候補）

※ 記憶ベースで列挙したもの。巻号・ページは **docs/ で bib 化する前に必ず原典で確認** すること。

## SVモデル・MCMC
- Taylor, S. J. (1986). *Modelling Financial Time Series*. Wiley. — SV モデルの古典
- Jacquier, E., Polson, N. G., & Rossi, P. E. (1994). Bayesian analysis of stochastic volatility models. *Journal of Business & Economic Statistics*, 12(4), 371–389.
- Harvey, A., Ruiz, E., & Shephard, N. (1994). Multivariate stochastic variance models. *Review of Economic Studies*, 61(2), 247–264. — QML（線形化＋カルマンフィルタ）
- Kim, S., Shephard, N., & Chib, S. (1998). Stochastic volatility: Likelihood inference and comparison with ARCH models. *Review of Economic Studies*, 65(3), 361–393. — 混合正規近似による MCMC、事前分布の標準
- Omori, Y., Chib, S., Shephard, N., & Nakajima, J. (2007). Stochastic volatility with leverage: Fast and efficient likelihood inference. *Journal of Econometrics*, 140(2), 425–449.
- Kastner, G., & Frühwirth-Schnatter, S. (2014). Ancillarity-sufficiency interweaving strategy (ASIS) for boosting MCMC estimation of stochastic volatility models. *Computational Statistics & Data Analysis*, 76, 408–423. — R `stochvol`
- Chan, J. C. C., & Jeliazkov, I. (2009). Efficient simulation and integrated likelihood estimation in state space models. *International Journal of Mathematical Modelling and Numerical Optimisation*, 1(1/2), 101–120. — 帯精度行列による状態サンプリング
- 渡部敏明 (2000).『ボラティリティ変動モデル』朝倉書店. — 日本語の定番解説書
- Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity. *Journal of Econometrics*, 31(3), 307–327. — GARCH（比較対象）

## 変分推論
- Blei, D. M., Kucukelbir, A., & McAuliffe, J. D. (2017). Variational inference: A review for statisticians. *Journal of the American Statistical Association*, 112(518), 859–877.
- Kucukelbir, A., Tran, D., Ranganath, R., Gelman, A., & Blei, D. M. (2017). Automatic differentiation variational inference. *Journal of Machine Learning Research*, 18(14), 1–45.
- Kingma, D. P., & Welling, M. (2014). Auto-encoding variational Bayes. *ICLR*. — 再パラメータ化トリック
- Ormerod, J. T., & Wand, M. P. (2010). Explaining variational approximations. *The American Statistician*, 64(2), 140–153.
- Tan, L. S. L., & Nott, D. J. (2018). Gaussian variational approximation with sparse precision matrices. *Statistics and Computing*, 28, 259–275. — **構造化ガイドの直接の理論的背景**
- Archer, E., Park, I. M., Buesing, L., Cunningham, J., & Paninski, L. (2015). Black box variational inference for state space models. arXiv:1511.07367.
- Krishnan, R. G., Shalit, U., & Sontag, D. (2017). Structured inference networks for nonlinear state space models. *AAAI*.
- Gunawan, D., Kohn, R., & Nott, D. (2021). Variational Bayes approximation of factor stochastic volatility models. *International Journal of Forecasting*. — SV × VI の先行研究（巻号要確認）
- Yao, Y., Vehtari, A., Simpson, D., & Gelman, A. (2018). Yes, but did it work?: Evaluating variational inference. *ICML*. — PSIS $\hat k$ による VI 診断

## MCMC・ソフトウェア
- Hoffman, M. D., & Gelman, A. (2014). The No-U-Turn sampler. *Journal of Machine Learning Research*, 15, 1593–1623.
- Phan, D., Pradhan, N., & Jankowiak, M. (2019). Composable effects for flexible and accelerated probabilistic programming in NumPyro. arXiv:1912.11554.
- Bradbury, J., et al. (2018). JAX: composable transformations of Python+NumPy programs. http://github.com/jax-ml/jax

## VaR バックテスト
- Kupiec, P. H. (1995). Techniques for verifying the accuracy of risk measurement models. *Journal of Derivatives*, 3(2), 73–84.
- Christoffersen, P. F. (1998). Evaluating interval forecasts. *International Economic Review*, 39(4), 841–862.

## 追加調査したいもの
- SV × VI の最近の先行研究（2020 年以降）のサーベイ → 本稿の新規性の位置付けに必要
- 日本の株価指数での SV 推定例（大森・渡部らの実証研究）
