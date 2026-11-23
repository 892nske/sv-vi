"""1 期先予測分布（VaR と対数予測スコア）。

事後分布（または変分分布）からのサンプル (θ^s, h_T^s) を使って

    h_{T+1}^s = μ + φ (h_T^s - μ) + σ η^s
    y_{T+1}   ~ p(y | h_{T+1}^s, θ^s)

とする。レバレッジ SV では当日のショック ε_T = y_T e^{-h_T/2} を条件に η ~ N(ρ ε_T, 1 - ρ²)。
予測分布は混合分布 (1/S) Σ_s p(y | h_{T+1}^s, θ^s) なので、
- VaR：混合分布の分位点（各成分の CDF の平均を二分法で解く）
- 対数予測スコア：log (1/S) Σ_s p(y_obs | h_{T+1}^s, θ^s)
を計算する（乱数で y をサンプルするより精度が良い）。
"""

import numpy as np
from scipy import stats
from scipy.optimize import brentq
from scipy.special import logsumexp

from sv_vi.logdensity import ModelSpec


def next_h(theta: dict, h_last: np.ndarray, y_last: float, spec: ModelSpec, rng) -> np.ndarray:
    mu, phi, sigma = (np.asarray(theta[k]) for k in ("mu", "phi", "sigma"))
    z = rng.standard_normal(h_last.shape)
    if spec.leverage:
        rho = np.asarray(theta["rho"])
        eps_last = y_last * np.exp(-h_last / 2.0)
        eta = rho * eps_last + np.sqrt(1.0 - rho**2) * z
    else:
        eta = z
    return mu + phi * (h_last - mu) + sigma * eta


def _components(theta: dict, h_next: np.ndarray, spec: ModelSpec):
    """各成分の分布（scipy の frozen 分布）。"""
    sd = np.exp(h_next / 2.0)
    if spec.student_t:
        nu = np.asarray(theta["nu"])
        return stats.t(df=nu, loc=0.0, scale=sd * np.sqrt((nu - 2.0) / nu))
    return stats.norm(loc=0.0, scale=sd)


def var_quantile(theta: dict, h_next: np.ndarray, spec: ModelSpec, alpha: float) -> float:
    """予測分布の下側 alpha 分位点（VaR はこの値。負の数）。"""
    comp = _components(theta, h_next, spec)
    f = lambda q: np.mean(comp.cdf(q)) - alpha  # noqa: E731
    sd_max = float(np.exp(h_next.max() / 2.0))
    return brentq(f, -50.0 * sd_max, 0.0, xtol=1e-8)


def log_score(theta: dict, h_next: np.ndarray, spec: ModelSpec, y_obs: float) -> float:
    lp = _components(theta, h_next, spec).logpdf(y_obs)
    return float(logsumexp(lp) - np.log(len(lp)))


def forecast(post, y_last: float, y_obs: float, spec: ModelSpec, seed: int, alphas=(0.01, 0.05)):
    """PosteriorSummary（theta と h_last を持つ）から 1 期先の VaR・対数予測スコア・予測 sd を返す。"""
    rng = np.random.default_rng(seed)
    theta = {k: np.asarray(v) for k, v in post.theta.items()}
    h_next = next_h(theta, np.asarray(post.h_last), y_last, spec, rng)
    out = {f"var{int(a * 100)}": var_quantile(theta, h_next, spec, a) for a in alphas}
    out["log_score"] = log_score(theta, h_next, spec, y_obs)
    out["pred_sd"] = float(np.sqrt(np.mean(np.exp(h_next))))
    return out
