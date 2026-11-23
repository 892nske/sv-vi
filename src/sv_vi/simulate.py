"""合成データの生成。

リターンは % 単位（日次標準偏差 = exp(h_t / 2) [%]）。

- 基本SV       : y_t = exp(h_t/2) ε_t,  h_{t+1} = μ + φ(h_t - μ) + σ η_t
- レバレッジSV : corr(ε_t, η_t) = ρ （当期リターンと翌期ボラのショックが相関。Omori et al. 2007 型）
- SV-t         : ε_t は分散 1 に基準化した自由度 ν の t 分布
"""

from dataclasses import asdict, dataclass

import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class SVParams:
    mu: float
    phi: float
    sigma: float
    rho: float = 0.0
    nu: float | None = None

    def as_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


# memo/02_model.md §2.4 の真値シナリオ（% 単位）
SCENARIOS: dict[str, SVParams] = {
    "standard": SVParams(mu=0.0, phi=0.97, sigma=0.15),
    "high_persistence": SVParams(mu=0.0, phi=0.99, sigma=0.10),
    "low_persistence": SVParams(mu=0.0, phi=0.80, sigma=0.40),
    "leverage": SVParams(mu=0.0, phi=0.97, sigma=0.15, rho=-0.5),
    "student_t": SVParams(mu=0.0, phi=0.97, sigma=0.15, nu=8.0),
}


def simulate_sv(key: jax.Array, T: int, params: SVParams) -> dict[str, np.ndarray]:
    """SV モデルから長さ T の系列を生成し、{"y": リターン, "h": 対数分散} を返す。"""
    mu, phi, sigma, rho, nu = params.mu, params.phi, params.sigma, params.rho, params.nu
    if nu is not None and rho != 0.0:
        raise ValueError("レバレッジと t 誤差の同時指定は扱わない")
    k_eps, k_eta, k_h1, k_t = jax.random.split(key, 4)

    z1 = jax.random.normal(k_eps, (T,))
    z2 = jax.random.normal(k_eta, (T,))
    eps = z1
    eta = rho * z1 + jnp.sqrt(1.0 - rho**2) * z2  # corr(eps_t, eta_t) = rho
    if nu is not None:
        eps = jax.random.t(k_t, nu, (T,)) * jnp.sqrt((nu - 2.0) / nu)

    h1 = mu + sigma / jnp.sqrt(1.0 - phi**2) * jax.random.normal(k_h1)

    def step(h_t, eta_t):
        h_next = mu + phi * (h_t - mu) + sigma * eta_t
        return h_next, h_next

    _, h_rest = jax.lax.scan(step, h1, eta[:-1])
    h = jnp.concatenate([h1[None], h_rest])
    y = jnp.exp(h / 2.0) * eps
    return {"y": np.asarray(y), "h": np.asarray(h)}
