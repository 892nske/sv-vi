"""SV モデルの同時対数密度 log p(y, θ, h)（純 JAX）。

自前 ADVI（advi.py）はこのモジュールだけを使う。NumPyro 版（models.py）と同じ確率モデルであることは
tests/test_logdensity.py で確認している。

パラメータ化:
- 中心化   (centered=True)  : 潜在変数は h そのもの。h ~ AR1(μ, φ, σ)
- 非中心化 (centered=False) : 潜在変数は x = (h - μ)/σ。x ~ AR1(0, φ, 1)、h = μ + σ x

無制約空間への変換（ADVI 用）:
    μ = u_μ,  φ = tanh(u_φ),  σ = exp(u_σ),  ρ = tanh(u_ρ),  ν = 2 + exp(u_ν)
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.scipy import stats

# 事前分布のハイパーパラメータ（memo/02_model.md）
PRIOR = {
    "mu_loc": 0.0,
    "mu_scale": 10.0,
    "phi_a": 20.0,  # (φ+1)/2 ~ Beta(20, 1.5)
    "phi_b": 1.5,
    "sigma_scale": 1.0,  # σ ~ HalfNormal(1)
    "ig_shape": 2.5,  # σ² ~ InvGamma(2.5, 0.025)（Kim–Shephard–Chib 1998）
    "ig_rate": 0.025,
    "nu_shape": 2.0,  # ν - 2 ~ Gamma(2, rate=0.1)
    "nu_rate": 0.1,
}


@dataclass(frozen=True)
class ModelSpec:
    """どのモデル・パラメータ化を使うか。"""

    leverage: bool = False
    student_t: bool = False
    centered: bool = True
    sigma_prior: str = "halfnormal"  # or "invgamma"

    def __post_init__(self):
        if self.leverage and self.student_t:
            raise ValueError("レバレッジと t 誤差の同時指定は扱わない")
        if self.sigma_prior not in ("halfnormal", "invgamma"):
            raise ValueError(self.sigma_prior)

    @property
    def global_names(self) -> list[str]:
        names = ["mu", "phi", "sigma"]
        if self.leverage:
            names.append("rho")
        if self.student_t:
            names.append("nu")
        return names

    @property
    def n_global(self) -> int:
        return len(self.global_names)

    @property
    def latent_name(self) -> str:
        return "h" if self.centered else "x"


# ---------------------------------------------------------------------------
# 構成要素
# ---------------------------------------------------------------------------


def ar1_log_prob(v: jax.Array, mu, phi, sigma) -> jax.Array:
    """定常 AR(1) 過程の対数密度。v は (..., T)。計算量 O(T)。"""
    sd1 = sigma / jnp.sqrt(1.0 - phi**2)
    lp1 = stats.norm.logpdf(v[..., 0], mu, sd1)
    resid = v[..., 1:] - mu - phi * (v[..., :-1] - mu)
    return lp1 + jnp.sum(stats.norm.logpdf(resid, 0.0, sigma), axis=-1)


def log_prior(theta: dict, spec: ModelSpec) -> jax.Array:
    """大域パラメータの事前対数密度（制約付き空間）。"""
    lp = stats.norm.logpdf(theta["mu"], PRIOR["mu_loc"], PRIOR["mu_scale"])
    # (φ+1)/2 ~ Beta  →  φ の密度にはヤコビアン 1/2 が付く
    lp += stats.beta.logpdf((theta["phi"] + 1.0) / 2.0, PRIOR["phi_a"], PRIOR["phi_b"]) - jnp.log(
        2.0
    )
    sigma = theta["sigma"]
    if spec.sigma_prior == "halfnormal":
        lp += jnp.log(2.0) + stats.norm.logpdf(sigma, 0.0, PRIOR["sigma_scale"])
    else:  # σ² ~ InvGamma(a, b)  →  p(σ) = p(σ²) · 2σ
        a, b = PRIOR["ig_shape"], PRIOR["ig_rate"]
        lp += stats.gamma.logpdf(1.0 / sigma**2, a, scale=1.0 / b) - 4.0 * jnp.log(sigma)
        lp += jnp.log(2.0 * sigma)
    if spec.leverage:
        lp += -jnp.log(2.0)  # ρ ~ Uniform(-1, 1)
    if spec.student_t:
        lp += stats.gamma.logpdf(theta["nu"] - 2.0, PRIOR["nu_shape"], scale=1.0 / PRIOR["nu_rate"])
    return lp


def obs_log_lik(y: jax.Array, h: jax.Array, theta: dict, spec: ModelSpec) -> jax.Array:
    """観測の対数尤度 log p(y | h, θ)。"""
    sd = jnp.exp(h / 2.0)
    if spec.student_t:
        nu = theta["nu"]
        scale = sd * jnp.sqrt((nu - 2.0) / nu)  # 分散 1 に基準化した t
        return jnp.sum(stats.t.logpdf(y, nu, 0.0, scale), axis=-1)
    if spec.leverage:
        # η_t = (h_{t+1} - μ - φ(h_t - μ))/σ を使って、y_t | h_t, h_{t+1} の条件付き正規分布
        mu, phi, sigma, rho = theta["mu"], theta["phi"], theta["sigma"], theta["rho"]
        eta = (h[..., 1:] - mu - phi * (h[..., :-1] - mu)) / sigma
        loc = rho * sd[..., :-1] * eta
        scale = jnp.sqrt(1.0 - rho**2) * sd[..., :-1]
        lp = jnp.sum(stats.norm.logpdf(y[:-1], loc, scale), axis=-1)
        return lp + stats.norm.logpdf(y[-1], 0.0, sd[..., -1])
    return jnp.sum(stats.norm.logpdf(y, 0.0, sd), axis=-1)


def latent_to_h(latent: jax.Array, theta: dict, spec: ModelSpec) -> jax.Array:
    if spec.centered:
        return latent
    return theta["mu"] + theta["sigma"] * latent


def log_joint(theta: dict, latent: jax.Array, y: jax.Array, spec: ModelSpec) -> jax.Array:
    """制約付き空間での log p(y, θ, latent)。"""
    if spec.centered:
        lp_latent = ar1_log_prob(latent, theta["mu"], theta["phi"], theta["sigma"])
    else:
        lp_latent = ar1_log_prob(latent, 0.0, theta["phi"], 1.0)
    h = latent_to_h(latent, theta, spec)
    return log_prior(theta, spec) + lp_latent + obs_log_lik(y, h, theta, spec)


# ---------------------------------------------------------------------------
# 無制約空間（ADVI はここで正規分布を当てる）
# ---------------------------------------------------------------------------


def constrain(u: jax.Array, spec: ModelSpec) -> tuple[dict, jax.Array]:
    """無制約ベクトル u（長さ n_global）→ θ と log|ヤコビアン|。"""
    theta = {"mu": u[0], "phi": jnp.tanh(u[1]), "sigma": jnp.exp(u[2])}
    log_jac = jnp.log1p(-(theta["phi"] ** 2)) + u[2]
    i = 3
    if spec.leverage:
        theta["rho"] = jnp.tanh(u[i])
        log_jac += jnp.log1p(-(theta["rho"] ** 2))
        i += 1
    if spec.student_t:
        theta["nu"] = 2.0 + jnp.exp(u[i])
        log_jac += u[i]
    return theta, log_jac


def unconstrain(theta: dict, spec: ModelSpec) -> jax.Array:
    u = [theta["mu"], jnp.arctanh(theta["phi"]), jnp.log(theta["sigma"])]
    if spec.leverage:
        u.append(jnp.arctanh(theta["rho"]))
    if spec.student_t:
        u.append(jnp.log(theta["nu"] - 2.0))
    return jnp.stack([jnp.asarray(v, dtype=float) for v in u])


def log_joint_unconstrained(z: jax.Array, y: jax.Array, spec: ModelSpec) -> jax.Array:
    """z = (u, latent) に対する log p(y, z)（変換のヤコビアン込み）。ADVI の目的関数の中身。"""
    k = spec.n_global
    theta, log_jac = constrain(z[:k], spec)
    return log_joint(theta, z[k:], y, spec) + log_jac
