"""自前実装の ADVI（自動微分変分推論）。論文付録 B に抜粋を載せる。

z = (u, v) とする。u は大域パラメータ（無制約空間, 長さ k）、v は潜在変数 h または x（長さ T）。
log p(y, z) は logdensity.log_joint_unconstrained（変換のヤコビアン込み）。

変分族
- "meanfield"   : q(z) = Π_i N(z_i | m_i, s_i^2)                          パラメータ数 2(k+T)
- "structured"  : q(z) = N(u | m_u, C C^T) × N(v | m_v, (L L^T)^{-1})
                  C は k×k の下三角（フルランク）、L は下二重対角            パラメータ数 ≈ 3T
- "conditional" : q(z) = N(u | m_u, C C^T) × q(h | θ(u))
                  q(h | θ) = AR(1) 事前 p(h | θ) × Π_t N(ỹ_t | h_t, 1/λ_t) を正規化したもの
                  （擬似観測 ỹ_t と精度 λ_t が変分パラメータ）。          パラメータ数 ≈ 2T
                  精度行列 Q(θ) = Q_AR1(φ, σ) + diag(λ) は三重対角なので O(T) で扱える。
                  θ を固定したときの真の事後分布 p(h | y, θ) のガウス近似と同じ形をしており、
                  σ が小さければ h は滑らかに、μ が上がれば h 全体が上がる、という依存関係を自然に表現する。
                  レバレッジ SV では、リターンを条件にすると h のイノベーション分散が σ²(1-ρ²) に縮むので、
                  q(h | θ) の遷移部分のイノベーション sd を σ√(1-ρ²) にする。

ELBO(λ) = E_q[log p(y, z)] + H[q] を、再パラメータ化 z = T_λ(ε), ε ~ N(0, I) による
モンテカルロ推定で Adam により最大化する。最適化ループ全体を lax.scan で書き、jit 1 回で回す。
1 系列分の純関数なので、多銘柄では jax.vmap でそのままバッチ化できる。
"""

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import optax

from sv_vi import banded
from sv_vi.initialization import initial_latent_factor
from sv_vi.logdensity import ModelSpec, constrain, log_joint_unconstrained, unconstrain

LOG_2PI = jnp.log(2.0 * jnp.pi)
FAMILIES = ("meanfield", "structured", "conditional")


@dataclass(frozen=True)
class ADVIConfig:
    family: str = "conditional"
    num_steps: int = 10_000
    lr: float = 0.01
    lr_final_ratio: float = 0.1  # 学習率を最後に lr × この値まで指数的に下げる
    num_mc: int = 4  # ELBO 1 回あたりのモンテカルロサンプル数


# ---------------------------------------------------------------------------
# 変分パラメータの初期化
# ---------------------------------------------------------------------------


def init_params(init: dict, spec: ModelSpec, family: str, y: jax.Array | None = None) -> dict:
    u0 = unconstrain(init, spec)
    v0 = init[spec.latent_name]
    k = spec.n_global
    u_part = {"u_loc": u0, "u_tril": jnp.diag(jnp.full(k, jnp.log(0.1)))}  # 対角は log で持つ
    if family == "meanfield":
        d0, s0 = initial_latent_factor(init, spec)
        # 潜在変数の初期 sd は 1/sqrt(Q_tt)（ガウス目標に対する平均場の最適解）。大域パラメータは 0.1
        # Q = L L^T の対角は Q_tt = d_t^2 + s_{t-1}^2
        q_diag = d0**2 + jnp.concatenate([jnp.zeros(1), s0**2])
        return {
            "loc": jnp.concatenate([u0, v0]),
            "log_scale": jnp.concatenate([jnp.full(k, jnp.log(0.1)), -0.5 * jnp.log(q_diag)]),
        }
    if family == "structured":
        d0, s0 = initial_latent_factor(init, spec)
        return u_part | {"v_loc": v0, "v_log_diag": jnp.log(d0), "v_off": s0}
    if family == "conditional":
        h0 = init["h"] if spec.centered else init["mu"] + init["sigma"] * init["x"]
        # 擬似観測の精度の初期値 1/2 は、観測 1 つあたりの尤度の曲率（≈ 1/2）に合わせたもの
        params = u_part | {"pseudo_obs": h0, "log_pseudo_prec": jnp.full(h0.shape, jnp.log(0.5))}
        if spec.leverage:
            # ẽ_t の初期値は基準化リターン y_t / exp(h_t/2)（0 で始めると ρ と ẽ の勾配が両方 0 になる）
            params["shock"] = jnp.asarray(y)[:-1] * jnp.exp(-h0[:-1] / 2.0)
        return params
    raise ValueError(family)


def _tril(raw: jax.Array) -> jax.Array:
    """下三角行列。対角は exp で正にする。"""
    return jnp.tril(raw, -1) + jnp.diag(jnp.exp(jnp.diag(raw)))


def conditional_factor(params: dict, u: jax.Array, spec: ModelSpec):
    """q(h | θ(u)) の平均 m と精度のコレスキー因子 (d, s) を返す。すべて O(T)。

    q(h | θ) ∝ N(h_1 | μ, σ²/(1-φ²)) Π_t N(h_{t+1} | μ + φ(h_t - μ) + δ_t, s²) Π_t N(ỹ_t | h_t, 1/λ_t)
      基本・t 誤差 : δ_t = 0,          s = σ             （AR(1) 事前そのもの）
      レバレッジ   : δ_t = ρ σ ẽ_t,    s = σ √(1-ρ²)      （リターンを条件にした遷移。ẽ_t も変分パラメータ）
    """
    theta, _ = constrain(u, spec)
    mu, phi, sigma = theta["mu"], theta["phi"], theta["sigma"]
    T = params["pseudo_obs"].shape[0]
    if spec.leverage:
        rho = theta["rho"]
        s2 = sigma**2 * (1.0 - rho**2)
        drift = rho * sigma * params["shock"]
    else:
        s2 = sigma**2
        drift = jnp.zeros(T - 1)
    a, b = banded.ar1_precision(T, phi, sigma, jnp.sqrt(s2))
    lam = jnp.exp(params["log_pseudo_prec"])
    d, s = banded.tridiag_cholesky(a + lam, b)
    # 精度 Q = Q_AR + diag(λ) のガウス分布の平均 m は Q m = c + λ ỹ を解いて得る。
    # c は遷移・初期分布の 2 次形式の 1 次の項：e_t = μ(1-φ) + δ_t（h_{t+1} - φ h_t の平均）とすると
    #   c_1 = μ/v0 - φ e_1/s²,  c_t = (e_{t-1} - φ e_t)/s²,  c_T = e_{T-1}/s²
    e = mu * (1.0 - phi) + drift
    v0 = sigma**2 / (1.0 - phi**2)
    c = (jnp.concatenate([jnp.zeros(1), e]) - phi * jnp.concatenate([e, jnp.zeros(1)])) / s2
    c = c.at[0].add(mu / v0)
    m = banded.precision_solve(d, s, c + lam * params["pseudo_obs"])
    return theta, m, d, s


# ---------------------------------------------------------------------------
# 再パラメータ化サンプリング（サンプルと、そのサンプルでのエントロピー推定値を返す）
# ---------------------------------------------------------------------------


def draw(params: dict, family: str, key: jax.Array, n: int, spec: ModelSpec):
    """q から n 個のサンプル z（(n, k+T)）と、エントロピー H[q] の推定値を返す。"""
    if family == "meanfield":
        eps = jax.random.normal(key, (n,) + params["loc"].shape)
        z = params["loc"] + jnp.exp(params["log_scale"]) * eps
        D = params["loc"].shape[0]
        return z, 0.5 * D * (1.0 + LOG_2PI) + jnp.sum(params["log_scale"])

    k_u, k_v = jax.random.split(key)
    k = params["u_loc"].shape[0]
    C = _tril(params["u_tril"])
    u = params["u_loc"] + jax.random.normal(k_u, (n, k)) @ C.T
    ent_u = 0.5 * k * (1.0 + LOG_2PI) + jnp.sum(jnp.diag(params["u_tril"]))

    if family == "structured":
        d = jnp.exp(params["v_log_diag"])
        v = banded.sample(k_v, params["v_loc"], d, params["v_off"], shape=(n,))
        return jnp.concatenate([u, v], axis=-1), ent_u + banded.entropy(d)

    # conditional：u ごとに q(h | θ(u)) を組み立ててサンプルする。H[q] = H[q(u)] + E_u H[q(h|u)]
    T = params["pseudo_obs"].shape[0]
    eps_v = jax.random.normal(k_v, (n, T))

    def one(u_i, eps_i):
        theta, m, d, s = conditional_factor(params, u_i, spec)
        h = m + banded.lt_solve(d, s, eps_i)
        ent_h = banded.entropy(d)
        if spec.centered:
            return h, ent_h
        # x = (h - μ)/σ の線形変換：エントロピーは T log σ だけ減る
        return (h - theta["mu"]) / theta["sigma"], ent_h - T * jnp.log(theta["sigma"])

    v, ent_h = jax.vmap(one)(u, eps_v)
    return jnp.concatenate([u, v], axis=-1), ent_u + jnp.mean(ent_h)


def log_q(params: dict, family: str, z: jax.Array, spec: ModelSpec) -> jax.Array:
    """log q(z)。PSIS 診断で使う。z は (n, k+T)。"""
    if family == "meanfield":
        scale = jnp.exp(params["log_scale"])
        return jnp.sum(jax.scipy.stats.norm.logpdf(z, params["loc"], scale), axis=-1)
    k = params["u_loc"].shape[0]
    C = _tril(params["u_tril"])
    u, v = z[:, :k], z[:, k:]
    lq_u = jax.scipy.stats.multivariate_normal.logpdf(u, params["u_loc"], C @ C.T)
    if family == "structured":
        d = jnp.exp(params["v_log_diag"])
        return lq_u + banded.log_prob(params["v_loc"], d, params["v_off"], v)

    def one(u_i, v_i):
        theta, m, d, s = conditional_factor(params, u_i, spec)
        if spec.centered:
            return banded.log_prob(m, d, s, v_i)
        h = theta["mu"] + theta["sigma"] * v_i
        return banded.log_prob(m, d, s, h) + v_i.shape[0] * jnp.log(theta["sigma"])

    return lq_u + jax.vmap(one)(u, v)


# ---------------------------------------------------------------------------
# ELBO と最適化
# ---------------------------------------------------------------------------


def elbo(params, key, y, spec: ModelSpec, family: str, num_mc: int) -> jax.Array:
    z, ent = draw(params, family, key, num_mc, spec)
    log_p = jax.vmap(lambda zi: log_joint_unconstrained(zi, y, spec))(z)
    return jnp.mean(log_p) + ent


@partial(jax.jit, static_argnames=("spec", "config"))
def fit(params0: dict, y: jax.Array, key: jax.Array, spec: ModelSpec, config: ADVIConfig):
    """ADVI を実行し、(最終の変分パラメータ, 各ステップの ELBO 推定値) を返す。"""
    schedule = optax.exponential_decay(
        config.lr, transition_steps=config.num_steps, decay_rate=config.lr_final_ratio
    )
    optimizer = optax.adam(schedule)
    loss_and_grad = jax.value_and_grad(
        lambda p, k: -elbo(p, k, y, spec, config.family, config.num_mc)
    )

    def step(carry, key_t):
        params, opt_state = carry
        loss, grads = loss_and_grad(params, key_t)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return (optax.apply_updates(params, updates), opt_state), -loss

    keys = jax.random.split(key, config.num_steps)
    (params, _), elbo_trace = jax.lax.scan(step, (params0, optimizer.init(params0)), keys)
    return params, elbo_trace


# ---------------------------------------------------------------------------
# warm start：推定窓を 1 日進めたときの変分パラメータの引き継ぎ
# ---------------------------------------------------------------------------


def _shift(v: jax.Array, fill=None) -> jax.Array:
    """先頭を捨てて末尾に 1 つ足す（既定では最後の値を複製）。"""
    last = v[-1:] if fill is None else jnp.full((1,) + v.shape[1:], fill, dtype=v.dtype)
    return jnp.concatenate([v[1:], last])


def shift_params(params: dict, family: str, spec: ModelSpec) -> dict:
    """長さ T の窓を 1 日進めたときの初期値。大域パラメータはそのまま、時点ごとのパラメータは 1 つずらす。

    新しい日の値は前日の値を複製する（h は持続性が高いので良い初期値になる）。
    レバレッジのドリフト ẽ は新しい日の分を 0 から始める。
    """
    p = dict(params)
    if family == "meanfield":
        k = spec.n_global
        for name in ("loc", "log_scale"):
            p[name] = jnp.concatenate([params[name][:k], _shift(params[name][k:])])
        return p
    for name in ("v_loc", "v_log_diag", "v_off", "pseudo_obs", "log_pseudo_prec"):
        if name in p:
            p[name] = _shift(params[name])
    if "shock" in p:
        p["shock"] = _shift(params["shock"], fill=0.0)
    return p
