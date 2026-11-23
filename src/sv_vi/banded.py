"""三重対角の精度行列を持つガウス分布のための O(T) 演算。

精度行列を Q = L L^T と分解し、L を下二重対角とする:

    L[t, t]   = d[t]  (> 0),   t = 0..T-1
    L[t+1, t] = s[t],          t = 0..T-2

このとき x ~ N(m, Q^{-1}) は、ε ~ N(0, I) について L^T (x - m) = ε を解けば得られる。
L^T は上二重対角なので、末尾から順に解く後退代入（O(T)）で済む。密行列は一切作らない。
"""

import jax
import jax.numpy as jnp

LOG_2PI = jnp.log(2.0 * jnp.pi)


def lt_matvec(d: jax.Array, s: jax.Array, x: jax.Array) -> jax.Array:
    """L^T x を計算する。(L^T x)[t] = d[t] x[t] + s[t] x[t+1]。"""
    upper = jnp.concatenate([s * x[1:], jnp.zeros((1,), dtype=x.dtype)])
    return d * x + upper


def _lt_solve_1d(d: jax.Array, s: jax.Array, b: jax.Array) -> jax.Array:
    x_last = b[-1] / d[-1]

    def step(x_next, inputs):
        d_t, s_t, b_t = inputs
        x_t = (b_t - s_t * x_next) / d_t
        return x_t, x_t

    _, x_head = jax.lax.scan(step, x_last, (d[:-1], s, b[:-1]), reverse=True)
    return jnp.concatenate([x_head, x_last[None]])


def lt_solve(d: jax.Array, s: jax.Array, b: jax.Array) -> jax.Array:
    """L^T x = b を後退代入で解く（O(T)）。b は (..., T) でよい（先頭はサンプル数などのバッチ次元）。"""
    return jnp.vectorize(_lt_solve_1d, signature="(n),(k),(n)->(n)")(d, s, b)


def _l_solve_1d(d: jax.Array, s: jax.Array, b: jax.Array) -> jax.Array:
    x0 = b[0] / d[0]

    def step(x_prev, inputs):
        d_t, s_prev, b_t = inputs
        x_t = (b_t - s_prev * x_prev) / d_t
        return x_t, x_t

    _, x_tail = jax.lax.scan(step, x0, (d[1:], s, b[1:]))
    return jnp.concatenate([x0[None], x_tail])


def l_solve(d: jax.Array, s: jax.Array, b: jax.Array) -> jax.Array:
    """L x = b を前進代入で解く（O(T)）。"""
    return jnp.vectorize(_l_solve_1d, signature="(n),(k),(n)->(n)")(d, s, b)


def precision_solve(d: jax.Array, s: jax.Array, b: jax.Array) -> jax.Array:
    """Q x = b（Q = L L^T）を解く。前進代入と後退代入を 1 回ずつ（O(T)）。"""
    return lt_solve(d, s, l_solve(d, s, b))


def log_prob(m: jax.Array, d: jax.Array, s: jax.Array, x: jax.Array) -> jax.Array:
    """log N(x | m, (L L^T)^{-1})。x は (..., T)。"""
    z = jnp.vectorize(lt_matvec, signature="(n),(k),(n)->(n)")(d, s, x - m)
    T = m.shape[-1]
    return -0.5 * T * LOG_2PI + jnp.sum(jnp.log(d)) - 0.5 * jnp.sum(z**2, axis=-1)


def entropy(d: jax.Array) -> jax.Array:
    """エントロピー。log det(Q^{-1}) = -2 Σ log d なので対角の和だけで済む。"""
    T = d.shape[-1]
    return 0.5 * T * (1.0 + LOG_2PI) - jnp.sum(jnp.log(d), axis=-1)


def sample(key: jax.Array, m: jax.Array, d: jax.Array, s: jax.Array, shape=()) -> jax.Array:
    """再パラメータ化サンプル x = m + L^{-T} ε。"""
    eps = jax.random.normal(key, tuple(shape) + m.shape, dtype=m.dtype)
    return m + lt_solve(d, s, eps)


def tridiag_cholesky(a: jax.Array, b: jax.Array) -> tuple[jax.Array, jax.Array]:
    """三重対角の正定値行列 Q（対角 a、副対角 b）のコレスキー分解 Q = L L^T。

    L は下二重対角になり、その (対角 d, 下側の対角 s) を返す。O(T)。変分パラメータの初期化に使う。
    """

    def step(d_prev, inputs):
        a_t, b_prev = inputs
        s_prev = b_prev / d_prev
        d_t = jnp.sqrt(a_t - s_prev**2)
        return d_t, (d_t, s_prev)

    d0 = jnp.sqrt(a[0])
    _, (d_rest, s) = jax.lax.scan(step, d0, (a[1:], b))
    return jnp.concatenate([d0[None], d_rest]), s


def ar1_precision(T: int, phi, sigma, sigma_innov=None) -> tuple[jax.Array, jax.Array]:
    """AR(1) 過程の精度行列（三重対角）の (対角, 副対角)。

    h_1 ~ N(μ, σ²/(1-φ²))（定常分布）、h_{t+1} | h_t のイノベーション sd は sigma_innov（省略時は σ）。
    sigma_innov = σ なら定常 AR(1) の事前分布そのもの。
    """
    s2 = sigma**2 if sigma_innov is None else sigma_innov**2
    v0 = sigma**2 / (1.0 - phi**2)
    inner = jnp.full((T,), (1.0 + phi**2) / s2)
    a = inner.at[0].set(1.0 / v0 + phi**2 / s2).at[-1].set(1.0 / s2)
    b = jnp.full((T - 1,), -phi / s2)
    return a, b


def dense_precision(d: jax.Array, s: jax.Array) -> jax.Array:
    """テスト用：精度行列 Q = L L^T を密行列で作る（T が小さいときだけ使う）。"""
    L = jnp.diag(d) + jnp.diag(s, k=-1)
    return L @ L.T
