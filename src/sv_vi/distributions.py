"""NumPyro 用のカスタム分布。

- AR1                   : 定常 AR(1) 過程の事前分布（O(T) の log_prob）。中心化モデルの h に使う
- BandedPrecisionNormal : 精度行列が三重対角のガウス分布。構造化変分族 q(h) に使う
"""

import jax
import jax.numpy as jnp
from numpyro.distributions import Distribution, constraints

from sv_vi import banded
from sv_vi.logdensity import ar1_log_prob


class AR1(Distribution):
    """v_1 ~ N(μ, σ²/(1-φ²)),  v_t = μ + φ(v_{t-1} - μ) + σ η_t。パラメータはスカラーのみ対応。"""

    arg_constraints = {
        "mu": constraints.real,
        "phi": constraints.interval(-1.0, 1.0),
        "sigma": constraints.positive,
    }
    support = constraints.real_vector
    reparametrized_params = ["mu", "phi", "sigma"]

    def __init__(self, mu, phi, sigma, T: int, *, validate_args=None):
        self.mu, self.phi, self.sigma, self.T = mu, phi, sigma, T
        super().__init__(batch_shape=(), event_shape=(T,), validate_args=validate_args)

    def sample(self, key, sample_shape=()):
        eps = jax.random.normal(key, tuple(sample_shape) + (self.T,))
        eps = jnp.moveaxis(eps, -1, 0)  # 時間軸を先頭へ（scan 用）
        v1 = self.mu + self.sigma / jnp.sqrt(1.0 - self.phi**2) * eps[0]

        def step(v_prev, e):
            v = self.mu + self.phi * (v_prev - self.mu) + self.sigma * e
            return v, v

        _, rest = jax.lax.scan(step, v1, eps[1:])
        return jnp.moveaxis(jnp.concatenate([v1[None], rest]), 0, -1)

    def log_prob(self, value):
        return ar1_log_prob(value, self.mu, self.phi, self.sigma)


class BandedPrecisionNormal(Distribution):
    """N(loc, (L L^T)^{-1})、L は下二重対角（対角 diag > 0、下側の対角 off）。"""

    arg_constraints = {
        "loc": constraints.real_vector,
        "diag": constraints.independent(constraints.positive, 1),
        "off": constraints.real_vector,
    }
    support = constraints.real_vector
    reparametrized_params = ["loc", "diag", "off"]

    def __init__(self, loc, diag, off, *, validate_args=None):
        self.loc, self.diag, self.off = loc, diag, off
        super().__init__(batch_shape=(), event_shape=loc.shape[-1:], validate_args=validate_args)

    def sample(self, key, sample_shape=()):
        return banded.sample(key, self.loc, self.diag, self.off, sample_shape)

    def log_prob(self, value):
        return banded.log_prob(self.loc, self.diag, self.off, value)

    def entropy(self):
        return banded.entropy(self.diag)
