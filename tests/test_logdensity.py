from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpyro.infer.util import log_density

from sv_vi.logdensity import ModelSpec, constrain, log_joint, log_joint_unconstrained, unconstrain
from sv_vi.models import sv_model
from sv_vi.simulate import SCENARIOS, simulate_sv

SPECS = [
    ModelSpec(),
    ModelSpec(centered=False),
    ModelSpec(sigma_prior="invgamma"),
    ModelSpec(leverage=True),
    ModelSpec(student_t=True, centered=False),
]


@pytest.mark.parametrize("spec", SPECS)
def test_matches_numpyro(spec):
    T = 50
    y = jnp.asarray(simulate_sv(jax.random.PRNGKey(0), T, SCENARIOS["standard"])["y"])
    theta = {"mu": 0.1, "phi": 0.95, "sigma": 0.2}
    if spec.leverage:
        theta["rho"] = -0.3
    if spec.student_t:
        theta["nu"] = 9.0
    theta = {k: jnp.asarray(v) for k, v in theta.items()}
    latent = 0.3 * jnp.sin(jnp.arange(T) / 5.0)
    expected, _ = log_density(
        partial(sv_model, spec=spec), (y,), {}, theta | {spec.latent_name: latent}
    )
    np.testing.assert_allclose(log_joint(theta, latent, y, spec), expected, rtol=1e-10)


def test_unconstrain_roundtrip():
    spec = ModelSpec(leverage=True)
    theta = {"mu": 0.1, "phi": 0.95, "sigma": 0.2, "rho": -0.4}
    back, _ = constrain(unconstrain(theta, spec), spec)
    for k, v in theta.items():
        np.testing.assert_allclose(back[k], v)


def test_unconstrained_jacobian():
    """σ 方向だけ動かして数値積分：u_σ で積分した値と σ で積分した値が一致する（ヤコビアンの確認）。

    φ は固定しているので、φ のヤコビアン（定数）は σ 側にも足しておく。
    """
    spec = ModelSpec()
    y = jnp.array([0.5, -1.0, 0.8])
    latent = jnp.array([0.3, -0.2, 0.5])

    def lp_u(u_sigma):
        z = jnp.concatenate([jnp.array([0.0, 1.5, u_sigma]), latent])
        return log_joint_unconstrained(z, y, spec)

    def lp_sigma(sigma):
        th = {"mu": 0.0, "phi": jnp.tanh(1.5), "sigma": sigma}
        return log_joint(th, latent, y, spec) + jnp.log1p(-(jnp.tanh(1.5) ** 2))

    us = jnp.linspace(-8, 4, 20001)
    sig = jnp.linspace(1e-6, 60, 200001)
    i_u = jnp.trapezoid(jnp.exp(jax.vmap(lp_u)(us)), us)
    i_s = jnp.trapezoid(jnp.exp(jax.vmap(lp_sigma)(sig)), sig)
    np.testing.assert_allclose(i_u, i_s, rtol=1e-3)
