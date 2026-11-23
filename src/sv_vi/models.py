"""NumPyro による SV モデル（基本・レバレッジ・t 誤差、中心化／非中心化）。

確率モデルは logdensity.py と同一（tests/test_logdensity.py で一致を確認）。
"""

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.distributions import constraints, transforms

from sv_vi.distributions import AR1
from sv_vi.logdensity import PRIOR, ModelSpec


def _phi_prior():
    # (φ+1)/2 ~ Beta(20, 1.5)
    return dist.TransformedDistribution(
        dist.Beta(PRIOR["phi_a"], PRIOR["phi_b"]),
        transforms.AffineTransform(-1.0, 2.0, domain=constraints.unit_interval),
    )


def _sigma_prior(kind: str):
    if kind == "halfnormal":
        return dist.HalfNormal(PRIOR["sigma_scale"])
    # σ² ~ InvGamma(2.5, 0.025)  →  σ = (σ²)^{1/2}
    return dist.TransformedDistribution(
        dist.InverseGamma(PRIOR["ig_shape"], PRIOR["ig_rate"]), transforms.PowerTransform(0.5)
    )


def sv_model(y, spec: ModelSpec = ModelSpec()):
    T = y.shape[0]
    mu = numpyro.sample("mu", dist.Normal(PRIOR["mu_loc"], PRIOR["mu_scale"]))
    phi = numpyro.sample("phi", _phi_prior())
    sigma = numpyro.sample("sigma", _sigma_prior(spec.sigma_prior))

    if spec.centered:
        h = numpyro.sample("h", AR1(mu, phi, sigma, T))
    else:
        x = numpyro.sample("x", AR1(0.0, phi, 1.0, T))
        h = numpyro.deterministic("h", mu + sigma * x)

    sd = jnp.exp(h / 2.0)
    if spec.student_t:
        nu = numpyro.sample(
            "nu",
            dist.TransformedDistribution(
                dist.Gamma(PRIOR["nu_shape"], PRIOR["nu_rate"]),
                transforms.AffineTransform(2.0, 1.0, domain=constraints.positive),
            ),
        )
        numpyro.sample("y", dist.StudentT(nu, 0.0, sd * jnp.sqrt((nu - 2.0) / nu)), obs=y)
    elif spec.leverage:
        rho = numpyro.sample("rho", dist.Uniform(-1.0, 1.0))
        eta = (h[1:] - mu - phi * (h[:-1] - mu)) / sigma
        loc = jnp.concatenate([rho * sd[:-1] * eta, jnp.zeros(1)])
        scale = jnp.concatenate([jnp.sqrt(1.0 - rho**2) * sd[:-1], sd[-1:]])
        numpyro.sample("y", dist.Normal(loc, scale).to_event(1), obs=y)
    else:
        numpyro.sample("y", dist.Normal(0.0, sd).to_event(1), obs=y)
