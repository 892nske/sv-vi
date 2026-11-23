import jax.numpy as jnp
import numpy as np
from scipy.stats import multivariate_normal

from sv_vi import banded
from sv_vi.distributions import AR1


def test_ar1_log_prob_matches_dense_mvn():
    T, mu, phi, sigma = 6, 0.2, 0.9, 0.3
    a, b = banded.ar1_precision(T, phi, sigma)
    Q = np.diag(a) + np.diag(b, 1) + np.diag(b, -1)
    x = np.linspace(-1.0, 1.5, T)
    expected = multivariate_normal(np.full(T, mu), np.linalg.inv(Q)).logpdf(x)
    np.testing.assert_allclose(AR1(mu, phi, sigma, T).log_prob(jnp.asarray(x)), expected)
