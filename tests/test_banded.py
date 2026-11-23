import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import multivariate_normal

from sv_vi import banded

T = 7


def _factor(seed=0):
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(seed), 3)
    d = jnp.exp(0.3 * jax.random.normal(k1, (T,)))
    s = 0.5 * jax.random.normal(k2, (T - 1,))
    m = jax.random.normal(k3, (T,))
    return m, d, s


def test_solves_match_dense():
    m, d, s = _factor()
    L = np.diag(d) + np.diag(s, -1)
    b = np.arange(T, dtype=float)
    np.testing.assert_allclose(banded.lt_solve(d, s, b), np.linalg.solve(L.T, b), rtol=1e-10)
    np.testing.assert_allclose(banded.l_solve(d, s, b), np.linalg.solve(L, b), rtol=1e-10)
    np.testing.assert_allclose(
        banded.precision_solve(d, s, b), np.linalg.solve(L @ L.T, b), rtol=1e-10
    )


def test_log_prob_and_entropy_match_dense():
    m, d, s = _factor(1)
    cov = np.linalg.inv(np.asarray(banded.dense_precision(d, s)))
    x = np.linspace(-1, 1, T)
    mvn = multivariate_normal(np.asarray(m), cov)
    np.testing.assert_allclose(banded.log_prob(m, d, s, x), mvn.logpdf(x), rtol=1e-10)
    np.testing.assert_allclose(banded.entropy(d), mvn.entropy(), rtol=1e-10)


def test_sample_covariance():
    m, d, s = _factor(2)
    x = banded.sample(jax.random.PRNGKey(3), m, d, s, shape=(200_000,))
    cov = np.linalg.inv(np.asarray(banded.dense_precision(d, s)))
    np.testing.assert_allclose(np.cov(np.asarray(x).T), cov, atol=0.03 * np.abs(cov).max())
    np.testing.assert_allclose(x.mean(0), m, atol=0.02)


def test_tridiag_cholesky():
    a, b = banded.ar1_precision(T, 0.9, 0.3)
    a = a + 0.5
    d, s = banded.tridiag_cholesky(a, b)
    Q = np.diag(a) + np.diag(b, 1) + np.diag(b, -1)
    np.testing.assert_allclose(banded.dense_precision(d, s), Q, rtol=1e-10)
