import jax
import numpy as np

from sv_vi.simulate import SVParams, simulate_sv


def test_stationary_variance():
    p = SVParams(mu=0.3, phi=0.9, sigma=0.3)
    h = simulate_sv(jax.random.PRNGKey(0), 200_000, p)["h"]
    assert abs(h.mean() - 0.3) < 0.02
    assert abs(h.var() / (0.3**2 / (1 - 0.9**2)) - 1) < 0.05


def test_leverage_correlation():
    p = SVParams(mu=0.0, phi=0.9, sigma=0.3, rho=-0.5)
    d = simulate_sv(jax.random.PRNGKey(1), 200_000, p)
    eps = d["y"] / np.exp(d["h"] / 2)
    eta = (d["h"][1:] - 0.9 * d["h"][:-1]) / 0.3
    assert abs(np.corrcoef(eps[:-1], eta)[0, 1] + 0.5) < 0.01


def test_student_t_unit_variance():
    p = SVParams(mu=0.0, phi=0.9, sigma=0.3, nu=8.0)
    d = simulate_sv(jax.random.PRNGKey(2), 400_000, p)
    eps = d["y"] / np.exp(d["h"] / 2)
    assert abs(eps.var() - 1) < 0.03
