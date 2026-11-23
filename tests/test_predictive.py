import numpy as np
import pytest
from scipy import stats

from sv_vi.logdensity import ModelSpec
from sv_vi.predictive import forecast, next_h, var_quantile


class _Post:
    def __init__(self, theta, h_last):
        self.theta, self.h_last = theta, h_last


def test_degenerate_case_is_normal():
    """σ→0, φ=0 なら h_{T+1} = μ なので、予測分布は N(0, e^μ)。"""
    S = 500
    theta = {"mu": np.full(S, 0.4), "phi": np.zeros(S), "sigma": np.full(S, 1e-8)}
    post = _Post(theta, np.zeros(S))
    out = forecast(post, y_last=0.0, y_obs=0.3, spec=ModelSpec(), seed=0)
    sd = np.exp(0.2)
    assert out["var5"] == pytest.approx(stats.norm.ppf(0.05) * sd, rel=1e-5)
    assert out["log_score"] == pytest.approx(stats.norm.logpdf(0.3, 0, sd), rel=1e-6)
    assert out["pred_sd"] == pytest.approx(sd, rel=1e-6)


def test_student_t_quantile_is_heavier():
    h = np.zeros(10)
    th_n = {"mu": h, "phi": h, "sigma": h}
    th_t = th_n | {"nu": np.full(10, 5.0)}
    q_n = var_quantile(th_n, h, ModelSpec(), 0.01)
    q_t = var_quantile(th_t, h, ModelSpec(student_t=True), 0.01)
    assert q_t < q_n  # 分散が同じでも t の方が 1% 点は深い


def test_leverage_negative_return_raises_vol():
    S = 2000
    theta = {
        k: np.full(S, v) for k, v in {"mu": 0.0, "phi": 0.95, "sigma": 0.2, "rho": -0.6}.items()
    }
    rng = np.random.default_rng(0)
    down = next_h(theta, np.zeros(S), -3.0, ModelSpec(leverage=True), rng).mean()
    up = next_h(theta, np.zeros(S), 3.0, ModelSpec(leverage=True), rng).mean()
    assert down > up
