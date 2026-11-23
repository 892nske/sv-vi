import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sv_vi import advi
from sv_vi.inference import run_advi, run_nuts, run_svi
from sv_vi.initialization import initial_values
from sv_vi.logdensity import ModelSpec
from sv_vi.simulate import SCENARIOS, simulate_sv


@pytest.fixture(scope="module")
def data():
    return simulate_sv(jax.random.PRNGKey(0), 300, SCENARIOS["standard"])


@pytest.mark.parametrize("family", advi.FAMILIES)
@pytest.mark.parametrize(
    "spec", [ModelSpec(), ModelSpec(centered=False), ModelSpec(leverage=True)], ids=str
)
def test_entropy_matches_log_q(data, family, spec):
    """draw が返すエントロピーと、サンプルでの -log q の平均が一致する（log_q と draw の整合性）。"""
    init = initial_values(data["y"], spec)
    params = advi.init_params(init, spec, family, data["y"])
    params["u_tril" if family != "meanfield" else "log_scale"] += 0.05  # 初期値から少しずらす
    z, ent = advi.draw(params, family, jax.random.PRNGKey(1), 4000, spec)
    neg_lq = -advi.log_q(params, family, z, spec)
    np.testing.assert_allclose(neg_lq.mean(), ent, atol=4 * neg_lq.std() / np.sqrt(4000) + 0.05)


def test_elbo_increases(data):
    spec = ModelSpec()
    cfg = advi.ADVIConfig(family="conditional", num_steps=2000)
    r = run_advi(data["y"], spec, jax.random.PRNGKey(2), cfg, khat=False)
    assert r.elbo_trace[-200:].mean() > r.elbo_trace[:200].mean()


def test_conditional_is_parameterization_invariant(data):
    cfg = advi.ADVIConfig(family="conditional", num_steps=3000)
    a = run_advi(data["y"], ModelSpec(centered=True), jax.random.PRNGKey(3), cfg, khat=False)
    b = run_advi(data["y"], ModelSpec(centered=False), jax.random.PRNGKey(3), cfg, khat=False)
    np.testing.assert_allclose(a.h_mean, b.h_mean, atol=1e-6)


def test_self_structured_matches_numpyro(data):
    """同じ変分族（大域フルランク × 三重対角精度）なら、自前と NumPyro で ELBO・事後平均が近い。"""
    spec = ModelSpec()
    cfg = advi.ADVIConfig(family="structured", num_steps=8000)
    a = run_advi(data["y"], spec, jax.random.PRNGKey(4), cfg, khat=False)
    b = run_svi(data["y"], spec, "structured", jax.random.PRNGKey(4), num_steps=8000, khat=False)
    assert abs(a.elbo_trace[-500:].mean() - b.elbo_trace[-500:].mean()) < 3.0
    np.testing.assert_allclose(a.h_mean, b.h_mean, atol=0.1)
    for k in spec.global_names:
        assert abs(a.theta[k].mean() - b.theta[k].mean()) < 0.05


def test_conditional_close_to_nuts(data):
    spec = ModelSpec()
    cfg = advi.ADVIConfig(family="conditional", num_steps=5000)
    a = run_advi(data["y"], spec, jax.random.PRNGKey(5), cfg, khat=False)
    n = run_nuts(data["y"], spec, jax.random.PRNGKey(6), num_warmup=1000, num_samples=1000)
    np.testing.assert_allclose(a.h_mean, n.h_mean, atol=0.15)
    np.testing.assert_allclose(np.median(a.h_sd / n.h_sd), 1.0, atol=0.15)
    assert abs(a.theta["phi"].mean() - n.theta["phi"].mean()) < 2 * n.theta["phi"].std()
    assert abs(a.theta["sigma"].mean() - n.theta["sigma"].mean()) < 2 * n.theta["sigma"].std()


def test_conditional_mean_without_drift_is_prior_mean():
    """擬似観測の精度 λ→0 なら、q(h | θ) の平均は事前平均 μ（ドリフトなし）に一致する。"""
    spec = ModelSpec()
    T = 50
    params = {
        "pseudo_obs": jnp.zeros(T),
        "log_pseudo_prec": jnp.full(T, -30.0),
    }
    u = jnp.array([0.7, 1.2, -1.5])
    _, m, _, _ = advi.conditional_factor(params, u, spec)
    np.testing.assert_allclose(m, 0.7, atol=1e-8)
