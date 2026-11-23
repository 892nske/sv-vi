"""VI の当てはまり診断と収束判定。

- PSIS の Pareto k̂（Yao et al. 2018）：q を提案分布とした重要度重み w = p(y,z)/q(z) の裾の重さ。
  k̂ < 0.7 なら近似は概ね信頼できる、k̂ > 0.7 なら q は事後分布の重要な部分を取りこぼしている。
- ELBO の収束判定：ELBO 推定値の窓平均の相対変化が閾値を下回った最初のステップ
"""

import jax
import jax.numpy as jnp
import numpy as np
from numpyro import handlers
from numpyro.infer.util import log_density


def psis_khat(log_weights: np.ndarray) -> float:
    """対数重要度重みから Pareto k̂ を計算する（arviz-stats の実装を利用）。"""
    from arviz_stats.base import array_stats

    lw = np.asarray(log_weights, dtype=float)
    _, khat = array_stats.psislw(lw - lw.max())
    return float(khat)


def numpyro_log_weights(model, guide, params: dict, y, key, num_samples: int) -> np.ndarray:
    """NumPyro のガイドについて log p(y, z) - log q(z) を z ~ q で num_samples 個計算する。"""

    def one(k):
        seeded = handlers.substitute(handlers.seed(guide, k), data=params)
        tr = handlers.trace(seeded).get_trace(y)
        log_q = 0.0
        latents = {}
        for name, site in tr.items():
            if site["type"] != "sample" or site["is_observed"]:
                continue
            log_q = log_q + jnp.sum(site["fn"].log_prob(site["value"]))
            if not site["infer"].get("is_auxiliary", False):
                latents[name] = site["value"]
        log_p, _ = log_density(model, (y,), {}, latents)
        return log_p - log_q

    keys = jax.random.split(key, num_samples)
    return np.asarray(jax.lax.map(one, keys, batch_size=100))


def convergence_step(elbo_trace: np.ndarray, window: int = 200, rtol: float = 1e-4) -> int:
    """窓平均の相対変化が rtol 未満になった最初のステップ（収束しなければ最終ステップ）。"""
    trace = np.asarray(elbo_trace)
    n = len(trace) // window
    if n < 2:
        return len(trace)
    means = trace[: n * window].reshape(n, window).mean(axis=1)
    rel = np.abs(np.diff(means)) / np.abs(means[1:])
    hit = np.nonzero(rel < rtol)[0]
    return int((hit[0] + 2) * window) if len(hit) else len(trace)
