"""推論の実行ラッパ。NUTS・NumPyro SVI・自前 ADVI を共通の形式（PosteriorSummary）で返す。"""

import time
from dataclasses import dataclass, field
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import optax
from numpyro.diagnostics import effective_sample_size, split_gelman_rubin
from numpyro.infer import MCMC, NUTS, SVI, Predictive, Trace_ELBO
from numpyro.infer.initialization import init_to_value
from numpyro.infer.util import unconstrain_fn

from sv_vi import advi
from sv_vi.diagnostics import numpyro_log_weights, psis_khat
from sv_vi.guides import make_guide
from sv_vi.initialization import initial_values
from sv_vi.logdensity import ModelSpec, constrain, latent_to_h, log_joint_unconstrained
from sv_vi.models import sv_model


@dataclass
class PosteriorSummary:
    """手法によらない事後分布の要約。h はメモリ節約のため要約統計だけを持つ。"""

    method: str
    theta: dict[str, np.ndarray]  # 大域パラメータのサンプル（各 (S,)）
    h_mean: np.ndarray
    h_sd: np.ndarray
    h_q025: np.ndarray
    h_q975: np.ndarray
    time_total: float  # 実行時間 [秒]（コンパイル込み）
    time_compile: float = float("nan")  # そのうち JIT コンパイルの時間（推定値）
    elbo_trace: np.ndarray | None = None
    extra: dict = field(default_factory=dict)

    @property
    def time_run(self) -> float:
        return self.time_total - (0.0 if np.isnan(self.time_compile) else self.time_compile)

    def theta_summary(self) -> dict[str, dict[str, float]]:
        out = {}
        for k, v in self.theta.items():
            q = np.quantile(v, [0.025, 0.975])
            out[k] = {"mean": float(v.mean()), "sd": float(v.std()), "q025": q[0], "q975": q[1]}
        return out


def _summarize(method, theta_samples, h_samples, **kw) -> PosteriorSummary:
    h = np.asarray(h_samples)
    q = np.quantile(h, [0.025, 0.975], axis=0)
    return PosteriorSummary(
        method=method,
        theta={k: np.asarray(v).ravel() for k, v in theta_samples.items()},
        h_mean=h.mean(0),
        h_sd=h.std(0),
        h_q025=q[0],
        h_q975=q[1],
        **kw,
    )


# ---------------------------------------------------------------------------
# NUTS（ゴールドスタンダード）
# ---------------------------------------------------------------------------


def run_nuts(
    y,
    spec: ModelSpec,
    key,
    num_warmup: int = 1000,
    num_samples: int = 1000,
    num_chains: int = 4,
    target_accept: float = 0.9,
    init_jitter: float = 0.1,
    thinning: int = 1,
) -> PosteriorSummary:
    y = jnp.asarray(y)
    model = partial(sv_model, spec=spec)
    init = initial_values(y, spec)
    kernel = NUTS(model, init_strategy=init_to_value(values=init), target_accept_prob=target_accept)
    chain_method = "parallel" if jax.local_device_count() >= num_chains else "sequential"
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        thinning=thinning,
        chain_method=chain_method,
        progress_bar=False,
    )
    # チェーンごとに初期値を少しずらす（R̂ を意味のある診断にするため）
    k_init, k_run = jax.random.split(key)
    u0 = unconstrain_fn(model, (y,), {}, init)
    keys = jax.random.split(k_init, len(u0))
    init_params = {
        name: v + init_jitter * jax.random.normal(k, (num_chains,) + jnp.shape(v))
        for (name, v), k in zip(u0.items(), keys, strict=True)
    }

    t0 = time.perf_counter()
    mcmc.run(k_run, y, init_params=init_params)
    samples = mcmc.get_samples(group_by_chain=True)
    jax.block_until_ready(samples)
    elapsed = time.perf_counter() - t0

    names = spec.global_names
    rhat = {k: float(split_gelman_rubin(np.asarray(samples[k]))) for k in names}
    ess = {k: float(effective_sample_size(np.asarray(samples[k]))) for k in names}
    h = np.asarray(samples["h"])  # (chains, draws, T)
    rhat_h = np.asarray(split_gelman_rubin(h))
    ess_h = np.asarray(effective_sample_size(h))
    divergences = int(np.asarray(mcmc.get_extra_fields()["diverging"]).sum())
    theta = {k: np.asarray(samples[k]).reshape(-1) for k in names}
    del samples
    return _summarize(
        "NUTS",
        theta,
        h.reshape(-1, h.shape[-1]),
        time_total=elapsed,
        extra={
            "rhat": rhat,
            "ess": ess,
            "rhat_h_max": float(rhat_h.max()),
            "ess_h_min": float(ess_h.min()),
            "divergences": divergences,
            "min_ess_per_sec": min(ess.values()) / elapsed,
        },
    )


# ---------------------------------------------------------------------------
# NumPyro SVI
# ---------------------------------------------------------------------------


def run_svi(
    y,
    spec: ModelSpec,
    kind: str,
    key,
    num_steps: int = 10_000,
    lr: float = 0.01,
    lr_final_ratio: float = 0.1,
    num_mc: int = 4,
    num_post: int = 1000,
    khat: bool = True,
    rank: int = 10,
) -> PosteriorSummary:
    y = jnp.asarray(y)
    model = partial(sv_model, spec=spec)
    init = initial_values(y, spec)
    guide = make_guide(kind, model, spec, init, rank=rank)
    schedule = optax.exponential_decay(lr, num_steps, lr_final_ratio)
    optim = numpyro.optim.optax_to_numpyro(optax.adam(schedule))
    svi = SVI(model, guide, optim, Trace_ELBO(num_particles=num_mc))
    k_fit, k_post, k_w = jax.random.split(key, 3)

    # NumPyro の svi.run はステップ数ごとに再コンパイルするため、コンパイル時間は分離せず総時間で測る
    t0 = time.perf_counter()
    result = svi.run(k_fit, num_steps, y, progress_bar=False)
    jax.block_until_ready(result.params)
    elapsed = time.perf_counter() - t0

    post = Predictive(guide, params=result.params, num_samples=num_post)(k_post, y)
    theta = {k: post[k] for k in spec.global_names}
    latent = post[spec.latent_name]
    h = latent if spec.centered else theta["mu"][:, None] + theta["sigma"][:, None] * latent
    extra = {}
    if khat:
        lw = numpyro_log_weights(model, guide, result.params, y, k_w, num_post)
        extra["khat"] = psis_khat(lw)
    return _summarize(
        f"SVI-{kind}",
        theta,
        h,
        time_total=elapsed,
        elbo_trace=-np.asarray(result.losses),
        extra=extra,
    )


# ---------------------------------------------------------------------------
# 自前 ADVI
# ---------------------------------------------------------------------------


def run_advi(
    y,
    spec: ModelSpec,
    key,
    config: advi.ADVIConfig = advi.ADVIConfig(),
    num_post: int = 1000,
    khat: bool = True,
) -> PosteriorSummary:
    y = jnp.asarray(y)
    init = initial_values(y, spec)
    params0 = advi.init_params(init, spec, config.family, y)
    k_fit, k_post = jax.random.split(key)

    t0 = time.perf_counter()
    compiled = advi.fit.lower(params0, y, k_fit, spec, config).compile()
    t_compile = time.perf_counter() - t0
    params, elbo_trace = compiled(params0, y, k_fit)
    jax.block_until_ready(params)
    elapsed = time.perf_counter() - t0

    z, _ = advi.draw(params, config.family, k_post, num_post, spec)
    k = spec.n_global
    theta, _ = jax.vmap(lambda u: constrain(u, spec))(z[:, :k])
    h = jax.vmap(lambda zl, th: latent_to_h(zl, th, spec))(z[:, k:], theta)
    extra = {}
    if khat:
        log_p = jax.vmap(lambda zi: log_joint_unconstrained(zi, y, spec))(z)
        extra["khat"] = psis_khat(np.asarray(log_p - advi.log_q(params, config.family, z, spec)))
    return _summarize(
        f"ADVI-{config.family}",
        {n: theta[n] for n in spec.global_names},
        h,
        time_total=elapsed,
        time_compile=t_compile,
        elbo_trace=np.asarray(elbo_trace),
        extra=extra,
    )
