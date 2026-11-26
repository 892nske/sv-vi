"""4 章の図表：変分族の比較。

- 図 fig_sigma_roughness：S2 の合成データ（高持続, T=1000）で、事後サンプルごとの σ と h の経路の粗さ
  sqrt(mean_t (h_{t+1} - h_t)^2) の散布図。NUTS（S2 と同じ長さ）、提案族（条件付き構造化）、構造化族。
  真の事後分布では σ が大きいほど h の経路が粗い（正の相関）。q(h) が θ に依存しない構造化族ではこの依存が消える。
- 表 tab_families.tex：各変分族のパラメータ数（基本 SV, T=1000）と性質

出力：results/f4/（サンプルの npz）、docs/figures/fig_sigma_roughness.pdf、docs/tables/tab_families.tex
"""

from functools import partial

import common
import jax
import jax.numpy as jnp
import numpy as np
from numpyro.infer import MCMC, NUTS, init_to_value
from numpyro.infer.util import unconstrain_fn

from sv_vi import advi, plotting
from sv_vi.initialization import initial_values
from sv_vi.logdensity import ModelSpec, constrain
from sv_vi.models import sv_model

T = 1000
SPEC = ModelSpec()
VI_STEPS = 20_000  # S2 と同じ
NUM_DRAWS = 4000


def roughness(h: np.ndarray) -> np.ndarray:
    """経路の粗さ sqrt(mean_t (h_{t+1} - h_t)^2)。h は (S, T)。"""
    return np.sqrt(np.mean(np.diff(h, axis=-1) ** 2, axis=-1))


def nuts_draws(y, key, cfg: common.RunConfig):
    """S2 と同じ設定の NUTS（4 連鎖、warmup 2000、32000 回を 8 つおき）の σ と h。"""
    model = partial(sv_model, spec=SPEC)
    init = initial_values(y, SPEC)
    kernel = NUTS(model, init_strategy=init_to_value(values=init), target_accept_prob=0.9)
    num_chains = 4
    chain_method = "parallel" if jax.local_device_count() >= num_chains else "sequential"
    mcmc = MCMC(
        kernel,
        num_warmup=cfg.nuts_warmup,
        num_samples=16 * cfg.nuts_samples,
        num_chains=num_chains,
        thinning=8,
        chain_method=chain_method,
        progress_bar=False,
    )
    k_init, k_run = jax.random.split(key)
    u0 = unconstrain_fn(model, (y,), {}, init)
    keys = jax.random.split(k_init, len(u0))
    init_params = {
        name: v + 0.1 * jax.random.normal(k, (num_chains,) + jnp.shape(v))
        for (name, v), k in zip(u0.items(), keys, strict=True)
    }
    mcmc.run(k_run, jnp.asarray(y), init_params=init_params)
    s = mcmc.get_samples()
    return np.asarray(s["sigma"]), np.asarray(s["h"])


def vi_draws(y, family: str, key, cfg: common.RunConfig):
    k_fit, k_draw = jax.random.split(key)
    y = jnp.asarray(y)
    init = initial_values(y, SPEC)
    params0 = advi.init_params(init, SPEC, family, y)
    config = advi.ADVIConfig(family=family, num_steps=VI_STEPS, lr=cfg.vi_lr, num_mc=cfg.vi_num_mc)
    params, _ = advi.fit(params0, y, k_fit, SPEC, config)
    z, _ = advi.draw(params, family, k_draw, NUM_DRAWS, SPEC)
    k = SPEC.n_global
    theta = jax.vmap(lambda u: constrain(u, SPEC)[0])(z[:, :k])
    return np.asarray(theta["sigma"]), np.asarray(z[:, k:])


def run(args, out):
    cfg = common.RunConfig()
    y = np.load(common.RESULTS / "s2" / "data.npz")["y"]
    assert len(y) == T
    res = {}
    for i, m in enumerate(["NUTS", "ADVI-conditional", "ADVI-structured"]):
        path = out / f"{m}.npz"
        if path.exists():
            d = np.load(path)
            res[m] = (d["sigma"], d["rough"])
            continue
        key = common.data_key(args.seed, 4, i)
        if m == "NUTS":
            sigma, h = nuts_draws(y, key, cfg)
        else:
            sigma, h = vi_draws(y, m.removeprefix("ADVI-"), key, cfg)
        rough = roughness(h)
        np.savez(path, sigma=sigma, rough=rough)
        res[m] = (sigma, rough)
        print(f"{m}: corr(σ, rough) = {np.corrcoef(sigma, rough)[0, 1]:.3f}", flush=True)
    return res


def fig(res):
    plt = plotting.plt
    fig, ax = plt.subplots(figsize=(plotting.TEXT_WIDTH_IN * 0.75, 3.4))
    rng = np.random.default_rng(0)
    for m, (sigma, rough) in res.items():
        idx = rng.choice(len(sigma), size=min(1500, len(sigma)), replace=False)
        r = np.corrcoef(sigma, rough)[0, 1]
        ax.scatter(
            sigma[idx],
            rough[idx],
            s=6,
            alpha=0.35,
            color=plotting.color(m),
            marker=plotting.marker(m),
            linewidths=0,
            zorder=3 if m == "NUTS" else 2,  # 正解の NUTS を手前に描く
            label=f"{plotting.label(m)}  (corr = {r:.2f})",
        )
    ax.set_xlabel(r"$\sigma$")
    ax.set_ylabel(r"roughness of $h$:  $\sqrt{\mathrm{mean}_t\,(h_{t+1}-h_t)^2}$")
    leg = ax.legend(loc="upper left", markerscale=2.5)
    for lh in leg.legend_handles:
        lh.set_alpha(1.0)
    return plotting.save(fig, "fig_sigma_roughness")


def param_counts(k: int = 3, T: int = T, rank: int = 10) -> dict:
    """各族の自由な変分パラメータの数（下三角行列は k(k+1)/2 個と数える）。"""
    d = k + T
    tri = k * (k + 1) // 2
    counts = {
        "meanfield": 2 * d,
        "lowrank": d * (rank + 2),  # 平均 d、因子 d×r、対角 d
        "fullrank": d + d * (d + 1) // 2,
        "structured": k + tri + T + T + (T - 1),  # q(u)、h の平均・対角・副対角
        "conditional": k + tri + 2 * T,  # q(u)、擬似観測・その精度
    }
    # 自前実装の変分パラメータの実数の個数と照合する（u_tril は k×k で持つので上三角の分を引く）
    y = jnp.zeros(T) + 1.0
    init = initial_values(y, SPEC)
    for fam in ("meanfield", "structured", "conditional"):
        p = advi.init_params(init, SPEC, fam, y)
        n = sum(int(np.size(v)) for v in jax.tree_util.tree_leaves(p))
        if "u_tril" in p:
            n -= k * (k - 1) // 2
        assert n == counts[fam], (fam, n, counts[fam])
    return counts


def table():
    c = param_counts()
    rows = [
        ("平均場", "対角", "対角", "×", "×", "$O(T)$", c["meanfield"], "×"),
        ("低ランク（$r=10$）", r"\multicolumn{2}{c}{$WW^\top + D$}", None, "△", "△", "$O(Tr)$", c["lowrank"], "×"),
        ("フルランク", r"\multicolumn{2}{c}{密な共分散 $LL^\top$}", None, "○", "△", "$O(T^2)$", c["fullrank"], "×"),
        ("構造化", "フルランク", "三重対角の精度", "○", "×", "$O(T)$", c["structured"], "×"),
        ("条件付き構造化", "フルランク", r"$Q_{\mathrm{AR}}(\theta) + \diag(\lambda)$", "○", "○", "$O(T)$", c["conditional"], "○"),
    ]  # fmt: skip
    lines = [
        r"\begin{tabular}{lllcccrc}",
        r"\toprule",
        r" & \multicolumn{2}{c}{共分散（精度）の構造} & (R1) & (R2) & \multicolumn{2}{c}{(R3)} & (R4) \\",
        r"\cmidrule(lr){2-3} \cmidrule(lr){6-7}",
        r"変分族 & $q(\theta)$ & $q(h)$ & 時間相関 & $\theta$--$h$ 依存 & 計算量 & パラメータ数 & 不変性 \\",
        r"\midrule",
    ]
    for name, qt, qh, r1, r2, cost, n, r4 in rows:
        cov = qt if qh is None else f"{qt} & {qh}"
        lines.append(f"{name} & {cov} & {r1} & {r2} & {cost} & {n:,} & {r4} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (common.TABLES / "tab_families.tex").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    args = common.parse_args(__doc__)
    out = common.RESULTS / "f4"
    out.mkdir(parents=True, exist_ok=True)
    common.TABLES.mkdir(parents=True, exist_ok=True)
    table()
    res = run(args, out)
    plotting.setup()
    fig(res)


if __name__ == "__main__":
    main()
