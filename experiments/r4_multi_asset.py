"""実験 R4：多銘柄（TOPIX Core30 ＋ ETF 2 本）のボラティリティを一括で推定する。

「毎晩のバッチで全銘柄のボラを何分で更新できるか」をシステムの観点で示す。
- VI 並列   : 提案 VI（基本SV）を CPU の複数デバイスに銘柄を振り分けて並列実行（jax.pmap ＋ デバイス内は lax.map で順番に）
- VI vmap   : 銘柄方向に jax.vmap でベクトル化して 1 回の呼び出しで推定（参考。CPU ではかえって遅い）
- VI 逐次   : 同じ VI を銘柄ごとに Python のループで順に実行（コンパイルは最初の 1 回だけ）
- NUTS 逐次 : 銘柄ごとに NUTS（4 チェーン並列、warmup・sample 各 2000）
全銘柄を共通の取引日に揃える（まとめて計算するには同じ長さが必要）。
デバイス数は環境変数 SV_VI_HOST_DEVICES で指定する（本番は 8。NUTS は そのうち 4 つで 4 チェーン並列）。

出力：results/r4/（per_stock.csv, timing.json）、図6、表A4
"""

import json
import time

import common
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from sv_vi import advi, plotting
from sv_vi.data import ETF_CODES, RAW_DIR, aligned_returns, cached_daily_bars, to_returns
from sv_vi.inference import run_advi
from sv_vi.initialization import initial_values
from sv_vi.logdensity import ModelSpec, constrain

SPEC = ModelSpec()
CONFIG = advi.ADVIConfig(family="conditional", num_steps=10_000)


def load_panel(quick: bool) -> pd.DataFrame:
    core = pd.read_csv(RAW_DIR / "core30.csv", dtype={"Code": str})
    codes = list(ETF_CODES) + core["Code"].tolist()
    if quick:
        codes = codes[:4]
    # 期間途中で上場した銘柄（例：87290）は共通期間を大きく縮めるので除外する
    lengths = {c: len(to_returns(cached_daily_bars(c))) for c in codes}
    full = max(lengths.values())
    dropped = [c for c, n in lengths.items() if n < 0.9 * full]
    if dropped:
        print("期間が短いため除外:", dropped, flush=True)
    panel = aligned_returns([c for c in codes if c not in dropped])
    return panel - panel.mean()


def _stack_params(Y, config):
    params0 = [advi.init_params(initial_values(y, SPEC), SPEC, config.family, y) for y in Y]
    return jax.tree.map(lambda *xs: jnp.stack(xs), *params0)


def _posterior_theta(params, key, config) -> dict:
    def post(p, k):
        z, _ = advi.draw(p, config.family, k, 1000, SPEC)
        theta, _ = jax.vmap(lambda u: constrain(u, SPEC))(z[:, : SPEC.n_global])
        return theta

    n = jax.tree.leaves(params)[0].shape[0]
    theta = jax.vmap(post)(params, jax.random.split(key, n))
    return {k: np.asarray(v) for k, v in theta.items()}


def vi_vmap(Y: np.ndarray, key, config) -> tuple[dict, float, float]:
    """全銘柄を vmap でベクトル化して一括推定。(事後サンプル, 総時間, コンパイル時間)。"""
    params0, keys, Yj = _stack_params(Y, config), jax.random.split(key, len(Y)), jnp.asarray(Y)

    @jax.jit
    def fit_all(p0, ys, ks):
        return jax.vmap(lambda p, y, k: advi.fit(p, y, k, spec=SPEC, config=config))(p0, ys, ks)

    t0 = time.perf_counter()
    compiled = fit_all.lower(params0, Yj, keys).compile()
    t_compile = time.perf_counter() - t0
    params, _ = compiled(params0, Yj, keys)
    jax.block_until_ready(params)
    t_total = time.perf_counter() - t0
    return _posterior_theta(params, jax.random.fold_in(key, 1), config), t_total, t_compile


def vi_parallel(Y: np.ndarray, key, config) -> tuple[dict, float, float]:
    """銘柄を CPU デバイスに振り分けて並列推定（pmap）。各デバイス内では lax.map で 1 銘柄ずつ順に処理する。"""
    n_dev = jax.local_device_count()
    n = len(Y)
    pad = (-n) % n_dev  # デバイス数で割り切れるよう、最後の銘柄を複製して埋める
    Yp = np.concatenate([Y, np.repeat(Y[-1:], pad, axis=0)]) if pad else Y
    params0, keys = _stack_params(Yp, config), jax.random.split(key, len(Yp))

    def per_device(p, y, k):
        return jax.lax.map(lambda a: advi.fit(*a, spec=SPEC, config=config), (p, y, k))

    resh = lambda x: x.reshape((n_dev, -1) + x.shape[1:])  # noqa: E731
    args = (jax.tree.map(resh, params0), resh(jnp.asarray(Yp)), resh(keys))
    fit_all = jax.pmap(per_device)
    t0 = time.perf_counter()
    compiled = fit_all.lower(*args).compile()
    t_compile = time.perf_counter() - t0
    params, _ = compiled(*args)
    jax.block_until_ready(params)
    t_total = time.perf_counter() - t0
    params = jax.tree.map(lambda x: x.reshape((-1,) + x.shape[2:])[:n], params)
    return _posterior_theta(params, jax.random.fold_in(key, 1), config), t_total, t_compile


def run(args, cfg, out):
    panel = load_panel(args.quick)
    codes = list(panel.columns)
    Y = panel.values.T  # (銘柄, T)
    print(
        f"{len(codes)} 銘柄 × T={Y.shape[1]}（{panel.index.min():%Y-%m-%d}〜{panel.index.max():%Y-%m-%d}）",
        flush=True,
    )
    config = CONFIG if not args.quick else advi.ADVIConfig(family="conditional", num_steps=1000)
    timing = {"n_assets": len(codes), "T": int(Y.shape[1])}

    timing["n_devices"] = jax.local_device_count()
    theta_b, timing["vi_parallel_total"], timing["vi_parallel_compile"] = vi_parallel(
        Y, common.data_key(args.seed, 40), config
    )
    _, timing["vi_vmap_total"], timing["vi_vmap_compile"] = vi_vmap(
        Y, common.data_key(args.seed, 43), config
    )
    print({k: round(v, 1) for k, v in timing.items()}, flush=True)

    rows = []
    t0 = time.perf_counter()
    seq_times = []
    for i, (code, y) in enumerate(zip(codes, Y, strict=True)):
        r = run_advi(y, SPEC, common.data_key(args.seed, 41, i), config, khat=False, aot=False)
        seq_times.append(r.time_total)
        rows.append({"code": code, **{f"vi_{k}_mean": float(v.mean()) for k, v in r.theta.items()},
                     **{f"vi_{k}_sd": float(v.std()) for k, v in r.theta.items()}})  # fmt: skip
    timing["vi_seq_total"] = time.perf_counter() - t0
    timing["vi_seq_first"] = seq_times[0]
    timing["vi_seq_median"] = (
        float(np.median(seq_times[1:])) if len(seq_times) > 1 else seq_times[0]
    )

    t0 = time.perf_counter()
    nuts_times = []
    for i, (code, y) in enumerate(zip(codes, Y, strict=True)):
        r = common.run_method("NUTS", y, SPEC, common.data_key(args.seed, 42, i), cfg)
        nuts_times.append(r.time_total)
        rows[i] |= {**{f"nuts_{k}_mean": float(v.mean()) for k, v in r.theta.items()},
                    **{f"nuts_{k}_sd": float(v.std()) for k, v in r.theta.items()},
                    "nuts_rhat_max": max(r.extra["rhat"].values()), "nuts_time": r.time_total}  # fmt: skip
        for k, v in theta_b.items():
            rows[i][f"vib_{k}_mean"] = float(v[i].mean())
        print(f"[{i + 1}/{len(codes)}] {code}: NUTS {r.time_total:.1f}s phi VI {rows[i]['vi_phi_mean']:.3f} "
              f"/ NUTS {rows[i]['nuts_phi_mean']:.3f}", flush=True)  # fmt: skip
        jax.clear_caches()
    timing["nuts_seq_total"] = time.perf_counter() - t0
    timing["nuts_median"] = float(np.median(nuts_times))

    core = pd.read_csv(RAW_DIR / "core30.csv", dtype={"Code": str}).set_index("Code")
    df = pd.DataFrame(rows)
    df["name"] = [core["CoName"].get(c, ETF_CODES.get(c, c)) for c in df["code"]]
    df.to_csv(out / "per_stock.csv", index=False)
    (out / "timing.json").write_text(json.dumps(timing, indent=2))
    return df, timing


def fig(df: pd.DataFrame, timing: dict):
    """図6：(a) 全銘柄の推定にかかった総時間、(b)(c) φ・σ の事後平均（VI vs NUTS）。"""
    plt = plotting.plt
    fig, axes = plt.subplots(1, 3, figsize=(plotting.TEXT_WIDTH_IN, 2.2), layout="constrained",
                             gridspec_kw={"width_ratios": [1.3, 1, 1]})  # fmt: skip
    ax = axes[0]
    bars = [
        ("NUTS\n(one by one)", timing["nuts_seq_total"], plotting.color("NUTS")),
        ("VI\n(one by one)", timing["vi_seq_total"], plotting.color("ADVI-conditional")),
        ("VI\n(vmap)", timing["vi_vmap_total"], plotting.color("ADVI-conditional")),
        (
            f"VI\n({timing['n_devices']} CPU devices)",
            timing["vi_parallel_total"],
            plotting.color("ADVI-conditional"),
        ),
    ]
    ypos = np.arange(len(bars))[::-1]
    for yp, (_lab, t, col) in zip(ypos, bars, strict=True):
        ax.barh(yp, t, color=col, height=0.55)
        ax.text(t * 1.15, yp, f"{t:.0f} s", va="center", fontsize=7)
    ax.set_yticks(ypos, [b[0] for b in bars])
    ax.set_xscale("log")
    ax.set_xlim(1, max(b[1] for b in bars) * 8)
    ax.set_xlabel("total time for all assets (s)")
    ax.set_title(f"(a) {timing['n_assets']} assets, T={timing['T']}", loc="left")
    ax.grid(axis="y", visible=False)
    for ax, (k, tex) in zip(axes[1:], [("phi", r"$\phi$"), ("sigma", r"$\sigma$")], strict=True):
        x, y = df[f"nuts_{k}_mean"], df[f"vi_{k}_mean"]
        ax.errorbar(x, y, xerr=df[f"nuts_{k}_sd"], yerr=df[f"vi_{k}_sd"], fmt="o", ms=3.5,
                    color=plotting.color("ADVI-conditional"), ecolor=plotting.GRID, elinewidth=0.8)  # fmt: skip
        lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
        pad = 0.05 * (hi - lo)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=plotting.INK_SECONDARY, lw=0.8)
        ax.set_xlabel(f"NUTS posterior mean of {tex}")
        ax.set_ylabel(f"VI posterior mean of {tex}")
        ax.set_title(f"({'b' if k == 'phi' else 'c'}) {tex}", loc="left")
    return plotting.save(fig, "fig6_multi_asset")


def table(df: pd.DataFrame, timing: dict):
    lines = [
        r"\begin{tabular}{lr}",
        r"\toprule",
        r"項目 & 値 \\",
        r"\midrule",
        f"銘柄数 × 系列長 & {timing['n_assets']} × {timing['T']} \\\\",
        f"NUTS（逐次）の総時間 & {timing['nuts_seq_total']:.0f} 秒 \\\\",
        f"提案 VI（逐次）の総時間 & {timing['vi_seq_total']:.1f} 秒 \\\\",
        f"提案 VI（vmap でベクトル化）の総時間 & {timing['vi_vmap_total']:.1f} 秒 \\\\",
        f"提案 VI（{timing['n_devices']} デバイス並列）の総時間 & {timing['vi_parallel_total']:.1f} 秒（うちコンパイル {timing['vi_parallel_compile']:.1f} 秒） \\\\",
        f"$\\phi$ の事後平均の差（中央値） & {np.median(np.abs(df['vi_phi_mean'] - df['nuts_phi_mean'])):.3f} \\\\",
        f"$\\sigma$ の事後平均の差（中央値） & {np.median(np.abs(df['vi_sigma_mean'] - df['nuts_sigma_mean'])):.3f} \\\\",
        f"並列と逐次の VI の $\\phi$ の差（最大） & {np.max(np.abs(df['vib_phi_mean'] - df['vi_phi_mean'])):.3f} \\\\",
        f"NUTS の $\\hat R$ > 1.05 の銘柄数 & {(df['nuts_rhat_max'] > 1.05).sum()} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    (common.TABLES / "tabA4_multi_asset.tex").write_text("\n".join(lines) + "\n")


def main():
    args = common.parse_args(__doc__)
    cfg = common.RunConfig.quick() if args.quick else common.RunConfig()
    out = common.RESULTS / ("r4_quick" if args.quick else "r4")
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    df, timing = run(args, cfg, out)
    plotting.setup()
    fig(df, timing)
    table(df, timing)
    common.write_env(out, args, cfg, started)
    print(json.dumps(timing, indent=2))
    print(df[["code", "vi_phi_mean", "nuts_phi_mean", "vi_sigma_mean", "nuts_sigma_mean", "nuts_rhat_max"]]
          .to_string(float_format=lambda v: f"{v:.3f}"))  # fmt: skip


if __name__ == "__main__":
    main()
