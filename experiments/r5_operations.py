"""実験 R5：運用に関わる測定（7 章）。

(a) warm start のステップ数：1321・1306 のローリング推定（R2 と同じ窓 250 日・検証 235 日）を、
    warm start のステップ数 K ∈ {100, 250, 500, 1000} で回し、毎回ゼロからの推定（R2 の 10,000 ステップ）と比べる。
    K = 2,000 と毎回ゼロからは R2 の forecasts.csv をそのまま使う。
(b) 多銘柄の日次更新：32 銘柄の直近の窓（250 日）について、初日は全銘柄をゼロから推定し、続く 5 日は warm start
    （2,000 ステップ）で更新する。CPU デバイスに銘柄を振り分け（pmap）、デバイス内では lax.map で順に処理する。
    銘柄数 N ∈ {32, 128, 512} は 32 銘柄を複製して乱数のキーだけ変えたもの（時間の測定のためだけの複製）。
    さらに 1 デバイス上で 32 銘柄を lax.map（1 銘柄ずつ）と vmap（ベクトル化）で処理した時間を比べる。
(c) 再現性：同じキーで 2 回推定したときの一致、逐次（jit）と並列（pmap + lax.map）の差、キーを変えたときのばらつき。

実行：SV_VI_HOST_DEVICES=8 uv run python experiments/r5_operations.py
出力：results/r5/、docs/tables/tab_warm_steps.tex、docs/tables/tab_multi_scaling.tex、docs/figures/fig7_monitoring.pdf
"""

import json
import time

import common
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from r1_full_period import ETF_JA, ETFS, load_returns
from r2_var_backtest import W
from r4_multi_asset import load_panel

from sv_vi import advi, plotting
from sv_vi.diagnostics import convergence_step
from sv_vi.inference import run_advi
from sv_vi.initialization import initial_values
from sv_vi.logdensity import ModelSpec
from sv_vi.predictive import forecast

SPEC = ModelSpec()
FAMILY = "conditional"
WARM_STEPS = [100, 250, 500, 1000]
COLD_STEPS = 10_000
R2_WARM_STEPS = 2000
N_ASSETS = [32, 128, 512]
N_WARM_DAYS = 5


# ---------------------------------------------------------------------------
# (a) warm start のステップ数
# ---------------------------------------------------------------------------


def run_warm_steps(args, out) -> pd.DataFrame:
    csv = out / "warm_steps.csv"
    if csv.exists():
        return pd.read_csv(csv, dtype={"code": str})
    rows = []
    for ci, code in enumerate(ETFS):
        raw = load_returns(code)
        dates, raw = raw.index, raw.values
        for K in WARM_STEPS:
            prev = None
            for d in range(len(raw) - W):
                t = W + d
                yw = raw[t - W : t]
                m = yw.mean()
                yw = yw - m
                y_obs = raw[t] - m
                key = common.data_key(args.seed, 50, ci, K, d)
                steps = K if prev is not None else COLD_STEPS
                config = advi.ADVIConfig(family=FAMILY, num_steps=steps)
                p0 = advi.shift_params(prev, FAMILY, SPEC) if prev is not None else None
                r = run_advi(
                    yw, SPEC, key, config, num_post=1000, khat=False, params0=p0, aot=False
                )
                prev = r.params
                f = forecast(r, yw[-1], y_obs, SPEC, seed=int(t * 100 + 7))
                rows.append({
                    "code": code, "date": str(dates[t].date()), "K": K, "y_obs": y_obs, **f,
                    "time": r.time_total, "phi": float(r.theta["phi"].mean()),
                    "sigma": float(r.theta["sigma"].mean()),
                    "elbo_final": float(np.mean(r.elbo_trace[-min(200, steps):])),
                    "conv_step": convergence_step(r.elbo_trace),
                })  # fmt: skip
            print(f"{code} K={K} done", flush=True)
            jax.clear_caches()
    df = pd.DataFrame(rows)
    df.to_csv(csv, index=False)
    return df


def summarize_warm_steps(ws: pd.DataFrame, out) -> pd.DataFrame:
    """毎回ゼロから（R2 の SV-cond(cold)）を基準にした、K ごとの差と時間。"""
    fc = pd.read_csv(common.RESULTS / "r2" / "forecasts.csv", dtype={"code": str})
    r2 = fc[fc["method"] == "SV-cond(warm)"].assign(K=R2_WARM_STEPS)
    cols = ["code", "date", "K", "y_obs", "var5", "var1", "log_score", "time", "phi", "sigma"]
    allk = pd.concat([ws[cols], r2[cols]], ignore_index=True)
    cold = fc[fc["method"] == "SV-cond(cold)"].set_index(["code", "date"])
    recs = []
    for (code, K), d in allk.groupby(["code", "K"]):
        d = d.sort_values("date").iloc[1:]  # 初日（ゼロから推定）を除く
        c = cold.loc[[(code, x) for x in d["date"]]]
        dv = np.abs(d["var5"].values - c["var5"].values)
        recs.append({
            "code": code, "K": K,
            "var5_med": np.median(dv), "var5_q95": np.quantile(dv, 0.95), "var5_max": dv.max(),
            "phi_med": np.median(np.abs(d["phi"].values - c["phi"].values)),
            "sigma_med": np.median(np.abs(d["sigma"].values - c["sigma"].values)),
            "exceed5": int((d["y_obs"] < d["var5"]).sum()),
            "score": d["log_score"].mean(), "time_med": d["time"].median(),
        })  # fmt: skip
    # 基準（毎回ゼロから）の行
    for code in ETFS:
        c = cold.loc[code].sort_index().iloc[1:]
        recs.append({
            "code": code, "K": COLD_STEPS, "var5_med": np.nan, "var5_q95": np.nan, "var5_max": np.nan,
            "phi_med": np.nan, "sigma_med": np.nan, "exceed5": int((c["y_obs"] < c["var5"]).sum()),
            "score": c["log_score"].mean(), "time_med": c["time"].median(),
        })  # fmt: skip
    s = pd.DataFrame(recs).sort_values(["code", "K"])
    s.to_csv(out / "warm_steps_summary.csv", index=False)
    return s


def table_warm_steps(s: pd.DataFrame):
    lines = [
        r"\begin{tabular}{rrrrrrrrr}",
        r"\toprule",
        r" & \multicolumn{3}{c}{5\% VaR の差の絶対値} & \multicolumn{2}{c}{差の中央値} & & & \\",
        r"\cmidrule(lr){2-4} \cmidrule(lr){5-6}",
        r"ステップ数 & 中央値 & 95\% 点 & 最大 & $\phi$ & $\sigma$ & 超過（回） & 予測スコア & 時間[s/日] \\",
    ]
    for code in ETFS:
        lines += [r"\midrule", rf"\multicolumn{{9}}{{l}}{{{ETF_JA[code]}}} \\"]
        for _, r in s[s["code"] == code].iterrows():
            if r["K"] == COLD_STEPS:
                head = f"{COLD_STEPS:,}（毎回ゼロから）"
                diffs = " & ".join(["--"] * 5)
            else:
                head = f"{int(r['K']):,}"
                diffs = (
                    f"{r['var5_med']:.3f} & {r['var5_q95']:.3f} & {r['var5_max']:.3f} & "
                    f"{r['phi_med']:.4f} & {r['sigma_med']:.4f}"
                )
            lines.append(
                f"{head} & {diffs} & {int(r['exceed5'])} & {r['score']:.3f} & {r['time_med']:.3f} \\\\"
            )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (common.TABLES / "tab_warm_steps.tex").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# (b) 多銘柄の日次更新、(c) 再現性
# ---------------------------------------------------------------------------


def _stack(trees):
    return jax.tree.map(lambda *xs: jnp.stack(xs), *trees)


def _init_stacked(Y):
    return _stack([advi.init_params(initial_values(y, SPEC), SPEC, FAMILY, y) for y in Y])


def _pmap_fit(config):
    """銘柄を CPU デバイスに振り分け、デバイス内は lax.map で 1 銘柄ずつ処理する関数。"""

    def per_device(p, y, k):
        return jax.lax.map(lambda a: advi.fit(*a, spec=SPEC, config=config)[0], (p, y, k))

    return jax.pmap(per_device)


def _to_dev(x, n_dev):
    return x.reshape((n_dev, -1) + x.shape[1:])


def _from_dev(x):
    return x.reshape((-1,) + x.shape[2:])


def run_multi(args, out) -> dict:
    path = out / "multi.json"
    if path.exists():
        return json.loads(path.read_text())
    panel = load_panel(False)
    Y_all = panel.values.T  # (32, 485)
    n_base, n_total = Y_all.shape
    n_dev = jax.local_device_count()
    start = n_total - W - N_WARM_DAYS  # 初日の窓の開始位置（最後の窓が直近 250 日になる）
    cold_cfg = advi.ADVIConfig(family=FAMILY, num_steps=COLD_STEPS)
    warm_cfg = advi.ADVIConfig(family=FAMILY, num_steps=R2_WARM_STEPS)
    shift = jax.jit(jax.vmap(lambda p: advi.shift_params(p, FAMILY, SPEC)))
    res = {"n_devices": n_dev, "T_window": W, "n_base": n_base, "warm_steps": R2_WARM_STEPS}
    for N in N_ASSETS:
        reps = N // n_base
        Y = np.tile(Y_all, (reps, 1))
        keys0 = jax.random.split(common.data_key(args.seed, 51, N), N)

        def window(i, Y=Y):
            yw = Y[:, start + i : start + i + W]
            return jnp.asarray(yw - yw.mean(axis=1, keepdims=True))

        # 初日：ゼロから
        y0 = window(0)
        p0 = _init_stacked(np.asarray(y0))
        fit_cold, fit_warm = _pmap_fit(cold_cfg), _pmap_fit(warm_cfg)
        t0 = time.perf_counter()
        c_cold = fit_cold.lower(
            _to_dev_tree(p0, n_dev), _to_dev(y0, n_dev), _to_dev(keys0, n_dev)
        ).compile()
        t_cc = time.perf_counter() - t0
        t0 = time.perf_counter()
        params = jax.block_until_ready(
            c_cold(_to_dev_tree(p0, n_dev), _to_dev(y0, n_dev), _to_dev(keys0, n_dev))
        )
        t_cold = time.perf_counter() - t0
        params = jax.tree.map(_from_dev, params)
        # 続く日：warm start
        warm_times, t_wc = [], None
        for i in range(1, N_WARM_DAYS + 1):
            yi = window(i)
            ki = jax.random.split(common.data_key(args.seed, 51, N, i), N)
            pi = shift(params)
            a = (_to_dev_tree(pi, n_dev), _to_dev(yi, n_dev), _to_dev(ki, n_dev))
            if t_wc is None:
                t0 = time.perf_counter()
                c_warm = fit_warm.lower(*a).compile()
                t_wc = time.perf_counter() - t0
            t0 = time.perf_counter()
            params = jax.tree.map(_from_dev, jax.block_until_ready(c_warm(*a)))
            warm_times.append(time.perf_counter() - t0)
        res[f"N{N}"] = {
            "cold_compile": t_cc, "cold_run": t_cold, "warm_compile": t_wc,
            "warm_runs": warm_times, "warm_median": float(np.median(warm_times)),
        }  # fmt: skip
        print(f"N={N}: cold {t_cold:.1f}s (+compile {t_cc:.1f}s), warm median {np.median(warm_times):.2f}s",
              flush=True)  # fmt: skip
        jax.clear_caches()

    # 1 デバイス上での lax.map と vmap（32 銘柄、warm start 相当の 2,000 ステップ）
    y0 = jnp.asarray(Y_all[:, -W:] - Y_all[:, -W:].mean(axis=1, keepdims=True))
    p0 = _init_stacked(np.asarray(y0))
    ks = jax.random.split(common.data_key(args.seed, 52), n_base)
    f_map = jax.jit(
        lambda p, y, k: jax.lax.map(
            lambda a: advi.fit(*a, spec=SPEC, config=warm_cfg)[0], (p, y, k)
        )
    )
    f_vmap = jax.jit(
        lambda p, y, k: jax.vmap(lambda *a: advi.fit(*a, spec=SPEC, config=warm_cfg)[0])(p, y, k)
    )
    f_one = jax.jit(lambda p, y, k: advi.fit(p, y, k, spec=SPEC, config=warm_cfg)[0])
    for name, f in (("map", f_map), ("vmap", f_vmap)):
        c = f.lower(p0, y0, ks).compile()
        jax.block_until_ready(c(p0, y0, ks))  # 1 回目は捨てる
        t0 = time.perf_counter()
        out_p = jax.block_until_ready(c(p0, y0, ks))
        res[f"single_device_{name}"] = time.perf_counter() - t0
        res[f"_{name}_params"] = out_p
    p_one = jax.tree.map(lambda x: x[0], p0)
    c1 = f_one.lower(p_one, y0[0], ks[0]).compile()
    jax.block_until_ready(c1(p_one, y0[0], ks[0]))
    t0 = time.perf_counter()
    for i in range(n_base):
        jax.block_until_ready(c1(jax.tree.map(lambda x, i=i: x[i], p0), y0[i], ks[i]))
    res["single_device_loop"] = time.perf_counter() - t0
    print({k: round(v, 2) for k, v in res.items() if k.startswith("single")}, flush=True)

    # (c) 再現性
    p_a = c1(p_one, y0[0], ks[0])
    p_b = c1(p_one, y0[0], ks[0])
    res["repro_same_key_maxdiff"] = _maxdiff(p_a, p_b)
    p_map0 = jax.tree.map(lambda x: x[0], res.pop("_map_params"))
    p_vmap0 = jax.tree.map(lambda x: x[0], res.pop("_vmap_params"))
    res["repro_loop_vs_map_maxdiff"] = _maxdiff(p_a, p_map0)
    res["repro_loop_vs_vmap_maxdiff"] = _maxdiff(p_a, p_vmap0)
    # pmap（8 デバイス）と逐次の差：同じキー
    fit_par = _pmap_fit(warm_cfg)
    pad = (-n_base) % n_dev
    yp = jnp.concatenate([y0, jnp.repeat(y0[-1:], pad, axis=0)]) if pad else y0
    pp = _init_stacked(np.asarray(yp))
    kp = jnp.concatenate([ks, jnp.repeat(ks[-1:], pad, axis=0)]) if pad else ks
    par = jax.tree.map(
        _from_dev, fit_par(_to_dev_tree(pp, n_dev), _to_dev(yp, n_dev), _to_dev(kp, n_dev))
    )
    res["repro_loop_vs_pmap_maxdiff"] = _maxdiff(p_a, jax.tree.map(lambda x: x[0], par))
    # キーを変えたときのばらつき（銘柄 0 の φ・σ の事後平均）
    theta = []
    for j in range(10):
        kj = common.data_key(args.seed, 53, j)
        r = run_advi(
            np.asarray(y0[0]), SPEC, kj, warm_cfg_cold(), num_post=4000, khat=False, aot=False
        )
        theta.append((float(r.theta["phi"].mean()), float(r.theta["sigma"].mean())))
    th = np.array(theta)
    res["seed_spread_phi_sd"], res["seed_spread_sigma_sd"] = (
        float(th[:, 0].std()),
        float(th[:, 1].std()),
    )
    res["seed_spread_phi_range"] = float(np.ptp(th[:, 0]))
    res["seed_spread_sigma_range"] = float(np.ptp(th[:, 1]))
    print({k: v for k, v in res.items() if k.startswith(("repro", "seed"))}, flush=True)
    path.write_text(json.dumps(res, indent=2, default=float))
    return res


def warm_cfg_cold():
    """キーによるばらつきを見るための、毎回ゼロからの設定（10,000 ステップ）。"""
    return advi.ADVIConfig(family=FAMILY, num_steps=COLD_STEPS)


def _to_dev_tree(tree, n_dev):
    return jax.tree.map(lambda x: _to_dev(x, n_dev), tree)


def _maxdiff(a, b) -> float:
    return float(
        max(
            jnp.max(jnp.abs(x - y))
            for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True)
        )
    )


def table_multi(res: dict):
    r4 = json.loads((common.RESULTS / "r4" / "timing.json").read_text())
    lines = [
        r"\begin{tabular}{llr}",
        r"\toprule",
        r"処理 & 方式 & 時間 [秒] \\",
        r"\midrule",
        rf"\multicolumn{{3}}{{l}}{{全期間の推定（{r4['n_assets']} 銘柄 × {r4['T']} 日、ゼロから）}} \\",
        rf"NUTS（4 連鎖、各 2,000 回） & 1 銘柄ずつ & {r4['nuts_seq_total']:.0f} \\",
        rf"提案 VI（10,000 ステップ） & 1 銘柄ずつ & {r4['vi_seq_total']:.1f} \\",
        rf" & vmap（ベクトル化） & {r4['vi_vmap_total']:.1f} \\",
        rf" & {r4['n_devices']} デバイスに振り分け & {r4['vi_parallel_total']:.1f}（うちコンパイル {r4['vi_parallel_compile']:.1f}） \\",
        r"\midrule",
        rf"\multicolumn{{3}}{{l}}{{1 デバイスでの比較（{res['n_base']} 銘柄 × {res['T_window']} 日、{res['warm_steps']:,} ステップ）}} \\",
        rf"提案 VI & 1 銘柄ずつ（Python のループ） & {res['single_device_loop']:.2f} \\",
        rf" & lax.map（1 銘柄ずつ） & {res['single_device_map']:.2f} \\",
        rf" & vmap（ベクトル化） & {res['single_device_vmap']:.2f} \\",
        r"\midrule",
        rf"\multicolumn{{3}}{{l}}{{日次の更新（{res['T_window']} 日の窓、{res['n_devices']} デバイスに振り分け）}} \\",
    ]  # fmt: skip
    for N in N_ASSETS:
        x = res[f"N{N}"]
        lines.append(rf"{N} 銘柄 & ゼロから（10,000 ステップ） & {x['cold_run']:.1f} \\")
        lines.append(
            rf" & warm start（{res['warm_steps']:,} ステップ） & {x['warm_median']:.2f} \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (common.TABLES / "tab_multi_scaling.tex").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# 監視の図（R2 のデータから）
# ---------------------------------------------------------------------------


def fig_monitoring():
    """図：ローリング推定の φ・σ（提案 VI の warm start は毎日、NUTS は 5 日ごと）。"""
    fc = pd.read_csv(common.RESULTS / "r2" / "forecasts.csv", dtype={"code": str})
    dates_all = pd.read_csv(common.RESULTS / "r1" / "dates_13210_all.csv").iloc[:, 0].tolist()
    exit_day = pd.Timestamp(
        dates_all[dates_all.index("2025-04-10") + W + 1]
    )  # 4/10 が窓から外れる日
    plt = plotting.plt
    fig, axes = plt.subplots(
        2, 2, figsize=(plotting.TEXT_WIDTH_IN, 3.6), sharex=True, layout="constrained"
    )
    for j, code in enumerate(ETFS):
        d = fc[fc["code"] == code]
        w = d[d["method"] == "SV-cond(warm)"].sort_values("date")
        n = d[d["method"] == "NUTS"].sort_values("date")
        for i, (par, tex) in enumerate((("phi", r"$\phi$"), ("sigma", r"$\sigma$"))):
            ax = axes[i, j]
            ax.plot(pd.to_datetime(w["date"]), w[par], color=plotting.color("ADVI-conditional"), lw=1.0,
                    label="VI: conditional (proposed), warm start, daily")  # fmt: skip
            ax.plot(pd.to_datetime(n["date"]), n[par], ls="none", marker="o", ms=3,
                    color=plotting.color("NUTS"), label="NUTS (every 5 days)")  # fmt: skip
            ax.axvline(exit_day, color=plotting.INK_SECONDARY, lw=0.8, ls="--")
            ax.set_ylabel(f"posterior mean of {tex}")
            if i == 0:
                ax.set_title(["1321 (Nikkei 225 ETF)", "1306 (TOPIX ETF)"][j], loc="left")
    for ax in axes[1]:
        ax.tick_params(axis="x", labelrotation=30)
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=7)
    return plotting.save(fig, "fig7_monitoring")


def main():
    args = common.parse_args(__doc__)
    out = common.RESULTS / "r5"
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    plotting.setup()
    fig_monitoring()
    ws = run_warm_steps(args, out)
    s = summarize_warm_steps(ws, out)
    table_warm_steps(s)
    print(s.to_string(float_format=lambda v: f"{v:.4f}"))
    res = run_multi(args, out)
    table_multi(res)
    common.write_env(out, args, common.RunConfig(), started)


if __name__ == "__main__":
    main()
