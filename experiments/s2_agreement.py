"""実験 S2：1 本のデータで、各 VI の事後分布を NUTS（長いチェーン）と比べる。

高持続シナリオ（φ=0.99, σ=0.10）、T=1000。平均場が最も苦しいケース。
出力：results/s2/（各手法の事後要約 npz, summary.csv）、図2・図3・図A1、表（付録用）
"""

import time

import common
import numpy as np
import pandas as pd

from sv_vi import plotting
from sv_vi.logdensity import ModelSpec
from sv_vi.simulate import SCENARIOS, simulate_sv

METHODS = [
    "NUTS",
    "ADVI-conditional",
    "ADVI-structured",
    "ADVI-meanfield",
    "SVI-structured",
    "SVI-meanfield",
    "SVI-lowrank",
    "SVI-fullrank",
]
T = 1000
SCENARIO = "high_persistence"


def run(args, cfg: common.RunConfig, out):
    params = SCENARIOS[SCENARIO]
    data = simulate_sv(common.data_key(args.seed, 2), T, params)
    np.savez(out / "data.npz", **data)
    spec = ModelSpec()
    for i, m in enumerate(METHODS):
        path = out / f"{m}.npz"
        if path.exists():
            continue
        key = common.data_key(args.seed, 2, 100 + i)
        if m == "NUTS":  # 正解として長めに回す（各チェーン 16 倍の長さを 8 つおきに保存）
            r = common.run_method(
                m, data["y"], spec, key, cfg,
                num_warmup=cfg.nuts_warmup, num_samples=16 * cfg.nuts_samples, thinning=8,
            )  # fmt: skip
        else:
            r = common.run_method(m, data["y"], spec, key, cfg, num_steps=2 * cfg.vi_steps)
        common.save_summary(r, path)
        print(f"{m}: {r.time_total:.1f}s", flush=True)


def summarize(out):
    data = np.load(out / "data.npz")
    truth = SCENARIOS[SCENARIO].as_dict()
    res = {m: common.load_summary(out / f"{m}.npz") for m in METHODS}
    ref = res["NUTS"]
    rows = [
        common.compare_row(r, truth, data["h"], ref if m != "NUTS" else None)
        for m, r in res.items()
    ]
    df = pd.DataFrame(rows)
    df.to_csv(out / "summary.csv", index=False)
    return data, res, df


# ---------------------------------------------------------------------------
# 図
# ---------------------------------------------------------------------------


def annual_vol(h):
    return np.sqrt(252.0) * np.exp(np.asarray(h) / 2.0)


def fig_bands(data, res):
    """図2：年率ボラの事後 95% 区間（NUTS vs 提案手法 / NUTS vs 平均場）と、事後 sd 比の推移。"""
    plt = plotting.plt
    t = np.arange(T)
    fig, axes = plt.subplots(
        3, 1, figsize=(plotting.TEXT_WIDTH_IN, 5.6), sharex=True,
        gridspec_kw={"height_ratios": [1, 1, 0.8], "hspace": 0.45},
    )  # fmt: skip
    ref = res["NUTS"]
    for ax, m, tag in zip(axes[:2], ["ADVI-conditional", "ADVI-meanfield"], "ab", strict=True):
        ax.fill_between(
            t, annual_vol(ref.h_q025), annual_vol(ref.h_q975),
            color=plotting.color("NUTS"), alpha=0.18, lw=0, label="NUTS 95% interval",
        )  # fmt: skip
        r = res[m]
        ax.plot(t, annual_vol(r.h_q025), color=plotting.color(m), lw=1.0, ls=plotting.linestyle(m))
        ax.plot(
            t, annual_vol(r.h_q975), color=plotting.color(m), lw=1.0, ls=plotting.linestyle(m),
            label=f"{plotting.label(m)}, 95% interval",
        )  # fmt: skip
        ax.plot(t, annual_vol(data["h"]), color=plotting.TRUTH, lw=0.8, label="true volatility")
        ax.set_ylabel("annualized vol (%)")
        ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3, borderaxespad=0.2)
        ax.text(-0.09, 1.08, f"({tag})", transform=ax.transAxes, fontweight="bold")
    ax = axes[2]
    for m in ["ADVI-conditional", "ADVI-structured", "ADVI-meanfield"]:
        ax.plot(
            t, res[m].h_sd / ref.h_sd, color=plotting.color(m), ls=plotting.linestyle(m),
            lw=1.0, label=plotting.label(m),
        )  # fmt: skip
    ax.axhline(1.0, color=plotting.INK_SECONDARY, lw=0.8)
    ax.set_ylabel("sd ratio of $h_t$\n(VI / NUTS)")
    ax.set_xlabel("day $t$")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3, borderaxespad=0.2)
    ax.text(-0.09, 1.08, "(c)", transform=ax.transAxes, fontweight="bold")
    ax.set_xlim(0, T - 1)
    return plotting.save(fig, "fig2_h_bands")


def _kde(x, grid):
    from scipy.stats import gaussian_kde

    return gaussian_kde(x)(grid)


def fig_marginals(res, truth):
    """図3：大域パラメータの周辺事後分布。平均場など範囲外のものは矢印と値で示す。"""
    plt = plotting.plt
    methods = ["NUTS", "ADVI-conditional", "ADVI-structured", "ADVI-meanfield"]
    names = [("mu", r"$\mu$"), ("phi", r"$\phi$"), ("sigma", r"$\sigma$")]
    notes = []
    fig, axes = plt.subplots(1, 3, figsize=(plotting.TEXT_WIDTH_IN, 2.0), layout="constrained")
    for ax, (k, tex) in zip(axes, names, strict=True):
        ref = res["NUTS"].theta[k]
        lo, hi = np.quantile(ref, [0.0005, 0.9995])
        pad = 0.25 * (hi - lo)
        lo, hi = lo - pad, hi + pad
        if k == "phi":
            hi = min(hi, 1.0)
        grid = np.linspace(lo, hi, 400)
        ymax = 1.5 * _kde(ref, grid).max()  # 鋭すぎるピークは切り詰める（キャプションで注記）
        offscale = []
        for m in methods:
            x = res[m].theta[k]
            if np.median(x) < lo or np.median(x) > hi:
                offscale.append((m, float(np.mean(x))))
                continue
            dens = _kde(x, grid)
            ax.plot(
                grid, dens, color=plotting.color(m), ls=plotting.linestyle(m),
                label=plotting.label(m),
            )  # fmt: skip
        ax.axvline(truth[k], color=plotting.TRUTH, lw=0.8, ls="--", label="true value")
        for m, v in offscale:
            notes.append(
                f"{tex} = {v:.2f}" + ("" if m == "ADVI-meanfield" else f" ({plotting.label(m)})")
            )
        ax.set_xlim(lo, hi)
        ax.set_ylim(0, ymax)
        ax.set_yticks([])
        ax.set_xlabel(tex, labelpad=1)
    handles, labels = axes[0].get_legend_handles_labels()
    for ax in axes[1:]:
        for h, lab in zip(*ax.get_legend_handles_labels(), strict=True):
            if lab not in labels:
                handles.append(h)
                labels.append(lab)
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.02))
    if notes:
        fig.text(
            0.5, -0.3, "Off scale — VI: mean-field posterior mean " + ", ".join(notes),
            ha="center", va="top", fontsize=7, color=plotting.INK_SECONDARY,
        )  # fmt: skip
    return plotting.save(fig, "fig3_theta_marginals")


def fig_elbo(res):
    """図A1：ELBO の推移（200 ステップの移動平均）。"""
    plt = plotting.plt
    fig, ax = plt.subplots(figsize=(plotting.TEXT_WIDTH_IN, 2.4))
    best = max(np.mean(r.elbo_trace[-500:]) for m, r in res.items() if m != "NUTS")
    for m, r in res.items():
        if r.elbo_trace is None:
            continue
        w = 200
        sm = np.convolve(r.elbo_trace, np.ones(w) / w, mode="valid")
        ax.plot(np.arange(len(sm)) + w, sm, color=plotting.color(m), ls=plotting.linestyle(m),
                lw=1.0, label=plotting.label(m))  # fmt: skip
    ax.set_ylim(best - 300, best + 20)
    ax.set_xlabel("optimization step")
    ax.set_ylabel("ELBO (moving avg.)")
    ax.legend(loc="lower right", ncol=2)
    return plotting.save(fig, "figA1_elbo")


def table(df):
    """付録用の表：NUTS との一致度。"""
    order = [m for m in METHODS if m != "NUTS"]
    d = df.set_index("method").loc[["NUTS"] + order]
    lines = [
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r"手法 & $\phi$ & $\sigma$ & $\mathrm{sd}(\phi)$比 & $\mathrm{sd}(\sigma)$比 & $\mathrm{sd}(h)$比 & ELBO & $\hat k$ & 時間[s] \\",
        r"\midrule",
    ]
    for m, r in d.iterrows():

        def f(key, fmt, r=r):
            v = r.get(key, np.nan)
            return "--" if pd.isna(v) else format(v, fmt)

        lines.append(
            f"{plotting.label(m)} & {f('phi_mean', '.3f')} & {f('sigma_mean', '.3f')} & "
            f"{f('phi_sd_ratio', '.2f')} & {f('sigma_sd_ratio', '.2f')} & {f('h_sd_ratio', '.2f')} & "
            f"{f('elbo_final', '.1f')} & {f('khat', '.2f')} & {f('time_total', '.1f')} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    common.TABLES.mkdir(parents=True, exist_ok=True)
    (common.TABLES / "tabA2_agreement.tex").write_text("\n".join(lines) + "\n")


def main():
    args = common.parse_args(__doc__)
    cfg = common.RunConfig.quick() if args.quick else common.RunConfig()
    out = common.RESULTS / ("s2_quick" if args.quick else "s2")
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    run(args, cfg, out)
    data, res, df = summarize(out)
    plotting.setup()
    fig_bands(data, res)
    fig_marginals(res, SCENARIOS[SCENARIO].as_dict())
    fig_elbo(res)
    table(df)
    common.write_env(out, args, cfg, started)
    cols = [
        "method",
        "phi_mean",
        "sigma_mean",
        "mu_mean",
        "phi_sd_ratio",
        "sigma_sd_ratio",
        "h_sd_ratio",
        "h_rmse",
        "h_cover",
        "khat",
        "elbo_final",
        "time_total",
        "conv_step",
    ]
    print(df[[c for c in cols if c in df]].to_string(float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
