"""実験 S3：系列長 T に対する計算時間のスケーリング。

標準シナリオで T を変え、各手法の所要時間を測る。
- NUTS：4 チェーン並列（warmup・sample 各 1000）の総時間と、最小 ESS/秒
- VI  ：ELBO が収束するまでの時間（JIT コンパイル時間 ＋ 実行時間 × 収束ステップ/総ステップ）

計測の公平性のため、この実験は他の重い処理と同時に走らせないこと。
出力：results/s3/timing.csv、図4、表（付録用）
"""

import time

import common
import jax
import pandas as pd

from sv_vi import plotting
from sv_vi.logdensity import ModelSpec
from sv_vi.simulate import SCENARIOS, simulate_sv

METHODS = ["NUTS", "ADVI-conditional", "ADVI-structured", "ADVI-meanfield", "SVI-meanfield"]


def extra_args(p):
    p.add_argument("--Ts", type=int, nargs="+", default=[250, 500, 1000, 2000, 5000, 10000])
    p.add_argument("--max-T-nuts", type=int, default=10000)


def run(args, cfg, out):
    csv = out / "timing.csv"
    done = pd.read_csv(csv) if csv.exists() else pd.DataFrame(columns=["T", "method"])
    spec = ModelSpec()
    for T in args.Ts:
        data = simulate_sv(common.data_key(args.seed, 3, T), T, SCENARIOS["standard"])
        for i, m in enumerate(METHODS):
            if ((done["T"] == T) & (done["method"] == m)).any():
                continue
            if m == "NUTS" and T > args.max_T_nuts:
                continue
            key = common.data_key(args.seed, 3, T, i)
            kw = {"num_warmup": 1000, "num_samples": 1000} if m == "NUTS" else {}
            r = common.run_method(m, data["y"], spec, key, cfg, khat=False, **kw)
            row = common.compare_row(r, SCENARIOS["standard"].as_dict(), data["h"], None)
            row["T"] = T
            done = pd.concat([done, pd.DataFrame([row])], ignore_index=True)
            done.to_csv(csv, index=False)
            print(f"T={T} {m}: total {r.time_total:.1f}s, conv {row['time_conv']:.1f}s", flush=True)
            del r
            jax.clear_caches()
    return done


def fig_scaling(df):
    """図4：所要時間と T（両対数）。"""
    plt = plotting.plt
    fig, ax = plt.subplots(figsize=(plotting.TEXT_WIDTH_IN * 0.62, 2.6))
    for m in [m for m in METHODS if not m.startswith("SVI-")]:  # NumPyro 版は表のみ
        d = df[df["method"] == m].sort_values("T")
        if d.empty:
            continue
        yv = d["time_total"] if m == "NUTS" else d["time_conv"]
        ax.plot(
            d["T"], yv, color=plotting.color(m), ls=plotting.linestyle(m),
            marker=plotting.marker(m), ms=4, label=plotting.label(m),
        )  # fmt: skip
        if m == "NUTS":  # NUTS は同じ時間でも有効サンプル数が T とともに激減することを注記
            for T, t, e in zip(
                d["T"], d["time_total"], d["min_ess_per_sec"] * d["time_total"], strict=True
            ):
                ax.annotate(f"ESS {e:.0f}", (T, t), xytext=(-4, 5), textcoords="offset points",
                            ha="right", fontsize=6.5, color=plotting.INK_SECONDARY)  # fmt: skip
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("series length $T$ (days)")
    ax.set_ylabel("wall-clock time (s)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    return plotting.save(fig, "fig4_scaling")


def table(df):
    """付録用：T ごとの所要時間 [s]。

    NUTS は全体の時間と θ の最小 ESS・最大 R̂、提案手法は「ELBO の収束まで」と「全ステップ」の両方、
    ほかの VI は収束までの時間。
    """
    d = df.set_index(["T", "method"])

    def cell(T, m, col, fmt):
        if (T, m) not in d.index:
            return "--"
        v = d.loc[(T, m), col]
        return "--" if pd.isna(v) else format(v, fmt)

    vi_cols = ["ADVI-structured", "ADVI-meanfield", "SVI-meanfield"]
    nuts, prop = plotting.label_ja("NUTS"), plotting.label_ja("ADVI-conditional")
    others = " & ".join(plotting.label_ja(m) for m in vi_cols)
    lines = [
        r"\begin{tabular}{rrrrrrrrr}",
        r"\toprule",
        rf" & \multicolumn{{3}}{{c}}{{{nuts}}} & \multicolumn{{2}}{{c}}{{{prop}}}"
        r" & \multicolumn{3}{c}{ほかの VI（収束まで）} \\",
        r"\cmidrule(lr){2-4} \cmidrule(lr){5-6} \cmidrule(lr){7-9}",
        rf"$T$ & 時間 & 最小 ESS & 最大 $\hat R$ & 収束まで & 全ステップ & {others} \\",
        r"\midrule",
    ]
    for T in sorted(df["T"].unique()):
        cells = [
            cell(T, "NUTS", "time_total", ".1f"),
            cell(T, "NUTS", "ess_min", ".0f"),
            cell(T, "NUTS", "rhat_max", ".2f"),
            cell(T, "ADVI-conditional", "time_conv", ".1f"),
            cell(T, "ADVI-conditional", "time_total", ".1f"),
        ] + [cell(T, m, "time_conv", ".1f") for m in vi_cols]
        lines.append(f"{T:,} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    common.TABLES.mkdir(parents=True, exist_ok=True)
    (common.TABLES / "tabA3_scaling.tex").write_text("\n".join(lines) + "\n")


def main():
    args = common.parse_args(__doc__, extra_args)
    cfg = common.RunConfig.quick() if args.quick else common.RunConfig()
    if args.quick:
        args.Ts = [250, 500]
    out = common.RESULTS / ("s3_quick" if args.quick else "s3")
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    df = run(args, cfg, out)
    plotting.setup()
    fig_scaling(df)
    table(df)
    common.write_env(out, args, cfg, started)
    cols = [
        "T",
        "method",
        "time_total",
        "time_compile",
        "time_conv",
        "conv_step",
        "min_ess_per_sec",
        "phi_mean",
        "sigma_mean",
    ]
    print(df[[c for c in cols if c in df]].to_string(float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
