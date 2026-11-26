"""実験 R2（＋R3）：ローリング推定による 1 期先 VaR のバックテストと、warm start の効果。

各 ETF について、直近 W=250 日のリターンで推定し、翌日の予測分布から 5%・1% VaR と対数予測スコアを出す。
これを窓を 1 日ずつ進めながら繰り返す（検証期間は約 235 日）。窓内の平均を引き、翌日のリターンにも同じ平均を使う。

手法
- SV-cond(warm) : 提案 VI・基本SV。前日の変分パラメータを 1 日ずらして初期値にし、2,000 ステップ（R3 の主役）
- SV-cond(cold) : 提案 VI・基本SV。毎日ゼロから 10,000 ステップ（warm start と比べる）
- SVlev-cond    : 提案 VI・レバレッジSV（warm）
- SVt-cond      : 提案 VI・SV-t（warm）
- SV-meanfield  : 平均場 VI・基本SV（warm）
- GARCH-n / GARCH-t : GARCH(1,1)（平均 0、正規 / t 誤差）。arch パッケージで最尤推定
- NUTS          : 基本SV。計算量の都合で 5 日に 1 回だけ

出力：results/r2/forecasts.csv、表3、図A2
"""

import time

import common
import jax
import numpy as np
import pandas as pd
from arch import arch_model
from r1_full_period import ETF_JA, ETFS, load_returns
from scipy import stats

from sv_vi import advi, metrics, plotting
from sv_vi.diagnostics import convergence_step
from sv_vi.inference import run_advi
from sv_vi.logdensity import ModelSpec
from sv_vi.predictive import forecast

W = 250
NUTS_EVERY = 5
VI_METHODS = {
    # 名前: (ModelSpec, 変分族, warm start するか)
    "SV-cond(warm)": (ModelSpec(), "conditional", True),
    "SV-cond(cold)": (ModelSpec(), "conditional", False),
    "SVlev-cond": (ModelSpec(leverage=True), "conditional", True),
    "SVt-cond": (ModelSpec(student_t=True), "conditional", True),
    "SV-meanfield": (ModelSpec(), "meanfield", True),
}
METHOD_JA = {
    "SV-cond(warm)": "SV・提案VI（warm start）",
    "SV-cond(cold)": "SV・提案VI（毎回ゼロから）",
    "SVlev-cond": "レバレッジSV・提案VI",
    "SVt-cond": "SV-$t$・提案VI",
    "SV-meanfield": "SV・平均場VI",
    "GARCH-n": "GARCH(1,1)・正規",
    "GARCH-t": "GARCH(1,1)・$t$",
    "NUTS": "SV・NUTS（5日ごと）",
}


def extra_args(p):
    p.add_argument("--warm-steps", type=int, default=2000)
    p.add_argument("--cold-steps", type=int, default=10_000)


def garch_forecast(yw: np.ndarray, y_obs: float, dist: str) -> dict:
    t0 = time.perf_counter()
    res = arch_model(yw, mean="Zero", vol="GARCH", p=1, q=1, dist=dist, rescale=False).fit(
        disp="off"
    )
    var = float(res.forecast(horizon=1, reindex=False).variance.values[-1, 0])
    sd = np.sqrt(var)
    if dist == "t":
        nu = float(res.params["nu"])
        comp = stats.t(df=nu, scale=sd * np.sqrt((nu - 2) / nu))
    else:
        comp = stats.norm(scale=sd)
    return {
        "var1": float(comp.ppf(0.01)),
        "var5": float(comp.ppf(0.05)),
        "log_score": float(comp.logpdf(y_obs)),
        "pred_sd": sd,
        "time": time.perf_counter() - t0,
    }


def run_etf(code: str, ci: int, args, cfg, out, done: pd.DataFrame) -> list[dict]:
    # load_returns は全期間平均を引いているが、窓ごとに窓平均を引き直すので定数のずれは打ち消され、先読みにならない
    y_all = load_returns(code)
    raw = y_all.values
    dates = y_all.index
    n_days = len(raw) - W
    if args.quick:
        n_days = min(n_days, 12)
    rows = []
    prev_params = {m: None for m in VI_METHODS}
    for d in range(n_days):
        t = W + d
        if ((done["code"] == code) & (done["date"] == str(dates[t].date()))).any():
            continue  # 再開時：この日は済んでいる（warm start の連続性は途切れるが cold で再開される）
        yw = raw[t - W : t]
        m = yw.mean()
        yw = yw - m
        y_obs = raw[t] - m
        base = {"code": code, "date": str(dates[t].date()), "y_obs": y_obs}
        for j, (name, (spec, family, warm)) in enumerate(VI_METHODS.items()):
            key = common.data_key(args.seed, 20, ci, d, j)
            p0 = prev_params[name]
            use_warm = warm and p0 is not None
            steps = args.warm_steps if use_warm else args.cold_steps
            if args.quick:
                steps = min(steps, 500)
            config = advi.ADVIConfig(family=family, num_steps=steps)
            params0 = advi.shift_params(p0, family, spec) if use_warm else None
            r = run_advi(
                yw, spec, key, config, num_post=1000, khat=False, params0=params0, aot=False
            )
            prev_params[name] = r.params
            f = forecast(r, yw[-1], y_obs, spec, seed=int(t * 100 + j))
            rows.append(base | f | {
                "method": name, "time": r.time_total, "steps": steps, "warm": use_warm,
                "conv_step": convergence_step(r.elbo_trace),
                "elbo_final": float(np.mean(r.elbo_trace[-200:])),
                "phi": float(r.theta["phi"].mean()), "sigma": float(r.theta["sigma"].mean()),
            })  # fmt: skip
        for dist, name in (("normal", "GARCH-n"), ("t", "GARCH-t")):
            rows.append(base | garch_forecast(yw, y_obs, dist) | {"method": name})
        if d % NUTS_EVERY == 0:
            key = common.data_key(args.seed, 21, ci, d)
            r = common.run_method(
                "NUTS", yw, ModelSpec(), key, cfg, num_warmup=1000, num_samples=1000
            )
            f = forecast(r, yw[-1], y_obs, ModelSpec(), seed=int(t * 100 + 99))
            rows.append(base | f | {"method": "NUTS", "time": r.time_total,
                                    "phi": float(r.theta["phi"].mean()),
                                    "sigma": float(r.theta["sigma"].mean())})  # fmt: skip
        if d % 10 == 0:
            print(f"{code} day {d + 1}/{n_days} ({dates[t].date()})", flush=True)
            jax.clear_caches() if d % 50 == 0 else None
    return rows


def backtest_table(fc: pd.DataFrame, out) -> pd.DataFrame:
    """表3：手法ごとの VaR 超過率・検定の p 値・平均対数予測スコア・GARCH-t との差の DM 検定・1 日あたりの推定時間。

    NUTS は 5 日に 1 回しか回していない（評価日が違う）ので、この表には入れず、nuts_days_table で同じ日どうしを比べる。
    """
    recs = []
    for code in ETFS:
        ref = fc[(fc["code"] == code) & (fc["method"] == "GARCH-t")].set_index("date")["log_score"]
        for name in METHOD_JA:
            d = fc[(fc["code"] == code) & (fc["method"] == name)].sort_values("date")
            if d.empty:
                continue
            rec = {"code": code, "method": name, "n": len(d)}
            for a, col in ((0.05, "var5"), (0.01, "var1")):
                hits = (d["y_obs"] < d[col]).values
                k = metrics.kupiec_pof(hits, a)
                c = metrics.christoffersen(hits, a)
                rec |= {f"rate{col[3:]}": k["rate"], f"exceed{col[3:]}": k["exceed"],
                        f"p_pof{col[3:]}": k["p"], f"p_ind{col[3:]}": c["p_ind"],
                        f"p_cc{col[3:]}": c["p_cc"]}  # fmt: skip
            rec["log_score"] = d["log_score"].mean()
            if name != "GARCH-t":  # GARCH-t 自身との差は 0 なので検定しない
                diff = d.set_index("date")["log_score"] - ref.reindex(d["date"]).values
                dm = metrics.diebold_mariano(diff.values)
                rec |= {"dm_diff": dm["mean"], "dm_p": dm["p"], "dm_lags": dm["lags"]}
            # warm start の効果を見るため、最初の 1 日（コンパイルを含む）を除いた中央値
            rec["time_median"] = d["time"].iloc[1:].median() if len(d) > 1 else d["time"].median()
            recs.append(rec)
    tab = pd.DataFrame(recs)
    tab.to_csv(out / "backtest.csv", index=False)

    lines = [
        r"\begin{tabular}{lrrrrrrrrr}",
        r"\toprule",
        r" & \multicolumn{3}{c}{5\% VaR} & \multicolumn{2}{c}{1\% VaR} & & \multicolumn{2}{c}{対 GARCH-$t$} & \\",
        r"\cmidrule(lr){2-4} \cmidrule(lr){5-6} \cmidrule(lr){8-9}",
        r"手法 & 超過率（回） & $p_{\mathrm{POF}}$ & $p_{\mathrm{CC}}$ & 超過率（回） & $p_{\mathrm{POF}}$ "
        r"& 予測スコア & 差 & $p_{\mathrm{DM}}$ & 時間[s/日] \\",
    ]  # fmt: skip
    for code in ETFS:
        lines += [r"\midrule", rf"\multicolumn{{10}}{{l}}{{{ETF_JA[code]}}} \\"]
        rows = tab[(tab["code"] == code) & (tab["method"] != "NUTS")]
        for _, r in rows.iterrows():
            dm = (
                ("--", "--")
                if r["method"] == "GARCH-t"
                else (f"{r['dm_diff']:+.3f}", f"{r['dm_p']:.2f}")
            )
            lines.append(
                f"{METHOD_JA[r['method']]} & {r['rate5']:.3f} ({r['exceed5']}) & {r['p_pof5']:.2f} & "
                f"{r['p_cc5']:.2f} & {r['rate1']:.3f} ({r['exceed1']}) & {r['p_pof1']:.2f} & "
                f"{r['log_score']:.3f} & {dm[0]} & {dm[1]} & {r['time_median']:.2f} \\\\"
            )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (common.TABLES / "tab3_var_backtest.tex").write_text("\n".join(lines) + "\n")
    return tab


NUTS_DAY_METHODS = ["NUTS", "SV-cond(warm)", "SV-cond(cold)", "SV-meanfield", "GARCH-t"]


def nuts_days_table(fc: pd.DataFrame, out) -> pd.DataFrame:
    """表3b：NUTS を回した日（5 日に 1 回）だけで、各手法を同じ日どうしで比べる。"""
    recs = []
    for code in ETFS:
        d = fc[fc["code"] == code]
        days = d[d["method"] == "NUTS"]["date"]
        v = d.pivot(index="date", columns="method", values="var5").loc[days]
        s = d.pivot(index="date", columns="method", values="log_score").loc[days]
        y = d.groupby("date")["y_obs"].first().loc[days]
        for m in NUTS_DAY_METHODS:
            recs.append({
                "code": code, "method": m, "n": len(days),
                "exceed5": int((y < v[m]).sum()), "log_score": s[m].mean(),
                "var5_mean": v[m].mean(),
                "var5_absdiff_nuts": np.nan if m == "NUTS" else (v[m] - v["NUTS"]).abs().median(),
            })  # fmt: skip
    tab = pd.DataFrame(recs)
    tab.to_csv(out / "nuts_days.csv", index=False)
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"手法 & 5\% VaR 超過（回） & 予測スコア & 5\% VaR の平均 & NUTS との VaR の差 \\",
    ]
    for code in ETFS:
        sub = tab[tab["code"] == code]
        lines += [
            r"\midrule",
            rf"\multicolumn{{5}}{{l}}{{{ETF_JA[code]}（{sub['n'].iloc[0]} 日）}} \\",
        ]
        for _, r in sub.iterrows():
            diff = "--" if pd.isna(r["var5_absdiff_nuts"]) else f"{r['var5_absdiff_nuts']:.3f}"
            name = "SV・NUTS" if r["method"] == "NUTS" else METHOD_JA[r["method"]]
            lines.append(
                f"{name} & {r['exceed5']} & {r['log_score']:.3f} & {r['var5_mean']:.2f} & {diff} \\\\"
            )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (common.TABLES / "tab3b_nuts_days.tex").write_text("\n".join(lines) + "\n")
    return tab


def warm_vs_cold(fc: pd.DataFrame, out) -> pd.DataFrame:
    """R3：warm start と cold start の比較（時間・収束ステップ・VaR の差）、NUTS との VaR の差。"""
    recs = []
    for code in ETFS:
        d = fc[fc["code"] == code]
        w = d[d["method"] == "SV-cond(warm)"].set_index("date")
        c = d[d["method"] == "SV-cond(cold)"].set_index("date")
        n = d[d["method"] == "NUTS"].set_index("date")
        both = w.index.intersection(c.index)[1:]
        with_nuts = w.index.intersection(n.index)
        recs.append({
            "code": code,
            "warm_time_median": w.loc[both, "time"].median(),
            "cold_time_median": c.loc[both, "time"].median(),
            "warm_conv_median": w.loc[both, "conv_step"].median(),
            "cold_conv_median": c.loc[both, "conv_step"].median(),
            "var5_absdiff_warm_cold_median": (w.loc[both, "var5"] - c.loc[both, "var5"]).abs().median(),
            "var5_absdiff_vi_nuts_median": (w.loc[with_nuts, "var5"] - n.loc[with_nuts, "var5"]).abs().median(),
            "var5_nuts_mean": n.loc[with_nuts, "var5"].mean(),
            "nuts_time_median": n["time"].median(),
        })  # fmt: skip
    df = pd.DataFrame(recs)
    df.to_csv(out / "warm_vs_cold.csv", index=False)
    return df


def fig_var(fc: pd.DataFrame):
    """図A2：実現リターンと 5% VaR（提案 VI と GARCH-t）。超過日に印。"""
    plt = plotting.plt
    fig, axes = plt.subplots(
        2, 1, figsize=(plotting.TEXT_WIDTH_IN, 4.0), sharex=True, layout="constrained"
    )
    styles = {"SV-cond(warm)": ("SV, proposed VI", plotting.color("ADVI-conditional"), "-"),
              "GARCH-t": ("GARCH(1,1)-t", plotting.color("ADVI-structured"), "--")}  # fmt: skip
    for ax, code in zip(axes, ETFS, strict=True):
        d = fc[fc["code"] == code]
        base = d[d["method"] == "SV-cond(warm)"].sort_values("date")
        dates = pd.to_datetime(base["date"])
        ax.vlines(
            dates, 0, base["y_obs"], color=plotting.INK_SECONDARY, lw=0.6, label="realized return"
        )
        for name, (lab, col, ls) in styles.items():
            s = d[d["method"] == name].sort_values("date")
            ax.plot(
                pd.to_datetime(s["date"]),
                s["var5"],
                color=col,
                ls=ls,
                lw=1.1,
                label=f"5% VaR: {lab}",
            )
        hit = base["y_obs"] < base["var5"]
        ax.plot(dates[hit.values], base["y_obs"][hit.values], ls="none", marker="o", ms=4,
                mfc="none", mec=plotting.color("ADVI-conditional"), label="exceedance (proposed VI)")  # fmt: skip
        ax.set_title(ETFS[code], loc="left")
        ax.set_ylabel("return (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=7)
    return plotting.save(fig, "figA2_var_backtest")


def main():
    args = common.parse_args(__doc__, extra_args)
    cfg = common.RunConfig.quick() if args.quick else common.RunConfig()
    out = common.RESULTS / ("r2_quick" if args.quick else "r2")
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    csv = out / "forecasts.csv"
    fc = (
        pd.read_csv(csv, dtype={"code": str})
        if csv.exists()
        else pd.DataFrame(columns=["code", "date", "method"])
    )
    for ci, code in enumerate(ETFS):
        new = run_etf(code, ci, args, cfg, out, fc)
        if new:
            fc = pd.concat([fc, pd.DataFrame(new)], ignore_index=True)
            fc.to_csv(csv, index=False)
    tab = backtest_table(fc, out)
    nuts_days_table(fc, out)
    wc = warm_vs_cold(fc, out)
    plotting.setup()
    fig_var(fc)
    common.write_env(out, args, cfg, started)
    print(tab.to_string(float_format=lambda v: f"{v:.3f}"))
    print(wc.T.to_string())


if __name__ == "__main__":
    main()
