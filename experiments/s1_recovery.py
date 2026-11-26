"""実験 S1：パラメータと潜在ボラの復元精度（複製データで評価）。

シナリオ × T × 複製 R 本のデータを生成し、各手法で推定して真値と比べる。
基本SV は 3 シナリオ（標準・高持続・低持続）、拡張モデル（レバレッジ・t 誤差）は標準パラメータで行う。
途中で止めても、results/s1/rows.csv に済んだ分が残り、再実行すると続きから回る。

出力：results/s1/rows.csv、表1（基本SV）、表1b（拡張モデル）
"""

import time

import common
import jax
import numpy as np
import pandas as pd

from sv_vi import plotting
from sv_vi.logdensity import ModelSpec
from sv_vi.simulate import SCENARIOS, simulate_sv

METHODS = ["NUTS", "ADVI-conditional", "ADVI-structured", "ADVI-meanfield"]
BASIC = ["standard", "high_persistence", "low_persistence"]
EXTENDED = ["leverage", "student_t"]
SCENARIO_ID = {s: i for i, s in enumerate(BASIC + EXTENDED)}
SCENARIO_LABEL = {
    "standard": r"標準 ($\phi=0.97$)",
    "high_persistence": r"高持続 ($\phi=0.99$)",
    "low_persistence": r"低持続 ($\phi=0.80$)",
    "leverage": r"レバレッジ ($\rho=-0.5$)",
    "student_t": r"$t$ 誤差 ($\nu=8$)",
}


def spec_for(scenario: str) -> ModelSpec:
    return ModelSpec(leverage=scenario == "leverage", student_t=scenario == "student_t")


def extra_args(p):
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--Ts", type=int, nargs="+", default=[500, 2000])
    p.add_argument("--scenarios", nargs="+", default=BASIC + EXTENDED)


def run(args, cfg, out):
    csv = out / "rows.csv"
    rows = pd.read_csv(csv) if csv.exists() else pd.DataFrame(columns=["scenario", "T", "rep"])
    for scenario in args.scenarios:
        params, spec = SCENARIOS[scenario], spec_for(scenario)
        for T in args.Ts:
            for rep in range(args.reps):
                hit = (rows["scenario"] == scenario) & (rows["T"] == T) & (rows["rep"] == rep)
                if hit.any():
                    continue
                ids = (1, SCENARIO_ID[scenario], T, rep)
                data = simulate_sv(common.data_key(args.seed, *ids), T, params)
                t0 = time.time()
                results = {
                    m: common.run_method(m, data["y"], spec, common.data_key(args.seed, *ids, i), cfg)
                    for i, m in enumerate(METHODS)
                }  # fmt: skip
                new = []
                for m, r in results.items():
                    ref = results["NUTS"] if m != "NUTS" else None
                    row = common.compare_row(r, params.as_dict(), data["h"], ref)
                    new.append(row | {"scenario": scenario, "T": T, "rep": rep})
                rows = pd.concat([rows, pd.DataFrame(new)], ignore_index=True)
                rows.to_csv(csv, index=False)
                nuts = results["NUTS"].extra
                print(
                    f"{scenario} T={T} rep={rep}: {time.time() - t0:.0f}s "
                    f"(NUTS rhat_max={max(nuts['rhat'].values()):.3f}, div={nuts['divergences']})",
                    flush=True,
                )
                del results
                jax.clear_caches()
    return rows


def aggregate(rows: pd.DataFrame) -> pd.DataFrame:
    """シナリオ × T × 手法ごとに集計する。"""
    rows = rows.copy()
    params = [c[:-5] for c in rows.columns if c.endswith("_true")]
    for k in params:
        rows[f"{k}_err"] = rows[f"{k}_mean"] - rows[f"{k}_true"]
        rows[f"{k}_cov"] = (rows[f"{k}_q025"] <= rows[f"{k}_true"]) & (
            rows[f"{k}_true"] <= rows[f"{k}_q975"]
        )
    agg = {}
    for k in params:
        agg[f"{k}_bias"] = (f"{k}_err", "mean")
        agg[f"{k}_rmse"] = (f"{k}_err", lambda e: float(np.sqrt(np.nanmean(e**2))))
        agg[f"{k}_cover"] = (f"{k}_cov", "mean")
        if f"{k}_sd_ratio" in rows:
            agg[f"{k}_sd_ratio"] = (f"{k}_sd_ratio", "median")
    agg |= {
        "h_rmse": ("h_rmse", "mean"),
        "h_cover": ("h_cover", "mean"),
        "h_sd_ratio": ("h_sd_ratio", "median"),
        "time": ("time_total", "median"),
        "n": ("rep", "count"),
    }
    agg = {k: v for k, v in agg.items() if v[0] in rows}
    return rows.groupby(["scenario", "T", "method"], sort=False).agg(**agg).reset_index()


def _fmt(v, f):
    return "--" if pd.isna(v) else format(v, f)


def table_basic(agg: pd.DataFrame, scenarios, name: str, extra_param: str | None = None):
    """表：シナリオ × T × 手法ごとの RMSE と 95% 区間の被覆率、h の指標、計算時間。"""
    cols = ["phi", "sigma"] + ([extra_param] if extra_param else [])
    tex = {"phi": r"$\phi$", "sigma": r"$\sigma$", "rho": r"$\rho$", "nu": r"$\nu$"}
    head = " & ".join(rf"\multicolumn{{2}}{{c}}{{{tex[c]}}}" for c in cols)
    sub = " & ".join(["RMSE & 被覆"] * len(cols))
    n_num = 2 * len(cols) + 4
    lines = [
        r"\begin{tabular}{ll" + "r" * n_num + "}",
        r"\toprule",
        rf" & & {head} & \multicolumn{{3}}{{c}}{{$h_t$}} & \\",
        rf"$T$ & 手法 & {sub} & RMSE & 被覆 & sd比 & 時間[s] \\",
    ]
    for scenario in scenarios:
        d = agg[agg["scenario"] == scenario]
        if d.empty:
            continue
        lines += [
            r"\midrule",
            rf"\multicolumn{{{n_num + 2}}}{{l}}{{{SCENARIO_LABEL[scenario]}}} \\",
        ]
        for T, dt in d.groupby("T", sort=True):
            for i, (_, r) in enumerate(
                dt.set_index("method").loc[METHODS].reset_index().iterrows()
            ):
                cells = []
                for c in cols:
                    cells += [_fmt(r.get(f"{c}_rmse"), ".3f"), _fmt(r.get(f"{c}_cover"), ".2f")]
                cells += [
                    _fmt(r["h_rmse"], ".3f"), _fmt(r["h_cover"], ".2f"),
                    _fmt(r.get("h_sd_ratio"), ".2f"), _fmt(r["time"], ".1f"),
                ]  # fmt: skip
                t_cell = str(T) if i == 0 else ""
                lines.append(
                    f"{t_cell} & {plotting.label_ja(r['method'])} & " + " & ".join(cells) + r" \\"
                )
    lines += [r"\bottomrule", r"\end{tabular}"]
    common.TABLES.mkdir(parents=True, exist_ok=True)
    (common.TABLES / f"{name}.tex").write_text("\n".join(lines) + "\n")


def main():
    args = common.parse_args(__doc__, extra_args)
    cfg = common.RunConfig.quick() if args.quick else common.RunConfig()
    if args.quick:
        args.reps, args.Ts = 2, [300]
    out = common.RESULTS / ("s1_quick" if args.quick else "s1")
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    rows = run(args, cfg, out)
    agg = aggregate(rows)
    agg.to_csv(out / "aggregate.csv", index=False)
    table_basic(agg, BASIC, "tab1_recovery")
    table_basic(agg[agg["scenario"] == "leverage"], ["leverage"], "tab1b_leverage", "rho")
    table_basic(agg[agg["scenario"] == "student_t"], ["student_t"], "tab1c_student_t", "nu")
    common.write_env(out, args, cfg, started)
    show = [
        "scenario",
        "T",
        "method",
        "phi_rmse",
        "phi_cover",
        "sigma_rmse",
        "sigma_cover",
        "h_rmse",
        "h_cover",
        "h_sd_ratio",
        "time",
        "n",
    ]
    print(agg[[c for c in show if c in agg]].to_string(float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
