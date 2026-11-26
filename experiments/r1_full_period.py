"""実験 R1：ETF（1321・1306）の全期間推定。

J-Quants 無料プランで取得できる全期間（約 2 年、T≈485）の % 対数リターン（平均除去）に対し、
基本SV・レバレッジSV・SV-t を NUTS と各 VI で推定する。分配落ち日のリターンを除いた場合の感度も見る。

出力：results/r1/（rows.csv、各推定の npz）、図5、表2
"""

import time

import common
import jax
import numpy as np
import pandas as pd

from sv_vi import plotting
from sv_vi.data import cached_daily_bars, to_returns
from sv_vi.logdensity import ModelSpec

ETFS = {"13210": "1321 (Nikkei 225 ETF)", "13060": "1306 (TOPIX ETF)"}
ETF_JA = {"13210": "1321（日経225連動）", "13060": "1306（TOPIX連動）"}
MODELS = {
    "basic": ModelSpec(),
    "leverage": ModelSpec(leverage=True),
    "student_t": ModelSpec(student_t=True),
}
MODEL_JA = {"basic": "基本SV", "leverage": "レバレッジSV", "student_t": "SV-$t$"}
METHODS = ["NUTS", "ADVI-conditional", "ADVI-structured", "ADVI-meanfield"]


def load_returns(code: str, drop_dates=()) -> pd.Series:
    r = to_returns(cached_daily_bars(code))
    r = r.drop(pd.to_datetime(list(drop_dates)), errors="ignore")
    return r - r.mean()


# 分配落ち日（決算日の前営業日。1321 は 7/8 決算、1306 は 7/10 決算）。
# fetch_data.py の出力で、この日に −1.8〜−2.7% の下落があることを確認した（2026-11-23）。
EX_DIVIDEND_DATES = {
    "13210": ["2025-07-07", "2026-07-07"],
    "13060": ["2025-07-09", "2026-07-09"],
}


def ex_dividend_dates(code: str) -> list[str]:
    return EX_DIVIDEND_DATES[code]


def run(args, cfg, out):
    csv = out / "rows.csv"
    rows = (
        pd.read_csv(csv, dtype={"code": str})
        if csv.exists()
        else pd.DataFrame(columns=["code", "model", "method", "variant"])
    )
    for ci, code in enumerate(ETFS):
        variants = {"all": (), "ex_div_removed": ex_dividend_dates(code)}
        for variant, drop in variants.items():
            y = load_returns(code, drop)
            np.save(out / f"y_{code}_{variant}.npy", y.values)
            pd.Series(y.index).to_csv(out / f"dates_{code}_{variant}.csv", index=False)
            for mi, (model, spec) in enumerate(MODELS.items()):
                # 分配落ちの感度は基本SV × (NUTS, 提案 VI) だけ
                methods = METHODS if variant == "all" else ["NUTS", "ADVI-conditional"]
                if variant != "all" and model != "basic":
                    continue
                for k, m in enumerate(methods):
                    hit = (rows["code"] == code) & (rows["model"] == model) & (rows["method"] == m)
                    if (hit & (rows["variant"] == variant)).any():
                        continue
                    key = common.data_key(args.seed, 10, ci, mi, k, len(variant))
                    r = common.run_method(m, y.values, spec, key, cfg)
                    common.save_summary(r, out / f"{code}_{variant}_{model}_{m}.npz")
                    row = common.compare_row(r, {}, np.zeros(len(y)), None)
                    row |= {"code": code, "model": model, "variant": variant, "T": len(y)}
                    rows = pd.concat([rows, pd.DataFrame([row])], ignore_index=True)
                    rows.to_csv(csv, index=False)
                    th = r.theta_summary()
                    print(f"{code} {variant} {model} {m}: {r.time_total:.1f}s "
                          + " ".join(f"{k}={v['mean']:.3f}±{v['sd']:.3f}" for k, v in th.items()),
                          flush=True)  # fmt: skip
                    jax.clear_caches()
    return rows


def annual_vol(h):
    return np.sqrt(252.0) * np.exp(np.asarray(h) / 2.0)


def fig_vol(out):
    """図5：2 つの ETF のリターンと年率ボラ（NUTS の 95% 帯 ＋ 提案 VI の 95% 区間）。"""
    plt = plotting.plt
    fig, axes = plt.subplots(
        4, 1, figsize=(plotting.TEXT_WIDTH_IN, 5.4), sharex=True,
        gridspec_kw={"height_ratios": [0.55, 1, 0.55, 1], "hspace": 0.35},
    )  # fmt: skip
    for i, (code, title) in enumerate(ETFS.items()):
        dates = pd.to_datetime(pd.read_csv(out / f"dates_{code}_all.csv").iloc[:, 0])
        y = np.load(out / f"y_{code}_all.npy")
        ax_r, ax_v = axes[2 * i], axes[2 * i + 1]
        ax_r.vlines(dates, 0, y, color=plotting.INK_SECONDARY, lw=0.6)
        ax_r.set_ylabel("return (%)")
        ax_r.set_title(title, loc="left")
        nuts = common.load_summary(out / f"{code}_all_basic_NUTS.npz")
        vi = common.load_summary(out / f"{code}_all_basic_ADVI-conditional.npz")
        ax_v.fill_between(dates, annual_vol(nuts.h_q025), annual_vol(nuts.h_q975),
                          color=plotting.color("NUTS"), alpha=0.18, lw=0, label="NUTS 95% interval")  # fmt: skip
        ax_v.plot(
            dates,
            annual_vol(nuts.h_mean),
            color=plotting.color("NUTS"),
            lw=1.0,
            label="NUTS posterior mean",
        )
        m = "ADVI-conditional"
        for q, lab in ((vi.h_q025, None), (vi.h_q975, f"{plotting.label(m)}, 95% interval")):
            ax_v.plot(
                dates,
                annual_vol(q),
                color=plotting.color(m),
                ls=plotting.linestyle(m),
                lw=1.0,
                label=lab,
            )
        ax_v.set_ylabel("annualized vol (%)")
        if i == 0:
            ax_v.legend(loc="upper right", ncol=1)
    return plotting.save(fig, "fig5_etf_vol")


def table(rows: pd.DataFrame):
    """表2：事後平均（事後 sd）と計算時間。"""
    d = rows[rows["variant"] == "all"]
    params = ["mu", "phi", "sigma", "rho", "nu"]
    tex = {"mu": r"$\mu$", "phi": r"$\phi$", "sigma": r"$\sigma$", "rho": r"$\rho$", "nu": r"$\nu$"}
    ncol = 2 + len(params) + 1
    lines = [
        r"\begin{tabular}{ll" + "r" * (len(params) + 1) + "}",
        r"\toprule",
        "モデル & 手法 & " + " & ".join(tex[p] for p in params) + r" & 時間[s] \\",
    ]
    for code in ETFS:
        lines += [r"\midrule", rf"\multicolumn{{{ncol}}}{{l}}{{{ETF_JA[code]}}} \\"]
        for mi, model in enumerate(MODELS):
            sub = d[(d["code"] == code) & (d["model"] == model)].set_index("method")
            methods = METHODS if model == "basic" else ["NUTS", "ADVI-conditional"]
            if mi:
                lines.append(r"\addlinespace")
            for j, m in enumerate(methods):
                if m not in sub.index:
                    continue
                r = sub.loc[m]
                cells = []
                for p in params:
                    v = r.get(f"{p}_mean", np.nan)
                    fmt = ".1f" if p == "nu" else ".3f"
                    cells.append("" if pd.isna(v) else f"{v:{fmt}} ({r[f'{p}_sd']:{fmt}})")
                c1 = MODEL_JA[model] if j == 0 else ""
                lines.append(
                    f"{c1} & {plotting.label_ja(m)} & "
                    + " & ".join(cells)
                    + f" & {r['time_total']:.1f} \\\\"
                )
    lines += [r"\bottomrule", r"\end{tabular}"]
    common.TABLES.mkdir(parents=True, exist_ok=True)
    (common.TABLES / "tab2_etf_posterior.tex").write_text("\n".join(lines) + "\n")


def main():
    args = common.parse_args(__doc__)
    cfg = common.RunConfig.quick() if args.quick else common.RunConfig()
    out = common.RESULTS / ("r1_quick" if args.quick else "r1")
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    rows = run(args, cfg, out)
    plotting.setup()
    fig_vol(out)
    table(rows)
    common.write_env(out, args, cfg, started)
    cols = [
        "code",
        "variant",
        "model",
        "method",
        "mu_mean",
        "phi_mean",
        "phi_sd",
        "sigma_mean",
        "sigma_sd",
        "rho_mean",
        "nu_mean",
        "rhat_max",
        "time_total",
    ]
    print(rows[[c for c in cols if c in rows]].to_string(float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
