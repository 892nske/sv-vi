"""2 章の図表：SV モデルの性質と事前分布。

- 図1 fig1_sv_illustration：標準シナリオの合成リターンと真の年率ボラ、y_t^2 の自己相関（理論値と標本値）
- 表 tab_sv_properties.tex：シナリオ・実データ推定値ごとの定常的な性質（尖度、半減期など）
- 表 tab_priors.tex：事前分布の平均と 95% 区間

実データの値は R1 の結果（results/r1/rows.csv、基本SV・NUTS・全期間）から読む。R1 が未実行なら実データの行は省く。
"""

import common
import jax
import numpy as np
import pandas as pd
from scipy import special, stats

from sv_vi import plotting
from sv_vi.logdensity import PRIOR
from sv_vi.simulate import SCENARIOS, simulate_sv

SCENARIO_JA = {
    "standard": "標準",
    "high_persistence": "高持続",
    "low_persistence": "低持続",
}
SCENARIO_EN = {
    "standard": "standard",
    "high_persistence": "high persistence",
    "low_persistence": "low persistence",
}
# シナリオは手法ではないので、手法用の色と重ならないように黒系＋線種で区別する
SCENARIO_LS = {"standard": "-", "high_persistence": "--", "low_persistence": ":"}
ETF_JA = {"13210": "1321", "13060": "1306"}


# ---------------------------------------------------------------------------
# 基本 SV モデルの定常的な性質（2 章の式）
# ---------------------------------------------------------------------------


def stationary_var(phi, sigma):
    """h_t の定常分散 s^2 = σ^2 / (1 - φ^2)。"""
    return sigma**2 / (1.0 - phi**2)


def kurtosis(phi, sigma):
    """y_t の尖度 3 exp(s^2)。"""
    return 3.0 * np.exp(stationary_var(phi, sigma))


def acf_sq_theory(phi, sigma, lags):
    """y_t^2 の自己相関 (exp(s^2 φ^k) - 1) / (3 exp(s^2) - 1)。"""
    s2 = stationary_var(phi, sigma)
    return np.expm1(s2 * phi ** np.asarray(lags)) / (3.0 * np.exp(s2) - 1.0)


def half_life(phi):
    """ボラのショックが半分になるまでの日数 log 0.5 / log φ。"""
    return np.log(0.5) / np.log(phi)


def annual_vol_band(mu, phi, sigma, q=0.025):
    """定常分布のもとでの年率ボラ √252 exp(h/2) の (q, 1-q) 分位点 [%]。"""
    s = np.sqrt(stationary_var(phi, sigma))
    z = stats.norm.ppf(1 - q)
    return tuple(np.sqrt(252.0) * np.exp((mu + c * z * s) / 2.0) for c in (-1, 1))


def sample_acf(x, lags):
    x = np.asarray(x) - np.mean(x)
    v = np.dot(x, x)
    return np.array([np.dot(x[:-k], x[k:]) / v for k in lags])


# ---------------------------------------------------------------------------
# 図・表
# ---------------------------------------------------------------------------


def fig_illustration(key, out):
    plt = plotting.plt
    k_short, k_long = jax.random.split(key)
    p = SCENARIOS["standard"]
    sim = simulate_sv(k_short, 1000, p)
    lags = np.arange(1, 101)
    long_y = simulate_sv(k_long, 200_000, p)["y"]
    acf_sample = sample_acf(long_y**2, lags)
    np.savez(out / "fig1_data.npz", y=sim["y"], h=sim["h"], acf_sample=acf_sample)

    fig = plt.figure(figsize=(plotting.TEXT_WIDTH_IN, 5.6))
    gs = fig.add_gridspec(3, 1, height_ratios=[0.8, 0.8, 1.1], hspace=0.75)
    ax_r = fig.add_subplot(gs[0])
    ax_v = fig.add_subplot(gs[1], sharex=ax_r)
    ax_a = fig.add_subplot(gs[2])
    t = np.arange(1, len(sim["y"]) + 1)
    ax_r.vlines(t, 0, sim["y"], color=plotting.INK_SECONDARY, lw=0.5)
    ax_r.set_ylabel("return (%)")
    ax_r.set_title(
        r"(a) simulated returns, standard scenario ($\mu=0,\ \phi=0.97,\ \sigma=0.15$)", loc="left"
    )
    ax_v.plot(t, np.sqrt(252.0) * np.exp(sim["h"] / 2.0), color=plotting.INK, lw=1.0)
    ax_v.set_ylabel("annualized vol (%)")
    ax_v.set_xlabel("day $t$")
    ax_v.set_title(r"(b) true annualized volatility $\sqrt{252}\,\exp(h_t/2)$", loc="left")

    for name, ls in SCENARIO_LS.items():
        s = SCENARIOS[name]
        ax_a.plot(
            lags,
            acf_sq_theory(s.phi, s.sigma, lags),
            color=plotting.INK,
            ls=ls,
            lw=1.2,
            label=f"theory: {SCENARIO_EN[name]} ($\\phi={s.phi}$, $\\sigma={s.sigma}$)",
        )
    ax_a.plot(
        lags[::3],
        acf_sample[::3],
        ls="none",
        marker="o",
        ms=3.5,
        mfc="white",
        mec=plotting.TRUTH,
        mew=0.9,
        label="sample ACF: standard, $T=200{,}000$",
    )
    ax_a.set_xlabel("lag $k$ (days)")
    ax_a.set_ylabel(r"autocorrelation of $y_t^2$")
    ax_a.set_title(r"(c) autocorrelation of squared returns", loc="left")
    ax_a.set_ylim(bottom=0)
    ax_a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2)
    return plotting.save(fig, "fig1_sv_illustration")


def real_estimates() -> list[tuple[str, dict]]:
    csv = common.RESULTS / "r1" / "rows.csv"
    if not csv.exists():
        return []
    rows = pd.read_csv(csv, dtype={"code": str})
    d = rows[(rows["model"] == "basic") & (rows["method"] == "NUTS") & (rows["variant"] == "all")]
    out = []
    for code, label in ETF_JA.items():
        r = d[d["code"] == code]
        if len(r):
            r = r.iloc[0]
            out.append((f"実データ：{label}", {k: r[f"{k}_mean"] for k in ("mu", "phi", "sigma")}))
    return out


def table_properties():
    items = [(f"合成：{SCENARIO_JA[n]}", SCENARIOS[n].as_dict()) for n in SCENARIO_JA]
    items += real_estimates()
    lines = [
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r" & $\mu$ & $\phi$ & $\sigma$ & $s$ & 尖度 & 半減期[日] & 年率ボラの95\%範囲[\%] \\",
        r"\midrule",
    ]
    for i, (name, p) in enumerate(items):
        mu, phi, sigma = p["mu"], p["phi"], p["sigma"]
        lo, hi = annual_vol_band(mu, phi, sigma)
        if i == len(SCENARIO_JA):
            lines.append(r"\addlinespace")
        lines.append(
            f"{name} & {mu:.2f} & {phi:.3f} & {sigma:.3f} & {np.sqrt(stationary_var(phi, sigma)):.3f}"
            f" & {kurtosis(phi, sigma):.2f} & {half_life(phi):.1f} & {lo:.1f}〜{hi:.1f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (common.TABLES / "tab_sv_properties.tex").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def table_priors():
    a, b = PRIOR["phi_a"], PRIOR["phi_b"]
    phi = stats.beta(a, b)
    ig_a, ig_b = PRIOR["ig_shape"], PRIOR["ig_rate"]
    ig = stats.invgamma(ig_a, scale=ig_b)
    hn = stats.halfnorm(scale=PRIOR["sigma_scale"])
    nu = stats.gamma(PRIOR["nu_shape"], scale=1.0 / PRIOR["nu_rate"])
    mu = stats.norm(PRIOR["mu_loc"], PRIOR["mu_scale"])

    def iv(lo, hi, f="{:.3f}"):
        return f"[{f.format(lo)},\\ {f.format(hi)}]"

    rows = [
        (
            r"$\mu$",
            r"$\Normal(0,\ 10^2)$",
            f"{mu.mean():.1f}",
            iv(*mu.ppf([0.025, 0.975]), "{:.1f}"),
        ),
        (
            r"$\phi$",
            r"$(\phi+1)/2 \sim \mathrm{Beta}(20,\ 1.5)$",
            f"{2 * phi.mean() - 1:.3f}",
            iv(*(2 * phi.ppf([0.025, 0.975]) - 1)),
        ),
        (
            r"$\sigma$",
            r"$\mathrm{HalfNormal}(1)$（標準）",
            f"{hn.mean():.3f}",
            iv(*hn.ppf([0.025, 0.975])),
        ),
        (
            r"$\sigma$",
            r"$\sigma^2 \sim \mathrm{InvGamma}(2.5,\ 0.025)$（感度分析）",
            # E[σ] = b^{1/2} Γ(a - 1/2) / Γ(a)
            f"{np.sqrt(ig_b) * np.exp(special.gammaln(ig_a - 0.5) - special.gammaln(ig_a)):.3f}",
            iv(*np.sqrt(ig.ppf([0.025, 0.975]))),
        ),
        (r"$\rho$", r"$\mathrm{Uniform}(-1,\ 1)$", "0.000", iv(-0.95, 0.95, "{:.2f}")),
        (
            r"$\nu$",
            r"$\nu-2 \sim \mathrm{Gamma}(2,\ 0.1)$",
            f"{2 + nu.mean():.1f}",
            iv(*(2 + nu.ppf([0.025, 0.975])), "{:.1f}"),
        ),
    ]
    lines = [
        r"\begin{tabular}{lllr}",
        r"\toprule",
        r"パラメータ & 事前分布 & 事前平均 & 95\% 区間 \\",
        r"\midrule",
    ]
    lines += [f"{p} & {d} & {m} & {i} \\\\" for p, d, m, i in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    # 列の並び（事前平均を右寄せ、区間を左寄せ）を揃える
    lines[0] = r"\begin{tabular}{llrl}"
    (common.TABLES / "tab_priors.tex").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    args = common.parse_args(__doc__)
    out = common.RESULTS / "f1"
    out.mkdir(parents=True, exist_ok=True)
    common.TABLES.mkdir(parents=True, exist_ok=True)
    plotting.setup()
    fig_illustration(common.data_key(args.seed, 1), out)
    table_properties()
    table_priors()


if __name__ == "__main__":
    main()
