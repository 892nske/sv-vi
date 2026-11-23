"""実験 S4：パラメータ化・事前分布・最適化設定に対する感度（付録 C）。

高持続シナリオ（T=1000）の R 本のデータで、次を比べる（評価は各データの NUTS との比）。
- パラメータ化：中心化 / 非中心化 × 変分族（平均場・構造化・条件付き）
- σ の事前分布：HalfNormal(1) / σ² ~ InvGamma(2.5, 0.025)（NUTS も同じ事前で回す）
- ELBO の MC サンプル数：1 / 4 / 16（条件付き）
- 学習率：0.003 / 0.01 / 0.03（条件付き）

出力：results/s4/rows.csv、表A1
"""

import time

import common
import jax
import pandas as pd

from sv_vi.logdensity import ModelSpec
from sv_vi.simulate import SCENARIOS, simulate_sv

T = 1000
SCENARIO = "high_persistence"

# (設定名, グループ, 手法, ModelSpec, 上書きする最適化設定)
VARIANTS = (
    [
        (f"{fam}/{'中心化' if c else '非中心化'}", "パラメータ化", f"ADVI-{fam}", ModelSpec(centered=c), {})
        for fam in ("meanfield", "structured", "conditional")
        for c in (True, False)
    ]
    + [("conditional/InvGamma", "σ の事前分布", "ADVI-conditional", ModelSpec(sigma_prior="invgamma"), {})]
    + [(f"conditional/MC={n}", "MC サンプル数", "ADVI-conditional", ModelSpec(), {"num_mc": n}) for n in (1, 4, 16)]
    + [(f"conditional/lr={lr}", "学習率", "ADVI-conditional", ModelSpec(), {"lr": lr}) for lr in (0.003, 0.01, 0.03)]
)  # fmt: skip


def extra_args(p):
    p.add_argument("--reps", type=int, default=5)


def run(args, cfg, out):
    csv = out / "rows.csv"
    rows = pd.read_csv(csv) if csv.exists() else pd.DataFrame(columns=["rep", "variant"])
    params = SCENARIOS[SCENARIO]
    for rep in range(args.reps):
        data = simulate_sv(common.data_key(args.seed, 4, rep), T, params)
        refs = {}
        for j, (name, group, method, spec, kw) in enumerate(VARIANTS):
            if ((rows["rep"] == rep) & (rows["variant"] == name)).any():
                continue
            if spec.sigma_prior not in refs:  # NUTS は事前分布ごとに 1 回（中心化で回す）
                ref_spec = ModelSpec(sigma_prior=spec.sigma_prior)
                key = common.data_key(args.seed, 4, rep, 999, len(refs))
                refs[spec.sigma_prior] = common.run_method("NUTS", data["y"], ref_spec, key, cfg)
            key = common.data_key(args.seed, 4, rep, j)
            r = common.run_method(method, data["y"], spec, key, cfg, khat=False, **kw)
            row = common.compare_row(r, params.as_dict(), data["h"], refs[spec.sigma_prior])
            row |= {"rep": rep, "variant": name, "group": group}
            rows = pd.concat([rows, pd.DataFrame([row])], ignore_index=True)
            rows.to_csv(csv, index=False)
            print(f"rep={rep} {name}: phi={row['phi_mean']:.3f} sigma={row['sigma_mean']:.3f}",
                  flush=True)  # fmt: skip
        jax.clear_caches()
    return rows


def table(rows: pd.DataFrame):
    g = rows.groupby(["group", "variant"], sort=False)
    agg = g.agg(
        phi=("phi_mean", "mean"),
        sigma=("sigma_mean", "mean"),
        phi_sd=("phi_sd_ratio", "median"),
        sigma_sd=("sigma_sd_ratio", "median"),
        h_sd=("h_sd_ratio", "median"),
        h_rmse=("h_rmse", "mean"),
        elbo=("elbo_final", "mean"),
        conv=("conv_step", "median"),
    ).reset_index()
    lines = [
        r"\begin{tabular}{llrrrrrrr}",
        r"\toprule",
        r"比較 & 設定 & $\bar\phi$ & $\bar\sigma$ & $\mathrm{sd}(\phi)$比 & $\mathrm{sd}(\sigma)$比 & $\mathrm{sd}(h)$比 & $h$ RMSE & 収束step \\",
        r"\midrule",
    ]
    prev = None
    for _, r in agg.iterrows():
        grp = r["group"] if r["group"] != prev else ""
        if prev is not None and grp:
            lines.append(r"\addlinespace")
        prev = r["group"]
        lines.append(
            f"{grp} & {r['variant']} & {r['phi']:.3f} & {r['sigma']:.3f} & {r['phi_sd']:.2f} & "
            f"{r['sigma_sd']:.2f} & {r['h_sd']:.2f} & {r['h_rmse']:.3f} & {r['conv']:.0f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    common.TABLES.mkdir(parents=True, exist_ok=True)
    (common.TABLES / "tabA1_sensitivity.tex").write_text("\n".join(lines) + "\n")
    return agg


def main():
    args = common.parse_args(__doc__, extra_args)
    cfg = common.RunConfig.quick() if args.quick else common.RunConfig()
    if args.quick:
        args.reps = 1
    out = common.RESULTS / ("s4_quick" if args.quick else "s4")
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    rows = run(args, cfg, out)
    agg = table(rows)
    common.write_env(out, args, cfg, started)
    print(agg.to_string(float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
