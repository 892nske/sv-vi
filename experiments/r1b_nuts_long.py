"""実験 R1b：ETF の全期間推定の「正解」を長い NUTS で確認する（6 章）。

R1 の NUTS（4 連鎖、warmup・sample 各 2,000）はパラメータの R̂ が最大 1.07 で、VI を評価する正解役としてはやや短い。
そこで S2 と同じ長さ（4 連鎖、warmup 2,000 の後 32,000 回更新して 8 つおきに保存）の NUTS を回し、
提案 VI（R1 の結果）と標準の長さの NUTS の両方を、この長い NUTS と比べる。

出力：results/r1b/（長い NUTS の npz、agreement.csv）、docs/tables/tab2b_etf_agreement.tex
"""

import time

import common
import jax
import numpy as np
import pandas as pd
from r1_full_period import ETF_JA, ETFS, MODEL_JA, MODELS

NUM_WARMUP = 2000
NUM_SAMPLES = 32_000
THINNING = 8
EXTRA = {"basic": None, "leverage": "rho", "student_t": "nu"}


def run(args, cfg, out):
    r1 = common.RESULTS / "r1"
    for ci, code in enumerate(ETFS):
        y = np.load(r1 / f"y_{code}_all.npy")
        for mi, (model, spec) in enumerate(MODELS.items()):
            path = out / f"{code}_{model}_NUTS-long.npz"
            if path.exists():
                continue
            key = common.data_key(args.seed, 12, ci, mi)
            r = common.run_method(
                "NUTS", y, spec, key, cfg,
                num_warmup=NUM_WARMUP, num_samples=NUM_SAMPLES, thinning=THINNING,
            )  # fmt: skip
            common.save_summary(r, path)
            ex = r.extra
            print(
                f"{code} {model}: {r.time_total:.0f}s rhat_max={max(ex['rhat'].values()):.3f} "
                f"ess_min={min(ex['ess'].values()):.0f} div={ex['divergences']}",
                flush=True,
            )
            jax.clear_caches()


def compare(out) -> pd.DataFrame:
    """長い NUTS を基準に、提案 VI と標準の NUTS を比べる。"""
    r1 = common.RESULTS / "r1"
    recs = []
    for code in ETFS:
        for model in MODELS:
            ref = common.load_summary(out / f"{code}_{model}_NUTS-long.npz")
            names = ["mu", "phi", "sigma"] + ([EXTRA[model]] if EXTRA[model] else [])
            rec = {"code": code, "model": model, "time_long": ref.time_total}
            rec |= {
                "rhat_long": max(ref.extra["rhat"].values()),
                "ess_long": min(ref.extra["ess"].values()),
                "div_long": ref.extra["divergences"],
            }
            for m, tag in (("ADVI-conditional", "vi"), ("NUTS", "nuts")):
                s = common.load_summary(r1 / f"{code}_all_{model}_{m}.npz")
                for k in names:
                    sd = ref.theta[k].std()
                    rec[f"{tag}_{k}_z"] = (s.theta[k].mean() - ref.theta[k].mean()) / sd
                    rec[f"{tag}_{k}_sd_ratio"] = s.theta[k].std() / sd
                rec[f"{tag}_h_sd_ratio"] = float(np.median(s.h_sd / ref.h_sd))
                rec[f"{tag}_h_z"] = float(np.median(np.abs(s.h_mean - ref.h_mean) / ref.h_sd))
                rec[f"{tag}_max_abs_z"] = max(abs(rec[f"{tag}_{k}_z"]) for k in names)
            for k in names:
                rec[f"long_{k}_mean"] = ref.theta[k].mean()
            recs.append(rec)
    df = pd.DataFrame(recs)
    df.to_csv(out / "agreement.csv", index=False)
    return df


def table(df: pd.DataFrame):
    """表：長い NUTS に対する提案 VI（と標準の NUTS）の一致度。"""
    tex = {"rho": r"$\rho$", "nu": r"$\nu$"}

    def f(v, fmt):
        return "--" if pd.isna(v) else format(v, fmt)

    lines = [
        r"\begin{tabular}{lrrrrrrrrr}",
        r"\toprule",
        r" & \multicolumn{3}{c}{提案：基準化した平均差} & \multicolumn{4}{c}{提案：sd 比} "
        r"& \multicolumn{2}{c}{標準の NUTS} \\",
        r"\cmidrule(lr){2-4} \cmidrule(lr){5-8} \cmidrule(lr){9-10}",
        r"モデル & $\phi$ & $\sigma$ & $\rho$ / $\nu$ & $\phi$ & $\sigma$ & $\rho$ / $\nu$ & $h$ "
        r"& 最大差 & $h$ sd 比 \\",
    ]  # fmt: skip
    for code in ETFS:
        lines += [r"\midrule", rf"\multicolumn{{10}}{{l}}{{{ETF_JA[code]}}} \\"]
        for model in MODELS:
            r = df[(df["code"] == code) & (df["model"] == model)].iloc[0]
            k = EXTRA[model]
            ex_z = f(r[f"vi_{k}_z"], "+.2f") if k else "--"
            ex_sd = f(r[f"vi_{k}_sd_ratio"], ".2f") if k else "--"
            if k:
                ex_z += f"（{tex[k]}）"
            lines.append(
                f"{MODEL_JA[model]} & "
                f"{r['vi_phi_z']:+.2f} & {r['vi_sigma_z']:+.2f} & {ex_z} & "
                f"{r['vi_phi_sd_ratio']:.2f} & {r['vi_sigma_sd_ratio']:.2f} & {ex_sd} & "
                f"{r['vi_h_sd_ratio']:.2f} & {r['nuts_max_abs_z']:.2f} & {r['nuts_h_sd_ratio']:.2f} \\\\"
            )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (common.TABLES / "tab2b_etf_agreement.tex").write_text("\n".join(lines) + "\n")


def main():
    args = common.parse_args(__doc__)
    cfg = common.RunConfig()
    out = common.RESULTS / "r1b"
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    run(args, cfg, out)
    df = compare(out)
    table(df)
    common.write_env(out, args, cfg, started)
    print(df.T.to_string(float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
