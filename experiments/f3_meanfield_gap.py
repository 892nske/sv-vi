"""3 章の表：平均場近似が失う情報（θ を固定したときの h の事後分布での計算）。

θ を固定すると、h の事後分布は精度行列 Λ = Q_AR(φ, σ) + (1/2) I の正規分布でよく近似できる
（1/2 は観測 1 つが h_t について持つ期待フィッシャー情報 E[y_t^2 e^{-h_t}] / 2）。
この正規分布を平均場（各 h_t 独立）で近似したときの
  - 隣り合う日の事後相関  corr(h_t, h_{t+1})
  - 平均場の sd と真の周辺 sd の比  sqrt((1/Λ_tt) / (Λ^{-1})_tt)
  - KL の最小値  (1/2) (Σ_t log Λ_tt - log det Λ)
を T = 1000 で計算する。ELBO(θ) = log p(y | θ) - KL なので、KL が大きい θ ほど平均場の ELBO では不利になる。

さらに S2 の合成データ（高持続, T = 1000）について、NUTS の事後平均と平均場 VI の推定値の 2 点で
ラプラス近似による log p(y | θ) と KL の最小値を計算し、平均場の ELBO が小さい φ を好む理由を数値で示す。

出力：docs/tables/tab_meanfield_gap.tex、docs/tables/tab_meanfield_laplace.tex
"""

import common
import numpy as np
import pandas as pd

from sv_vi.simulate import SCENARIOS

T = 1000
OBS_INFO = 0.5  # 観測 1 つあたりの期待フィッシャー情報


def posterior_precision(phi: float, sigma: float, T: int = T) -> np.ndarray:
    """Q_AR(φ, σ) + OBS_INFO * I（密行列。T = 1000 程度なら十分速い）。"""
    i = np.arange(T)
    Q = np.zeros((T, T))
    Q[i, i] = (1.0 + phi**2) / sigma**2
    Q[0, 0] = Q[-1, -1] = 1.0 / sigma**2
    Q[i[:-1], i[1:]] = Q[i[1:], i[:-1]] = -phi / sigma**2
    return Q + OBS_INFO * np.eye(T)


def gap(phi: float, sigma: float) -> dict:
    L = posterior_precision(phi, sigma)
    S = np.linalg.inv(L)
    _, logdet = np.linalg.slogdet(L)
    t = T // 2  # 端の影響を避けて中央の日で測る
    return {
        "corr": S[t, t + 1] / np.sqrt(S[t, t] * S[t + 1, t + 1]),
        "sd_ratio": np.sqrt((1.0 / L[t, t]) / S[t, t]),
        "kl": 0.5 * (np.sum(np.log(np.diag(L))) - logdet),
    }


def laplace_check(y: np.ndarray, mu: float, phi: float, sigma: float) -> dict:
    """θ を固定したときのラプラス近似。log p(y | θ) と、h の事後分布の正規近似に対する KL の最小値。"""
    n = len(y)
    Q = posterior_precision(phi, sigma, n) - OBS_INFO * np.eye(n)
    h = np.full(n, mu)
    for _ in range(200):  # ニュートン法で p(h | y, θ) の最頻値を求める
        grad = -0.5 + 0.5 * y**2 * np.exp(-h) - Q @ (h - mu)
        H = Q + np.diag(0.5 * y**2 * np.exp(-h))
        step = np.linalg.solve(H, grad)
        h = h + step
        if np.abs(step).max() < 1e-10:
            break
    H = Q + np.diag(0.5 * y**2 * np.exp(-h))
    _, logdet_q = np.linalg.slogdet(Q)
    _, logdet_h = np.linalg.slogdet(H)
    log_prior = 0.5 * logdet_q - 0.5 * n * np.log(2 * np.pi) - 0.5 * (h - mu) @ Q @ (h - mu)
    log_obs = np.sum(-0.5 * np.log(2 * np.pi) - h / 2 - 0.5 * y**2 * np.exp(-h))
    log_ml = log_prior + log_obs + 0.5 * n * np.log(2 * np.pi) - 0.5 * logdet_h
    kl = 0.5 * (np.sum(np.log(np.diag(H))) - logdet_h)
    return {"log_ml": log_ml, "kl": kl}


def table_laplace():
    s2 = common.RESULTS / "s2"
    if not (s2 / "summary.csv").exists():
        print("results/s2 がないので tab_meanfield_laplace は作らない")
        return
    y = np.load(s2 / "data.npz")["y"]
    r = pd.read_csv(s2 / "summary.csv").set_index("method")
    rows = [("NUTS の事後平均", r.loc["NUTS"]), ("平均場 VI の推定値", r.loc["ADVI-meanfield"])]
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"$\theta$ & $\mu$ & $\phi$ & $\sigma$ & $\log p(y \mid \theta)$ & KL の最小値 & 差 \\",
        r"\midrule",
    ]
    for name, row in rows:
        mu, phi, sigma = (float(row[f"{k}_mean"]) for k in ("mu", "phi", "sigma"))
        c = laplace_check(y, mu, phi, sigma)
        lines.append(
            f"{name} & {mu:.3f} & {phi:.3f} & {sigma:.3f} & {c['log_ml']:.1f} & {c['kl']:.1f}"
            f" & {c['log_ml'] - c['kl']:.1f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (common.TABLES / "tab_meanfield_laplace.tex").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def meanfield_estimate() -> tuple[float, float]:
    """S2 で平均場 VI が到達した (φ, σ)。結果がなければ表A2 の値。"""
    csv = common.RESULTS / "s2" / "summary.csv"
    if csv.exists():
        r = pd.read_csv(csv).set_index("method").loc["ADVI-meanfield"]
        return float(r["phi_mean"]), float(r["sigma_mean"])
    return 0.053, 0.703


def main():
    common.parse_args(__doc__)
    rows = [
        (
            "合成：高持続（真値）",
            SCENARIOS["high_persistence"].phi,
            SCENARIOS["high_persistence"].sigma,
        ),
        ("合成：標準（真値）", SCENARIOS["standard"].phi, SCENARIOS["standard"].sigma),
        (
            "合成：低持続（真値）",
            SCENARIOS["low_persistence"].phi,
            SCENARIOS["low_persistence"].sigma,
        ),
        ("平均場 VI の推定値", *meanfield_estimate()),
    ]
    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r" & $\phi$ & $\sigma$ & 隣接日の事後相関 & sd の比 & KL の最小値 [nats] \\",
        r"\midrule",
    ]
    for i, (name, phi, sigma) in enumerate(rows):
        g = gap(phi, sigma)
        if i == len(rows) - 1:
            lines.append(r"\addlinespace")
        lines.append(
            f"{name} & {phi:.3f} & {sigma:.3f} & {g['corr']:.3f} & {g['sd_ratio']:.3f} & {g['kl']:.1f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    common.TABLES.mkdir(parents=True, exist_ok=True)
    (common.TABLES / "tab_meanfield_gap.tex").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    table_laplace()


if __name__ == "__main__":
    main()
