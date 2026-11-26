"""評価指標。"""

import numpy as np
from scipy.stats import wasserstein_distance


def bias(est: np.ndarray, truth) -> float:
    return float(np.mean(np.asarray(est) - truth))


def rmse(est: np.ndarray, truth) -> float:
    return float(np.sqrt(np.mean((np.asarray(est) - truth) ** 2)))


def covered(lo, hi, truth) -> np.ndarray:
    """真値が区間 [lo, hi] に入っているか（要素ごと）。"""
    truth = np.asarray(truth)
    return (np.asarray(lo) <= truth) & (truth <= np.asarray(hi))


def sd_ratio(sd_vi, sd_ref) -> np.ndarray:
    """事後標準偏差の比 sd_VI / sd_NUTS。1 未満なら VI が不確実性を過小評価している。"""
    return np.asarray(sd_vi) / np.asarray(sd_ref)


def standardized_mean_diff(mean_vi, mean_ref, sd_ref) -> np.ndarray:
    """事後平均の差を NUTS の事後 sd で基準化したもの。"""
    return (np.asarray(mean_vi) - np.asarray(mean_ref)) / np.asarray(sd_ref)


def wasserstein(samples_vi: np.ndarray, samples_ref: np.ndarray) -> float:
    """1 次元の Wasserstein 距離。"""
    return float(wasserstein_distance(np.ravel(samples_vi), np.ravel(samples_ref)))


# ---------------------------------------------------------------------------
# VaR バックテスト
# ---------------------------------------------------------------------------


def _bern_loglik(n0: int, n1: int, p: float) -> float:
    """ベルヌーイ列（0 が n0 回、1 が n1 回）の対数尤度。0·log0 = 0 として扱う。"""
    from scipy.special import xlogy

    return float(xlogy(n0, 1.0 - p) + xlogy(n1, p))


def kupiec_pof(hits, alpha: float) -> dict:
    """Kupiec (1995) の超過率（POF）検定。H0: 超過確率 = alpha。LR ~ χ²(1)。"""
    from scipy.stats import chi2

    hits = np.asarray(hits, dtype=bool)
    n, x = len(hits), int(hits.sum())
    lr = -2.0 * (_bern_loglik(n - x, x, alpha) - _bern_loglik(n - x, x, x / n))
    return {"n": n, "exceed": x, "rate": x / n, "lr": lr, "p": float(chi2.sf(lr, 1))}


def christoffersen(hits, alpha: float) -> dict:
    """Christoffersen (1998) の独立性検定と条件付き被覆検定。

    超過が連続して起きやすいか（クラスタリング）を 1 次マルコフ連鎖で検定する。
    LR_ind ~ χ²(1)、LR_cc = LR_pof + LR_ind ~ χ²(2)。
    """
    from scipy.stats import chi2

    h = np.asarray(hits, dtype=int)
    prev, cur = h[:-1], h[1:]
    n00 = int(np.sum((prev == 0) & (cur == 0)))
    n01 = int(np.sum((prev == 0) & (cur == 1)))
    n10 = int(np.sum((prev == 1) & (cur == 0)))
    n11 = int(np.sum((prev == 1) & (cur == 1)))
    pi01 = n01 / max(n00 + n01, 1)
    pi11 = n11 / max(n10 + n11, 1)
    pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    l0 = _bern_loglik(n00 + n10, n01 + n11, pi)
    l1 = _bern_loglik(n00, n01, pi01) + _bern_loglik(n10, n11, pi11)
    lr_ind = -2.0 * (l0 - l1)
    lr_cc = kupiec_pof(hits, alpha)["lr"] + lr_ind
    return {
        "lr_ind": lr_ind,
        "p_ind": float(chi2.sf(lr_ind, 1)),
        "lr_cc": lr_cc,
        "p_cc": float(chi2.sf(lr_cc, 2)),
    }


def diebold_mariano(d, lags: int | None = None) -> dict:
    """Diebold & Mariano (1995) の予測精度の比較検定。

    d は 2 つの予測の損失（または対数予測スコア）の日ごとの差。H0: E[d] = 0。
    分散は Newey & West (1987) の HAC 推定（Bartlett 核、ラグ lags）で、統計量を標準正規分布と比べる（両側）。
    lags を省略すると floor(4 (n/100)^(2/9)) を使う。lags=0 なら通常の（標本分散による）検定になる。
    """
    from scipy.stats import norm

    d = np.asarray(d, dtype=float)
    d = d[~np.isnan(d)]
    n = len(d)
    if lags is None:
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    e = d - d.mean()
    lrv = np.dot(e, e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * np.dot(e[k:], e[:-k]) / n
    stat = d.mean() / np.sqrt(lrv / n)
    return {
        "mean": float(d.mean()),
        "stat": float(stat),
        "p": float(2.0 * norm.sf(abs(stat))),
        "lags": lags,
    }
