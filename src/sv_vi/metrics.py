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
