import numpy as np
import pytest

from sv_vi.metrics import christoffersen, kupiec_pof


def test_kupiec_matches_hand_calculation():
    hits = np.zeros(250, dtype=bool)
    hits[:5] = True  # 250 日中 5 回超過、alpha = 1%
    r = kupiec_pof(hits, 0.01)
    expected = -2 * (245 * np.log(0.99) + 5 * np.log(0.01) - 245 * np.log(0.98) - 5 * np.log(0.02))
    assert r["exceed"] == 5
    assert r["lr"] == pytest.approx(expected)


def test_kupiec_zero_exceedances():
    r = kupiec_pof(np.zeros(100, dtype=bool), 0.05)
    assert r["lr"] == pytest.approx(-2 * 100 * np.log(0.95))


def test_christoffersen_detects_clustering():
    clustered = np.zeros(200, dtype=bool)
    clustered[50:60] = True
    spread = np.zeros(200, dtype=bool)
    spread[::20] = True
    assert christoffersen(clustered, 0.05)["p_ind"] < 0.01
    assert christoffersen(spread, 0.05)["p_ind"] > 0.1


def test_diebold_mariano_without_lags_is_mean_test():
    from sv_vi.metrics import diebold_mariano

    rng = np.random.default_rng(0)
    d = rng.normal(0.1, 1.0, size=400)
    r = diebold_mariano(d, lags=0)
    expected = d.mean() / (d.std(ddof=0) / np.sqrt(len(d)))
    assert r["stat"] == pytest.approx(expected)
    assert r["lags"] == 0


def test_diebold_mariano_hac_widens_for_positive_autocorrelation():
    from sv_vi.metrics import diebold_mariano

    rng = np.random.default_rng(1)
    e = rng.normal(size=2000)
    d = 0.05 + np.convolve(e, np.ones(5) / 5, mode="valid")  # 正の自己相関を持つ系列
    assert abs(diebold_mariano(d, lags=8)["stat"]) < abs(diebold_mariano(d, lags=0)["stat"])
    assert diebold_mariano(d)["lags"] == int(np.floor(4 * (len(d) / 100) ** (2 / 9)))
