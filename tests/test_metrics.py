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
