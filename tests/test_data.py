import numpy as np
import pandas as pd

from sv_vi import data


class _FakeClient(data.JQuantsClient):
    def __init__(self, pages):  # API キーを読まない
        self.pages = list(pages)
        self.calls = []

    def _get(self, path, params):
        self.calls.append(dict(params))
        return self.pages.pop(0)


def test_pagination_concatenates_pages():
    c = _FakeClient([
        {"data": [{"Date": "2025-01-06"}], "pagination_key": "k1"},
        {"data": [{"Date": "2025-01-07"}]},
    ])  # fmt: skip
    rows = c.get_all("/equities/bars/daily", {"code": "13210"})
    assert [r["Date"] for r in rows] == ["2025-01-06", "2025-01-07"]
    assert c.calls[1]["pagination_key"] == "k1"
    assert c.calls[1]["code"] == "13210"


def test_to_returns_skips_null_days_and_uses_adjusted_close():
    bars = pd.DataFrame({
        "Date": ["2025-01-06", "2025-01-07", "2025-01-08", "2025-01-09"],
        "Code": ["13210"] * 4,
        "AdjC": [100.0, None, 110.0, 99.0],
    })  # fmt: skip
    r = data.to_returns(bars)
    np.testing.assert_allclose(r.values, 100 * np.log([110 / 100, 99 / 110]))
    assert list(r.index.strftime("%Y-%m-%d")) == ["2025-01-08", "2025-01-09"]


def test_core30_filter():
    m = pd.DataFrame({
        "Code": ["72030", "13210", "99840"],
        "CoName": ["a", "b", "c"],
        "S33Nm": ["x", "y", "z"],
        "ScaleCat": ["TOPIX Core30", "-", "TOPIX Large70"],
    })  # fmt: skip
    assert data.core30_codes(m)["Code"].tolist() == ["72030"]
