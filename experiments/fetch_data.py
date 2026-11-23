"""J-Quants（無料プラン）から実データを取得して data/raw/ にキャッシュする。

対象：ETF 1321（日経225連動）・1306（TOPIX連動）と TOPIX Core30 の構成銘柄。
無料プランは 5 リクエスト/分なので、30 銘柄強で 7〜8 分かかる。キャッシュ済みの銘柄は取得しない。
取得後に、各銘柄の期間・件数、絶対値の大きいリターン（分割調整の確認）、ETF の 7 月のリターン（分配落ちの確認）を表示する。
"""

import numpy as np
import pandas as pd

from sv_vi.data import (
    ETF_CODES,
    RAW_DIR,
    JQuantsClient,
    cached_daily_bars,
    cached_master,
    core30_codes,
    to_returns,
)


def main():
    client = None
    try:
        master = cached_master()
    except FileNotFoundError:
        client = JQuantsClient()
        master = cached_master(client)
    core = core30_codes(master)
    core.to_csv(RAW_DIR / "core30.csv", index=False)
    codes = list(ETF_CODES) + core["Code"].tolist()
    for i, code in enumerate(codes):
        if not (RAW_DIR / f"bars_{code}.csv").exists() and client is None:
            client = JQuantsClient()
        bars = cached_daily_bars(code, client)
        r = to_returns(bars)
        print(f"[{i + 1}/{len(codes)}] {code}: {len(r)} 日 {r.index.min():%Y-%m-%d}〜{r.index.max():%Y-%m-%d}"
              f"  sd={r.std():.2f}%  max|r|={r.abs().max():.1f}%", flush=True)  # fmt: skip

    print("\n=== 絶対値の大きいリターン（分割調整の漏れがないか） ===")
    for code in codes:
        r = to_returns(cached_daily_bars(code))
        big = r[r.abs() > 12]
        if len(big):
            print(code, big.round(1).to_dict())

    print("\n=== ETF の分配落ちの確認（7 月前半のリターンと、Δlog C と Δlog AdjC の差） ===")
    for code, name in ETF_CODES.items():
        b = cached_daily_bars(code)
        b["Date"] = pd.to_datetime(b["Date"])
        b = b.set_index("Date").sort_index()
        r = 100 * np.log(b["AdjC"]).diff()
        july = r[(r.index.month == 7) & (r.index.day <= 15)]
        print(name)
        print(july.round(2).to_string())
        adj = b["AdjFactor"][b["AdjFactor"] != 1.0]
        print("  AdjFactor != 1 の日:", adj.to_dict())


if __name__ == "__main__":
    main()
