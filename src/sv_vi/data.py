"""J-Quants API（V2）からの株価取得と前処理。

- 認証：API キーを `x-api-key` ヘッダに付ける。キーは環境変数 JQUANTS_API_KEY（リポジトリ直下の .env から読む）。
  キーはログにも結果ファイルにも出さない。
- 無料プランは 5 リクエスト/分（1 分のスライディングウィンドウ）。超過すると 429 が返り、Retry-After はない。
  → リクエスト間隔を 13 秒空け、429 なら 120 秒待って再試行する。
- 取得結果は data/raw/ に CSV でキャッシュし、再実行時は API を呼ばない。
- 調整後終値 AdjC は分割・併合・ライツのみ調整済み。現金配当（ETF の分配金）は調整されない。
"""

import json
import os
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

BASE_URL = "https://api.jquants.com/v2"
ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"

ETF_CODES = {"13210": "NEXT FUNDS 日経225連動型上場投信", "13060": "NEXT FUNDS TOPIX連動型上場投信"}


class JQuantsClient:
    def __init__(
        self, min_interval: float = 13.0, max_retries: int = 3, wait_on_429: float = 120.0
    ):
        load_dotenv(ROOT / ".env")
        key = os.environ.get("JQUANTS_API_KEY")
        if not key:
            raise RuntimeError("JQUANTS_API_KEY が設定されていません（.env を確認）")
        self._session = requests.Session()
        self._session.headers["x-api-key"] = key
        self.min_interval, self.max_retries, self.wait_on_429 = (
            min_interval,
            max_retries,
            wait_on_429,
        )
        self._last = 0.0

    def _get(self, path: str, params: dict) -> dict:
        for attempt in range(self.max_retries + 1):
            wait = self.min_interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            r = self._session.get(f"{BASE_URL}{path}", params=params, timeout=60)
            if r.status_code == 429 and attempt < self.max_retries:
                print(f"  429 Too Many Requests → {self.wait_on_429:.0f} 秒待機", flush=True)
                time.sleep(self.wait_on_429)
                continue
            if r.status_code != 200:
                # レスポンス本文にキーは含まれないが、念のため先頭だけ表示する
                raise RuntimeError(f"{path} {params}: HTTP {r.status_code} {r.text[:200]}")
            return r.json()
        raise RuntimeError(f"{path}: 再試行回数を超えました")

    def get_all(self, path: str, params: dict) -> list[dict]:
        """pagination_key がなくなるまで取得して data を結合する。"""
        out, params = [], dict(params)
        while True:
            body = self._get(path, params)
            out.extend(body.get("data", []))
            if "pagination_key" not in body:
                return out
            params["pagination_key"] = body["pagination_key"]

    def daily_bars(self, code: str) -> pd.DataFrame:
        """1 銘柄の日足（プランで取得できる全期間）。"""
        return pd.DataFrame(self.get_all("/equities/bars/daily", {"code": code}))

    def master(self) -> pd.DataFrame:
        return pd.DataFrame(self.get_all("/equities/master", {}))


# ---------------------------------------------------------------------------
# キャッシュ
# ---------------------------------------------------------------------------


def _meta_path() -> Path:
    return RAW_DIR / "meta.json"


def _read_meta() -> dict:
    p = _meta_path()
    return json.loads(p.read_text()) if p.exists() else {}


def _write_meta(meta: dict) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    _meta_path().write_text(json.dumps(meta, indent=2, ensure_ascii=False))


def cached_master(client: JQuantsClient | None = None) -> pd.DataFrame:
    path = RAW_DIR / "master.csv"
    if path.exists():
        return pd.read_csv(path, dtype={"Code": str})
    df = (client or JQuantsClient()).master()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    meta = _read_meta()
    meta["master"] = {"fetched": date.today().isoformat(), "rows": len(df)}
    _write_meta(meta)
    return df


def cached_daily_bars(code: str, client: JQuantsClient | None = None) -> pd.DataFrame:
    path = RAW_DIR / f"bars_{code}.csv"
    if path.exists():
        return pd.read_csv(path, dtype={"Code": str})
    df = (client or JQuantsClient()).daily_bars(code)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    meta = _read_meta()
    meta.setdefault("bars", {})[code] = {
        "fetched": date.today().isoformat(),
        "rows": len(df),
        "from": str(df["Date"].min()) if len(df) else None,
        "to": str(df["Date"].max()) if len(df) else None,
        "plan": "free",
    }
    _write_meta(meta)
    return df


# ---------------------------------------------------------------------------
# 前処理
# ---------------------------------------------------------------------------


def to_returns(bars: pd.DataFrame) -> pd.Series:
    """調整後終値から % 対数リターン 100·Δlog P を作る（日付インデックス）。

    売買のなかった日（価格が Null）は除外し、取引のあった日どうしでリターンを計算する。
    平均の除去はしない（推定窓ごとに行う）。
    """
    df = bars[["Date", "AdjC"]].dropna().copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").drop_duplicates("Date").set_index("Date")
    r = 100.0 * np.log(df["AdjC"]).diff().dropna()
    r.name = str(bars["Code"].iloc[0]) if "Code" in bars and len(bars) else "ret"
    return r


def core30_codes(master: pd.DataFrame) -> pd.DataFrame:
    """銘柄一覧から TOPIX Core30 の銘柄を抜き出す。"""
    m = master[master["ScaleCat"].astype(str).str.contains("Core30", na=False)]
    return m.drop_duplicates("Code")[["Code", "CoName", "S33Nm", "ScaleCat"]].reset_index(drop=True)


def aligned_returns(codes: list[str]) -> pd.DataFrame:
    """複数銘柄のリターンを共通の取引日に揃える（どれかが欠ける日は落とす）。"""
    series = [to_returns(cached_daily_bars(c)).rename(c) for c in codes]
    return pd.concat(series, axis=1, join="inner").sort_index()
