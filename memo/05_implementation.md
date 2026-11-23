# 05 実装方針

## 環境

- Python 3.12（`.python-version`）、パッケージ管理は uv
- 依存（予定）：
  - 本体：`jax`, `numpyro`, `numpy`, `scipy`, `pandas`, `matplotlib`, `arviz`
  - 実データ：`requests`（J-Quants V2 API）、`python-dotenv`（API キー読み込み）
  - 比較：`arch`（GARCH）
  - 開発：`pytest`, `ruff`
- 追加は `uv add <pkg>` / `uv add --dev <pkg>` で行い、`uv.lock` をコミットする
- 倍精度：SV は $\exp(-h_t)$ を扱うので `jax.config.update("jax_enable_x64", True)` を基本とする（float32 との比較も一応確認）
- CPU 前提で十分な規模（$T \le 10^4$、パラメータ数 $\sim 3\times10^4$、多銘柄でも 30 銘柄 × $T\approx480$）。GPU は必須にしない

## 秘密情報・データの扱い（社内限定だが API キーは必ず守る）

- J-Quants の API キーは環境変数 `JQUANTS_API_KEY`、またはリポジトリ直下の `.env` から読む
- `.env` と `data/`（取得データのキャッシュ）は `.gitignore` に追加する（実装着手時に追記）
- 取得した生データはコミットしない。再取得手順はスクリプトと 07 に残す

## ディレクトリ構成（予定）

```
sv-vi/
├── memo/                 # 方針・検討メモ（本ディレクトリ）
├── docs/                 # LaTeX 論文（Overleaf にアップロードする単位、詳細は 08）
│   ├── main.tex
│   ├── latexmkrc
│   ├── sections/
│   ├── figures/          # 実験スクリプトが PDF で出力
│   └── references.bib
├── src/sv_vi/
│   ├── __init__.py
│   ├── simulate.py       # 合成データ生成（基本SV・レバレッジ・t）
│   ├── models.py         # NumPyro モデル（中心化／非中心化、拡張）
│   ├── logdensity.py     # 自前実装用の log p(y, z) と変換・ヤコビアン（純 JAX）
│   ├── advi.py           # 自前 ADVI：平均場・構造化ガウス、ELBO、最適化ループ
│   ├── guides.py         # NumPyro 用カスタムガイド（三重対角精度ガウス）
│   ├── mcmc.py           # NUTS 実行ラッパ
│   ├── diagnostics.py    # PSIS k̂、ELBO 収束判定
│   ├── metrics.py        # バイアス・RMSE・被覆率・sd 比・Wasserstein、VaR 検定
│   └── data.py           # J-Quants V2 からの取得・キャッシュ・前処理（% 対数リターン）
├── experiments/          # 実験スクリプト（S1〜S4, R1〜R4）
├── scripts/              # 補助スクリプト（Overleaf 用 zip 作成など）
├── tests/                # pytest
├── data/                 # 取得データのキャッシュ（git 管理外）
└── pyproject.toml
```

## 実装の方針・注意

- **自前 ADVI と NumPyro の二本立て**
  - 自前実装（`advi.py`）は論文で「中身を見せる」ためのもの。付録 B に抜粋を載せられる読みやすさを優先（抜粋は 1 ページ以内）
  - NumPyro は「実務ではライブラリでここまで書ける」ことを示す＋自前実装の検算
  - 同じモデル・同じ変分族で両者の ELBO と事後要約が一致することをテストで確認
- **構造化ガイドの $O(T)$ 実装**：二重対角 $L$ の後退代入を `lax.scan` で。密行列を作らない（03 参照）
- **多銘柄の一括推定（R4）**：自前 ADVI の最適化ループを「1 銘柄分の純関数」として書き、`jax.vmap` で銘柄方向にバッチ化できるようにしておく。
  銘柄ごとに $T$ が違う場合（上場・売買停止）はマスクで揃える
- **JIT**：最適化ループ全体を `jax.jit` / `lax.scan` でまとめる。計測時はコンパイル時間を分離
- **乱数**：`jax.random.PRNGKey` をスクリプトの引数で受け、結果に記録
- **J-Quants クライアント**：レート制限（無料 5 件/分）を守るスリープ・リトライ、ページング（`pagination_key`）対応、レスポンスを `data/` に Parquet/CSV でキャッシュ
- **テスト**（最低限）
  - 合成データ生成の統計量（定常分散 $\sigma^2/(1-\phi^2)$ 等）
  - 構造化ガイドのサンプル共分散が $(LL^\top)^{-1}$ と一致、エントロピーが解析値と一致
  - $T$ が小さいとき、フルランク VI と NUTS がほぼ一致
  - 自前 ADVI と NumPyro SVI の結果一致
  - データ前処理（リターン計算、欠損処理）。API 呼び出し部分はキャッシュ済みデータ（または固定のダミー応答）でテスト
- **コード品質**：`ruff` でフォーマット・lint。型ヒントは付ける

## 実装順序

1. `simulate.py` ＋テスト
2. `models.py`（基本SV）＋ `mcmc.py` で NUTS が真値を復元することを確認
3. `advi.py`（平均場）→ NumPyro `AutoNormal` と一致確認
4. 構造化ガイド（自前・NumPyro 両方）
5. 実験 S1〜S3 のスクリプト化
6. 拡張モデル（レバレッジ → t）
7. `data.py`（J-Quants）と実データ実験 R1〜R4
8. docs/ の LaTeX 雛形、図の出力パイプライン、Overleaf 用 zip 作成スクリプト

## 実装メモ（2026-11-22、合成データ実験の実装時）

- 依存：`jax 0.12`, `numpyro 0.22`, `optax 0.2`, `arviz 1.3`（PSIS は `arviz_stats.base.array_stats.psislw`）, `scipy`, `pandas`, `matplotlib`。dev：`pytest`, `ruff`
- 実際のモジュール構成（予定から追加・変更したもの）
  - `banded.py`：三重対角精度の O(T) 演算（後退・前進代入、三重対角コレスキー、AR(1) 精度）
  - `distributions.py`：NumPyro 用の `AR1`（中心化モデルの h の事前）と `BandedPrecisionNormal`
  - `initialization.py`：全手法共通の初期値
  - `inference.py`：`run_nuts` / `run_svi` / `run_advi` が共通の `PosteriorSummary` を返す（h は要約統計だけ保持してメモリ節約）
  - `plotting.py`：図の共通スタイル。色は dataviz スキルの参照パレット（明背景用、検証済み）を手法に固定割当。NUTS は黒
  - `experiments/common.py`：手法の実行・評価行の作成・結果保存・実行環境の記録
- `sv_vi/__init__.py` で `numpyro.set_host_device_count(4)` と `enable_x64()` を行う（NUTS の 4 チェーンを CPU 並列）
- 計時：自前 ADVI は AOT コンパイル（`jit(...).lower().compile()`）でコンパイル時間を分離。NumPyro の `svi.run` はステップ数ごとに再コンパイルするので総時間のみ
- 図は `docs/figures/*.pdf`（フォント埋め込み、`pdf.fonttype=42`）、確認用 PNG は `results/figures_png/`
- メモリ：開発機は空きが約 2GB しかないため、実験は直列に回し、各データセット後に `jax.clear_caches()`
