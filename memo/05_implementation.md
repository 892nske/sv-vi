# 05 実装方針

## 環境

- Python 3.12（`.python-version`）、パッケージ管理は uv
- 依存（予定）：
  - 本体：`jax`, `numpyro`, `numpy`, `scipy`, `pandas`, `matplotlib`, `arviz`
  - 実データ・比較：`arch`（GARCH）、データ取得用ライブラリ（取得元確定後に決める → 06）
  - 開発：`pytest`, `ruff`
- 追加は `uv add <pkg>` / `uv add --dev <pkg>` で行い、`uv.lock` をコミットする
- 倍精度：SV は $\exp(-h_t)$ を扱うので `jax.config.update("jax_enable_x64", True)` を基本とする（float32 との比較も一応確認）
- CPU 前提で十分な規模（$T \le 10^4$、パラメータ数 $\sim 3\times10^4$）。GPU は必須にしない

## ディレクトリ構成（予定）

```
sv-vi/
├── memo/                 # 方針・検討メモ（本ディレクトリ）
├── docs/                 # LaTeX 論文（後で整備）
│   ├── main.tex
│   ├── sections/
│   ├── figures/          # 実験スクリプトが出力
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
│   └── data.py           # 実データ取得・前処理・キャッシュ
├── experiments/          # 実験スクリプト（S1〜S4, R1〜R3）
├── tests/                # pytest
└── pyproject.toml
```

## 実装の方針・注意

- **自前 ADVI と NumPyro の二本立て**
  - 自前実装（`advi.py`）は論文で「中身を見せる」ためのもの。数十〜百数十行に収め、論文付録にほぼそのまま載せられる読みやすさを優先
  - NumPyro は「実務ではライブラリでここまで書ける」ことを示す＋自前実装の検算
  - 同じモデル・同じ変分族で両者の ELBO と事後要約が一致することをテストで確認
- **構造化ガイドの $O(T)$ 実装**：二重対角 $L$ の後退代入を `lax.scan` で。密行列を作らない（03 参照）
- **JIT**：最適化ループ全体を `jax.jit` / `lax.scan` でまとめる。計測時はコンパイル時間を分離
- **乱数**：`jax.random.PRNGKey` をスクリプトの引数で受け、結果に記録
- **テスト**（最低限）
  - 合成データ生成の統計量（定常分散 $\sigma^2/(1-\phi^2)$ 等）
  - 構造化ガイドのサンプル共分散が $(LL^\top)^{-1}$ と一致、エントロピーが解析値と一致
  - $T$ が小さいとき、フルランク VI と NUTS がほぼ一致
  - 自前 ADVI と NumPyro SVI の結果一致
- **コード品質**：`ruff` でフォーマット・lint。型ヒントは付ける

## 実装順序

1. `simulate.py` ＋テスト
2. `models.py`（基本SV）＋ `mcmc.py` で NUTS が真値を復元することを確認
3. `advi.py`（平均場）→ NumPyro `AutoNormal` と一致確認
4. 構造化ガイド（自前・NumPyro 両方）
5. 実験 S1〜S3 のスクリプト化
6. 拡張モデル（レバレッジ → t）
7. `data.py` と実データ実験 R1〜R3
8. docs/ の LaTeX 雛形と図の出力パイプライン
