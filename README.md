# sv-vi：確率的ボラティリティモデルの変分推論

社内論文「確率的ボラティリティ（SV）モデルの変分推論による推定」の実験コード。
方針・検討メモは `memo/`、論文（LaTeX, Overleaf 用）は `docs/`。

## セットアップ

```bash
uv sync          # 依存パッケージのインストール（Python 3.12）
uv run pytest    # テスト
uv run ruff check . && uv run ruff format --check .
```

CPU だけで動く（GPU 不要）。NUTS の 4 チェーンは CPU 上で並列に回す。

## 構成

| パス | 内容 |
|---|---|
| `src/sv_vi/simulate.py` | 合成データ生成（基本SV・レバレッジ・t 誤差） |
| `src/sv_vi/logdensity.py` | 同時対数密度（純 JAX）。自前 ADVI が使う |
| `src/sv_vi/models.py` | 同じモデルの NumPyro 版（NUTS・NumPyro SVI が使う） |
| `src/sv_vi/banded.py` | 三重対角精度行列の O(T) 演算（後退代入・コレスキー分解など） |
| `src/sv_vi/advi.py` | 自前 ADVI（平均場・構造化・条件付き構造化の 3 変分族） |
| `src/sv_vi/guides.py` | NumPyro の変分族（AutoNormal 等と三重対角精度のカスタムガイド） |
| `src/sv_vi/inference.py` | NUTS・SVI・ADVI を共通形式で実行するラッパ |
| `src/sv_vi/diagnostics.py` | PSIS k̂、ELBO の収束判定 |
| `experiments/` | 実験スクリプト（S1〜S4） |

## 実験の再現

各スクリプトは `--quick`（動作確認用の小さい設定）と `--seed` を受け付ける。
結果は `results/`（git 管理外）、図は `docs/figures/`、表は `docs/tables/` に出力される。

```bash
uv run python experiments/s2_agreement.py    # S2: NUTS との一致度（図2・図3・図A1）
uv run python experiments/s3_scaling.py      # S3: 計算時間のスケーリング（図4）※他の処理と同時に走らせない
uv run python experiments/s4_sensitivity.py  # S4: 感度分析（表A1）
uv run python experiments/s1_recovery.py     # S1: 復元精度（表1）※数時間かかる。中断しても続きから再開できる
```
