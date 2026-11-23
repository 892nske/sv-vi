# memo/ インデックス

社内論文「確率的ボラティリティ（SV）モデルの変分推論による推定」の方針メモ置き場。
ここは**検討過程の記録**であり、清書は後で `docs/`（LaTeX）に整備する。

## ファイル一覧

| ファイル | 内容 |
|---|---|
| [01_paper_direction.md](01_paper_direction.md) | 目的・想定読者・中心メッセージ・タイトル案・章構成とページ配分 |
| [02_model.md](02_model.md) | 基本SVと拡張（レバレッジ・t誤差）の定式化、事前分布、パラメータ化、単位 |
| [03_inference.md](03_inference.md) | MCMC と VI の位置付け、変分族の設計（平均場／低ランク／構造化）、ELBO と勾配 |
| [04_experiments.md](04_experiments.md) | 合成データ・実データ実験の計画、評価指標、図表リスト |
| [05_implementation.md](05_implementation.md) | uv 環境、依存パッケージ、ディレクトリ構成、実装上の注意 |
| [06_open_questions.md](06_open_questions.md) | 確認事項（回答記録）・未決事項・TODO |
| [07_data_jquants.md](07_data_jquants.md) | J-Quants API による実データ取得の方針（無料プラン＋ETF で指数代替） |
| [08_docs_latex.md](08_docs_latex.md) | Overleaf 向け LaTeX の構成方針 |
| [09_results_synthetic.md](09_results_synthetic.md) | 合成データ実験（S1〜S4）の結果と論文への反映メモ |
| [10_results_empirical.md](10_results_empirical.md) | 実データ実験（R1〜R4）の結果と論文への反映メモ |
| [references.md](references.md) | 参考文献リスト |

## 決定事項サマリ（2026-11-22 更新）

| 項目 | 決定 |
|---|---|
| 実装基盤 | **JAX で ADVI を自前実装**（仕組みの解説用）＋ **NumPyro**（SVI と NUTS。NUTS を「正解」として比較） |
| 提案手法 | **条件付き構造化変分族**（q(θ)·q(h\|θ)、q(h\|θ) は AR(1) 事前×擬似観測。O(T)）。2026-11-22 追加（→ 03 §(e), 09） |
| モデル範囲 | **基本SV ＋ レバレッジ効果 ＋ Student-t 誤差**（確定） |
| データ | **合成データ**（真値既知 → 復元精度）＋ **J-Quants API 無料プラン**の実データ。指数は取れないので **ETF（1321 日経225連動、1306 TOPIX連動）で代替**＋大型個別株 20〜30 銘柄。S&P500 は対象外 |
| リターンの単位 | **% 単位の対数リターン**（$100\times\Delta\log P$）、標本平均を除去 |
| 論文形式 | **LaTeX（Overleaf でコンパイル）**。LuaLaTeX ＋ `ltjsarticle`。ローカルで一式を作り、ユーザーがアップロードして確認 |
| 分量 | **本文 約10ページ ＋ 付録 2〜4ページ**。テンプレ指定なし |
| 公開範囲 | **社内限定** |
| Python | 3.12、uv 管理 |

## 進め方

1. **方針確定**（本メモ群）（確認事項の回答を反映済み）
2. **実装**：合成データ生成 → 基本SV × NUTS → 基本SV × 各種VI → 比較実験 → 拡張モデル → 実データ
   - 2026-11-23：合成データ実験 S1〜S4 まで完了（→ 09）
   - 2026-11-23：実データ実験 R1〜R4 まで完了（→ 10）。次は docs/ の LaTeX 執筆 ← いまここ
3. **実験結果の蓄積**：`experiments/` で再現可能なスクリプト化、図表は `docs/figures/` に出力
4. **docs/ 整備**：LaTeX で清書し、Overleaf アップロード用 zip を作成

各フェーズで分かったこと・判断の変更は該当メモに追記していく（日付を付ける）。
