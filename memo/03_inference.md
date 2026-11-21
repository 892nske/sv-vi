# 03 推論手法

潜在変数とパラメータをまとめて $z = (\theta, h_{1:T})$（無制約空間に変換済み）と書く。目標は事後分布 $p(z\mid y)$。

## 3.1 位置付け：MCMC と VI

| | MCMC（NUTS） | VI |
|---|---|---|
| 何をするか | 事後分布からのサンプル列を生成 | 事後分布に最も近い分布を族 $\mathcal{Q}$ の中から最適化で探す |
| 精度 | 漸近的に正確 | 変分族の表現力で上限が決まる（バイアスあり） |
| 計算 | 重い。収束・混合の診断が必要 | 速い。最適化なので warm start・早期終了が可能 |
| 運用 | 実行時間が読みにくい（混合の悪さで大きく変動） | 反復数で実行時間を制御しやすい |
| 本稿での役割 | **正解（ゴールドスタンダード）** | **評価対象** |

## 3.2 変分推論の基礎（論文 4 章の素材）

- KL 最小化 ⇔ ELBO 最大化：
  $$\log p(y) = \underbrace{\mathbb{E}_{q}[\log p(y,z) - \log q(z)]}_{\mathrm{ELBO}(q)} + \mathrm{KL}(q\,\|\,p(\cdot\mid y))$$
- $\mathrm{KL}(q\|p)$ は **mode-seeking**：$q$ は $p$ の裾を切り捨てがち → **事後分散の過小評価**。
  リスク管理用途では「不確実性を小さく見せる」方向のバイアスである点を強調する。
- **再パラメータ化勾配**（Kingma & Welling 2014）：$z = T_\lambda(\epsilon) = m + C\epsilon,\ \epsilon\sim\mathcal{N}(0,I)$ と書けば
  $\mathrm{ELBO}(\lambda) = \mathbb{E}_\epsilon[\log p(y, T_\lambda(\epsilon))] + \mathcal{H}[q_\lambda]$、
  $\nabla_\lambda \mathrm{ELBO} = \mathbb{E}_\epsilon[\nabla_z \log p(y,z)\,\partial T_\lambda/\partial\lambda] + \nabla_\lambda \mathcal{H}[q_\lambda]$。
  ガウス族ならエントロピー $\mathcal{H}$ は $\log|\det C|$ で解析的に書ける。期待値はモンテカルロ1〜数サンプルで不偏推定 → Adam 等で確率的最適化。
- **ADVI**（Kucukelbir et al. 2017）：制約付きパラメータを無制約空間に写し、そこでガウス族を当てる。
  変換のヤコビアンを $\log p$ に足すのを忘れない。

## 3.3 比較する変分族（論文 5 章の核心）

$d = T+3$（基本SV）。

| 名称 | 共分散構造 | 変分パラメータ数 | サンプリング計算量 | 実装 |
|---|---|---|---|---|
| (a) 平均場ガウス | 対角 | $2d$ | $O(T)$ | 自前 JAX / NumPyro `AutoNormal` |
| (b) フルランクガウス | 密 $LL^\top$ | $O(T^2)$ | $O(T^2)$〜$O(T^3)$ | NumPyro `AutoMultivariateNormal`（$T$ 小で参照用） |
| (c) 低ランク＋対角 | $WW^\top + D$ | $O(Tk)$ | $O(Tk)$ | NumPyro `AutoLowRankMultivariateNormal` |
| (d) **構造化（帯精度）** | $\theta$：フルランク、$h$：精度行列が三重対角 | $\approx 3T$ | $O(T)$ | 自作ガイド（自前 JAX / NumPyro custom guide） |

### (d) 構造化ガイドの設計根拠と実装

- **根拠**：AR(1) 事前 $p(h\mid\theta)$ の精度行列は三重対角。観測の対数尤度は $h_t$ ごとに分離（対角ヘッセ）。
  よって $\theta$ を固定したときの $p(h\mid y,\theta)$ のラプラス近似（ガウス近似）の精度行列は**ちょうど三重対角**。
  → 「事後の真の相関構造を、計算量を増やさずに表現できる最小の族」と説明できる。
  （Tan & Nott 2018 のスパース精度ガウス変分近似、Chan & Jeliazkov 2009 の精度行列ベースのサンプリングが理論的背景）
- **表現**：$q(h) = \mathcal{N}(m,\ (LL^\top)^{-1})$、$L$ は下二重対角（対角は正：$\exp$ で保証）。
- **サンプリング**：$\epsilon\sim\mathcal{N}(0,I)$ に対し $L^\top x = \epsilon$ を後退代入で解き $h = m + x$。
  二重対角なので $O(T)$。JAX では `lax.scan`（逐次）または結合則を使った `associative_scan` で実装。
  ※ `jax.scipy.linalg.solve_triangular` に密行列を渡すと $O(T^2)$ メモリになるので避ける。
- **エントロピー**：$\mathcal{H}[q] = \tfrac{d}{2}(1+\log 2\pi) - \sum_t \log L_{tt}$（対数行列式が対角の和で済む）。
- **発展形（時間があれば）**：$q(\theta)\,q(h\mid\theta)$ として $m, L$ を $\theta$ の関数にする
  （$\theta$ と $h$ の事後相関、特に $\mu$–$h$、$\sigma$–$h$ の相関を捉える）。非中心化パラメータ化と組み合わせる手もある。

### パラメータ化 × 変分族の組み合わせ

中心化/非中心化（→ 02）と (a)〜(d) の組み合わせで結果が変わる可能性が高い。
仮説：平均場では非中心化（白色化）の方がマシ、構造化では中心化で十分。実験で確認する。

## 3.4 比較対象（ベースライン）

- **NUTS**（NumPyro, Hoffman & Gelman 2014）：4 チェーン、warmup 2000 / sample 2000 程度から。
  $\hat R$、ESS を報告。非中心化パラメータ化の方が発散が少ないはず。
- **（任意）QML**（Harvey, Ruiz & Shephard 1994）：$\log y_t^2$ を線形状態空間化してカルマンフィルタ。
  高速な古典手法として参考に出す程度。
- **（実データのみ）GARCH(1,1)**：読者が知っている実務の標準。VaR 比較用（`arch` パッケージ）。

## 3.5 VI の当てはまり診断（運用の観点で重要）

- ELBO の推移（収束判定：移動平均の相対変化が閾値以下）
- **PSIS の Pareto $\hat k$**（Yao et al. 2018）：$q$ を提案分布とした重要度重みの裾の重さ。
  $\hat k < 0.7$ なら近似は概ね信頼可能。**本番監視のアラート指標**として提案できる。
- NUTS との事後モーメント比較（合成・実データとも）

## 3.6 最適化設定（初期案）

- Adam、学習率 $10^{-2}$ 前後＋減衰、反復 5,000〜20,000、ELBO の MC サンプル数 1〜8
- 初期値：$h$ の平均は $\log(y_t^2 + c)$ を平滑化したもの、$\theta$ は事前平均付近
- 複数シードで実行し、結果のばらつきも報告する（再現性の議論に使う）
