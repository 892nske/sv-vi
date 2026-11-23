# 02 モデル定義

記法：$y_t$ は（平均を除去した）日次対数リターン、$h_t$ は対数分散（潜在変数）、$t = 1,\dots,T$。

**単位（2026-11-22 決定）**：リターンは **% 単位**、すなわち $y_t = 100\,(\log P_t - \log P_{t-1}) - \bar y$ とする。
このとき日次標準偏差は $\exp(h_t/2)$ [%]、年率ボラは $\sqrt{252}\exp(h_t/2)$ [%]。$\mu \approx 0$ が日次 1%（年率約 16%）に相当する。

## 2.1 基本SVモデル（Taylor 1986 型、離散時間・対数AR(1)）

$$
\begin{aligned}
y_t &= \exp(h_t/2)\,\varepsilon_t, & \varepsilon_t &\sim \mathcal{N}(0,1),\\
h_t &= \mu + \phi\,(h_{t-1} - \mu) + \sigma\,\eta_t, & \eta_t &\sim \mathcal{N}(0,1),\quad t \ge 2,\\
h_1 &\sim \mathcal{N}\!\left(\mu,\ \frac{\sigma^2}{1-\phi^2}\right) & &\text{（定常分布から開始）}
\end{aligned}
$$

- パラメータ $\theta = (\mu, \phi, \sigma)$：$\mu$ は対数分散の長期平均、$\phi$ は持続性、$\sigma$ はボラのボラ。
- 年率ボラ換算：$\sqrt{252}\exp(h_t/2)$（実データの図はこれで描く。読者に馴染みのある単位）。
- 観測の対数尤度（1期分）：
  $\log p(y_t \mid h_t) = -\tfrac12\log(2\pi) - \tfrac{h_t}{2} - \tfrac{y_t^2}{2}e^{-h_t}$
- **推定が難しい理由**：$p(y_{1:T}\mid\theta) = \int p(y_{1:T}\mid h_{1:T})\,p(h_{1:T}\mid\theta)\,dh_{1:T}$ が
  $T$ 次元積分で解析解がない（観測方程式が $h_t$ について非線形）。
  → 潜在変数 $h_{1:T}$ もパラメータと一緒に事後分布を求める（次元 $T+3$）。

### 事前分布（Kim–Shephard–Chib 1998 に準拠、`stochvol` パッケージと同系統）

| パラメータ | 事前分布 | 備考 |
|---|---|---|
| $\mu$ | $\mathcal{N}(0, 10^2)$ | 弱情報 |
| $\phi$ | $(\phi+1)/2 \sim \mathrm{Beta}(20, 1.5)$ | 事前平均 $\approx 0.86$、高い持続性を想定。$|\phi|<1$ を保証 |
| $\sigma$ | $\mathrm{HalfNormal}(1)$ | KSC は $\sigma^2 \sim$ 逆ガンマ。HalfNormal の方が $\sigma\to0$ 付近の挙動が穏当（Kastner & Frühwirth-Schnatter 2014 の議論）。感度分析で両方試す |

% 単位では典型値が $\mu \in [-1, 2]$（日次 0.6〜2.7%）、$\sigma \in [0.1, 0.5]$ なので、
$\mathcal{N}(0,10^2)$ は十分に弱情報、$\mathrm{HalfNormal}(1)$ も典型値を十分に覆う（尺度が単位に依存する点に注意。生の対数リターンだと $\mu\approx-9$ になり、事前分布の妥当性が変わる）。

### パラメータ化（実装上重要）

- **中心化**：$h_t$ を直接サンプル/変分近似する。
- **非中心化**：$\tilde h_t = (h_t - \mu)/\sigma$ とし、$\tilde h_t = \phi\tilde h_{t-1} + \eta_t$ を扱う
  （あるいはさらに $\eta_{1:T}$ 自体を潜在変数にする「白色化」）。
- MCMC ではデータの情報量によって効率の良い方が変わる（ASIS はその両取り）。
  VI でも変分族との相性でどちらが良いかが変わるはずなので、**実験要因として比較**する（→ 04）。
- 制約付きパラメータは無制約空間へ変換：$\phi = \tanh(\cdot)$ または $2\,\mathrm{sigmoid}(\cdot)-1$、
  $\sigma = \exp(\cdot)$ / softplus。ADVI の標準的な扱い（ヤコビアン項を忘れない）。

## 2.2 拡張①：レバレッジ効果付き SV（第一候補）

株式で「下落時にボラが上がる」非対称性。実務家に説明しやすく、VaR にも効く。
Omori et al. (2007) の時点の取り方（当期リターンと翌期ボラのショックが相関）に従う：

$$
\begin{aligned}
y_t &= \exp(h_t/2)\,\varepsilon_t,\\
h_{t+1} &= \mu + \phi\,(h_t-\mu) + \sigma\,\eta_t,\\
\begin{pmatrix}\varepsilon_t\\ \eta_t\end{pmatrix} &\sim \mathcal{N}\!\left(\mathbf{0}, \begin{pmatrix}1 & \rho\\ \rho & 1\end{pmatrix}\right).
\end{aligned}
$$

- $\rho<0$ が典型（株価指数で $-0.3$〜$-0.7$ 程度）。
- 実装上は条件付き分布で書ける：$\eta_t = (h_{t+1} - \mu - \phi(h_t-\mu))/\sigma$ として
  $y_t \mid h_t, h_{t+1} \sim \mathcal{N}\big(\rho\,e^{h_t/2}\eta_t,\ (1-\rho^2)\,e^{h_t}\big)$（$t<T$）、
  $y_T \mid h_T \sim \mathcal{N}(0, e^{h_T})$。
- 事前：$(\rho+1)/2 \sim \mathrm{Beta}(1,1)$（一様）を基本とする。
- VI の観点：観測が $h_t, h_{t+1}$ の両方に依存するため、事後の精度行列は依然三重対角近傍
  → 構造化ガイドがそのまま使える（要確認：ラプラス近似で帯幅が増えないことを確認する）。

## 2.3 拡張②：Student-t 誤差 SV（SV-t）

裾の厚さ（ボラ変動だけで説明しきれない急変）。VaR・ES に直結する。

$$
y_t = \exp(h_t/2)\,\varepsilon_t,\qquad \varepsilon_t \sim t_\nu\ \ \text{（分散1に基準化：}\ \varepsilon_t = \sqrt{(\nu-2)/\nu}\ \cdot t_\nu\text{）}
$$

- 基準化するなら $\nu>2$ が必要。事前は $\nu - 2 \sim \mathrm{Gamma}(2, 0.1)$ など（要検討。Juárez & Steel 型）。
- 代替表現：$\varepsilon_t = \sqrt{\lambda_t}\,z_t$, $\lambda_t \sim \mathrm{InvGamma}(\nu/2,\nu/2)$ の尺度混合。
  潜在変数が増えるが条件付き正規になる。VI では周辺化した $t$ 密度を直接使う方が素直。
- 拡張は①②とも扱うことで確定（2026-11-22）。本文では定式化と主要結果のみ、詳細は付録。

## 2.4 データ生成用の「真値」（合成実験）

日次株価指数の典型値に合わせる（KSC 1998、Omori et al. 2007 の推定値付近）：

| シナリオ | $\mu$ | $\phi$ | $\sigma$ | 備考 |
|---|---|---|---|---|
| 標準 | $0.0$（年率ボラ約 16%） | 0.97 | 0.15 | 典型的な株価指数 |
| 高持続 | $0.0$ | 0.99 | 0.10 | 平均場が最も苦しいケース |
| 低持続 | $0.0$ | 0.80 | 0.40 | 平均場でもそこそこ当たるはずのケース |
| レバレッジ | 上記標準 + $\rho=-0.5$ | | | 拡張① |
| t 誤差 | 上記標準 + $\nu=8$ | | | 拡張② |

（% 単位。$\mu=0$ は日次標準偏差 1%、$\sqrt{252}\times1\% \approx 15.9\%$。）
