# 08 docs/：Overleaf 向け LaTeX の方針

## 運用

- LaTeX 一式はローカルの `docs/` で作り、ユーザーが Overleaf にアップロードしてコンパイル・確認する
- ローカルに TeX 環境は入れない（コンパイルは Overleaf 側のみ）。そのため、Overleaf で確実に通る素直な構成にする
- アップロードは `docs/` を zip にまとめて行う（`scripts/make_overleaf_zip.sh` などを用意する予定）。
  Overleaf の「新規プロジェクト → プロジェクトのアップロード」で zip をそのまま取り込める

## コンパイラと文書クラス

- **LuaLaTeX ＋ `ltjsarticle`**（luatexja。日本語の標準的な構成で、Overleaf でそのまま使える）
- コンパイラ指定：`docs/latexmkrc` に `$pdf_mode = 4;`（LuaLaTeX）と書く。
  Overleaf のメニュー（Menu → Compiler → LuaLaTeX）でも指定できるので、手順書にも書いておく
- 1 段組、A4、10〜11pt。本文 約 10 ページに収まるよう余白・行送りを調整
- 主なパッケージ：`amsmath`, `amssymb`, `bm`, `graphicx`, `booktabs`, `hyperref`, `cleveref`, `listings`（または `minted` は Overleaf で設定が要るので避け、`listings` を使う）, `siunitx`（表の数値揃え）

## 参考文献

- **biblatex ＋ biber**（Overleaf は自動で biber を回す）。スタイルは `authoryear` 系
- `docs/references.bib` は `memo/references.md` を原典確認しながら作る

## ファイル構成

```
docs/
├── main.tex              # プリアンブルと \input
├── latexmkrc             # $pdf_mode = 4;
├── sections/
│   ├── 01_introduction.tex
│   ├── 02_sv_model.tex
│   ├── 03_inference.tex
│   ├── 04_variational_family.tex
│   ├── 05_synthetic.tex
│   ├── 06_empirical.tex
│   ├── 07_system.tex
│   ├── 08_conclusion.tex
│   └── appendix.tex
├── figures/              # *.pdf（実験スクリプトが出力）
└── references.bib
```

## 図表

- 図は matplotlib で **PDF（ベクター）**出力し、`docs/figures/` に置く
- **図中の文字は英語**、キャプションは日本語（図の PDF に日本語フォントを埋め込む問題を避けるため）
- 表は可能なら実験スクリプトから `booktabs` 形式の `.tex` 断片を出力し、`\input` する（数値の転記ミスを防ぐ）
- 図のサイズは本文幅基準で統一（`\linewidth`）。色はカラー前提だが、白黒印刷でも線種で区別できるようにする

## 注意

- Overleaf の無料プランはコンパイル時間に上限がある。大きな図（点の多い散布図など）はラスタ化するか点を間引く
- `\input` のパスは `docs/` 直下からの相対パスにする（zip を展開した位置がルートになるため）

## 執筆の進め方（2026-11-24 ユーザー指示）

- **執筆段階では分量（文字数・ページ数）を気にせず、省略せずに詳しく書く。分量は最後にまとめて調整する**（全章共通）
- 章ごとに書いてユーザーが Overleaf で確認する

## 雛形の作成（2026-11-24）

- `docs/main.tex`：ltjsarticle（A4・11pt・1 段組、余白 25mm）、amsmath/amssymb/bm/graphicx/booktabs/xcolor/enumitem、
  biblatex（authoryear、biber）、hyperref、cleveref。cleveref は日本語の書式を `\jacref` で定義（「第5章」「第5章・第7章」「第5章〜第8章」「図5」「表2」「付録A」）
- 記法マクロ：`\E`, `\KL`, `\ELBO`, `\Normal`, `\diag`, `\code`
- タイトルは 01 の第 1 案、著者は「（著者名）」の仮置き、概要は未執筆
- `docs/sections/01_introduction.tex` は本文あり。02〜08 と `appendix.tex` は `\section` と `\label` だけのスタブ（1 章からの相互参照が「??」にならないように）
  ラベル：`sec:intro`, `sec:model`, `sec:inference`, `sec:family`, `sec:synthetic`, `sec:empirical`, `sec:system`, `sec:conclusion`, `app:derivation`, `app:code`, `app:settings`, `app:data`
- `docs/references.bib`：1 章で引用する 25 件。Crossref・出版社・JMLR/PMLR/NeurIPS・arXiv・CiNii・BIS で巻号・ページ・DOI を確認したものだけを入れた
- zip：`uv run python scripts/make_overleaf_zip.py` → `results/overleaf/sv-vi-docs.zip`（git 管理外。main.tex・latexmkrc・bib・sections・figures・tables の 28 ファイル）
- Overleaf の手順：新規プロジェクト → プロジェクトのアップロード → zip を選ぶ。Menu → Compiler が LuaLaTeX になっていることを確認（latexmkrc でも指定）
- ローカルに TeX がないため、こちらでは引用キー・ラベル・環境の対応だけを機械的に確認している。初回のコンパイルエラーはユーザーの報告をもとに直す
