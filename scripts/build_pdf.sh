#!/usr/bin/env bash
# docs/main.tex をローカルでコンパイルする（LuaLaTeX + biber、latexmk）。
#
# 使い方：
#   scripts/build_pdf.sh          # docs/build/main.pdf を作る（変更があった分だけ再実行）
#   scripts/build_pdf.sh --clean  # 中間ファイルと PDF を消す
#
# TeX Live はユーザー領域（~/texlive/<年>）に入れてある前提。PATH に通っていなくても動くよう、
# ここで ~/texlive の最新の bin を PATH の先頭に足す。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TLBIN="$(ls -d "$HOME"/texlive/20*/bin/x86_64-linux 2>/dev/null | sort | tail -n 1 || true)"
if [[ -n "$TLBIN" ]]; then
  export PATH="$TLBIN:$PATH"
fi
if ! command -v latexmk >/dev/null 2>&1; then
  echo "latexmk が見つからない。TeX Live を ~/texlive に入れるか PATH を通すこと（memo/08 参照）" >&2
  exit 1
fi

cd "$ROOT/docs"
if [[ "${1:-}" == "--clean" ]]; then
  latexmk -outdir=build -C main.tex
  exit 0
fi

# -outdir は Overleaf に影響しないようコマンドラインで渡す（latexmkrc には書かない）
latexmk -outdir=build -interaction=nonstopmode -halt-on-error -file-line-error main.tex
echo "→ docs/build/main.pdf"
