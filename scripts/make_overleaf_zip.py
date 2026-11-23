"""docs/ を Overleaf にアップロードする zip にまとめる。

使い方：uv run python scripts/make_overleaf_zip.py
出力：results/overleaf/sv-vi-docs.zip（git 管理外）
Overleaf の「新規プロジェクト → プロジェクトのアップロード」でこの zip を選ぶと、
main.tex がプロジェクトのルートに来る。コンパイラは LuaLaTeX（latexmkrc で指定済み）。
"""

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
OUT = ROOT / "results" / "overleaf" / "sv-vi-docs.zip"
PATTERNS = [
    "main.tex",
    "latexmkrc",
    "references.bib",
    "sections/*.tex",
    "figures/*.pdf",
    "tables/*.tex",
]


def main() -> None:
    files = sorted({p for pat in PATTERNS for p in DOCS.glob(pat) if p.is_file()})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, p.relative_to(DOCS).as_posix())
    print(f"{OUT.relative_to(ROOT)}: {len(files)} files")
    for p in files:
        print("  " + p.relative_to(DOCS).as_posix())


if __name__ == "__main__":
    main()
