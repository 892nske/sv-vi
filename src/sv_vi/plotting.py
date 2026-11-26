"""論文用の図の共通スタイル（PDF 出力、図中の文字は英語）。

- 色は手法に固定で割り当てる（図によって色が変わらないように）。NUTS は「正解」なので黒系の中立色
- カテゴリ色は dataviz スキルの参照パレット（明るい背景用、検証済み）から順番どおりに取る
- 白黒印刷でも区別できるよう、線種とマーカーも手法ごとに固定する
- 凡例は必ず付ける（コントラストが 3:1 未満の色があるため、色だけに頼らない）
"""

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "docs" / "figures"
PNG_DIR = ROOT / "results" / "figures_png"  # 目視確認用（git 管理外）
TEXT_WIDTH_IN = 6.1  # A4・1 段組の本文幅（約 15.5 cm）

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
TRUTH = "#8a8986"

# 手法 → (表示名, 色, 線種, マーカー)
METHOD_STYLE = {
    "NUTS": ("NUTS (reference)", "#1f1f1e", "-", "o"),
    "ADVI-conditional": ("VI: conditional (proposed)", "#2a78d6", "-", "s"),
    "ADVI-structured": ("VI: structured", "#eb6834", "--", "^"),
    "ADVI-meanfield": ("VI: mean-field", "#1baf7a", ":", "v"),
    "SVI-lowrank": ("VI: low-rank (NumPyro)", "#eda100", "-.", "D"),
    "SVI-fullrank": ("VI: full-rank (NumPyro)", "#e87ba4", (0, (5, 1, 1, 1)), "P"),
    "SVI-meanfield": ("VI: mean-field (NumPyro)", "#1baf7a", (0, (1, 2)), "x"),
    "SVI-structured": ("VI: structured (NumPyro)", "#eb6834", (0, (4, 2)), "*"),
}


def label(method: str) -> str:
    return METHOD_STYLE.get(method, (method,))[0]


# 論文の表で使う日本語の手法名（図は英語の label を使う）
LABEL_JA = {
    "NUTS": "NUTS（基準）",
    "ADVI-conditional": "条件付き構造化（提案）",
    "ADVI-structured": "構造化",
    "ADVI-meanfield": "平均場",
    "SVI-structured": "構造化（NumPyro）",
    "SVI-meanfield": "平均場（NumPyro）",
    "SVI-lowrank": "低ランク（NumPyro）",
    "SVI-fullrank": "フルランク（NumPyro）",
}
FAMILY_JA = {"meanfield": "平均場", "structured": "構造化", "conditional": "条件付き構造化"}


def label_ja(method: str) -> str:
    return LABEL_JA.get(method, method)


def color(method: str) -> str:
    return METHOD_STYLE[method][1]


def linestyle(method: str):
    return METHOD_STYLE[method][2]


def marker(method: str) -> str:
    return METHOD_STYLE[method][3]


def setup() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9,
            "axes.labelsize": 8.5,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.edgecolor": INK_SECONDARY,
            "axes.labelcolor": INK,
            "xtick.color": INK_SECONDARY,
            "ytick.color": INK_SECONDARY,
            "text.color": INK,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 1.4,
            "legend.frameon": False,
            "pdf.fonttype": 42,  # フォントを埋め込む（Overleaf でも文字化けしない）
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def save(fig, name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / f"{name}.pdf"
    fig.savefig(path)
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG_DIR / f"{name}.png", dpi=150)
    plt.close(fig)
    return path
