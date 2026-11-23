"""実験スクリプト共通の部品：手法の実行、結果の保存、実行環境の記録。"""

import argparse
import json
import platform
import time
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import jax
import numpy as np

import sv_vi  # noqa: F401  （x64・デバイス数の設定を最初に行う）
from sv_vi import metrics
from sv_vi.advi import ADVIConfig
from sv_vi.diagnostics import convergence_step
from sv_vi.inference import PosteriorSummary, run_advi, run_nuts, run_svi
from sv_vi.logdensity import ModelSpec

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TABLES = ROOT / "docs" / "tables"

VI_METHODS = ("ADVI-meanfield", "ADVI-structured", "ADVI-conditional")


@dataclass(frozen=True)
class RunConfig:
    """手法の設定。--quick では小さくする。"""

    nuts_warmup: int = 2000
    nuts_samples: int = 2000
    vi_steps: int = 10_000
    vi_lr: float = 0.01
    vi_num_mc: int = 4
    num_post: int = 1000

    @classmethod
    def quick(cls) -> "RunConfig":
        return cls(nuts_warmup=300, nuts_samples=300, vi_steps=1000, num_post=300)


def parse_args(description: str, extra=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--quick", action="store_true", help="動作確認用の小さい設定で実行する")
    p.add_argument("--seed", type=int, default=20261122)
    if extra:
        extra(p)
    return p.parse_args()


def run_method(
    method: str, y, spec: ModelSpec, key, cfg: RunConfig, khat: bool = True, **overrides
) -> PosteriorSummary:
    if method == "NUTS":
        return run_nuts(
            y,
            spec,
            key,
            num_warmup=overrides.get("num_warmup", cfg.nuts_warmup),
            num_samples=overrides.get("num_samples", cfg.nuts_samples),
            thinning=overrides.get("thinning", 1),
        )
    steps = overrides.get("num_steps", cfg.vi_steps)
    lr = overrides.get("lr", cfg.vi_lr)
    num_mc = overrides.get("num_mc", cfg.vi_num_mc)
    if method.startswith("ADVI-"):
        acfg = ADVIConfig(family=method[5:], num_steps=steps, lr=lr, num_mc=num_mc)
        return run_advi(y, spec, key, acfg, num_post=cfg.num_post, khat=khat)
    if method.startswith("SVI-"):
        return run_svi(
            y, spec, method[4:], key, num_steps=steps, lr=lr, num_mc=num_mc,
            num_post=cfg.num_post, khat=khat,
        )  # fmt: skip
    raise ValueError(method)


def time_to_converge(r: PosteriorSummary) -> tuple[int, float]:
    """(収束ステップ, 収束までの時間 = コンパイル + 実行時間 × 収束ステップ/総ステップ)。"""
    if r.elbo_trace is None:
        return -1, r.time_total
    n = len(r.elbo_trace)
    step = convergence_step(r.elbo_trace)
    t_compile = 0.0 if np.isnan(r.time_compile) else r.time_compile
    return step, t_compile + r.time_run * step / n


def compare_row(r: PosteriorSummary, truth: dict, h_true, ref: PosteriorSummary | None) -> dict:
    """1 手法・1 データセットの評価指標を 1 行にまとめる。"""
    row = {"method": r.method, "time_total": r.time_total, "time_compile": r.time_compile}
    step, t_conv = time_to_converge(r)
    row |= {"conv_step": step, "time_conv": t_conv}
    for k, s in r.theta_summary().items():
        row |= {f"{k}_{stat}": v for stat, v in s.items()}
        if k in truth:
            row[f"{k}_true"] = truth[k]
        if ref is not None:
            row[f"{k}_sd_ratio"] = s["sd"] / ref.theta[k].std()
            row[f"{k}_w1"] = metrics.wasserstein(r.theta[k], ref.theta[k])
    row["h_rmse"] = metrics.rmse(r.h_mean, h_true)
    row["h_cover"] = float(metrics.covered(r.h_q025, r.h_q975, h_true).mean())
    row["h_width"] = float(np.mean(r.h_q975 - r.h_q025))
    if ref is not None:
        row["h_sd_ratio"] = float(np.median(metrics.sd_ratio(r.h_sd, ref.h_sd)))
        row["h_mean_diff"] = float(
            np.median(np.abs(metrics.standardized_mean_diff(r.h_mean, ref.h_mean, ref.h_sd)))
        )
    for k in ("khat", "divergences", "rhat_h_max", "ess_h_min", "min_ess_per_sec"):
        if k in r.extra:
            row[k] = r.extra[k]
    if "rhat" in r.extra:
        row["rhat_max"] = max(r.extra["rhat"].values())
        row["ess_min"] = min(r.extra["ess"].values())
    if r.elbo_trace is not None:
        row["elbo_final"] = float(np.mean(r.elbo_trace[-500:]))
    return row


def save_summary(r: PosteriorSummary, path: Path) -> None:
    """PosteriorSummary を npz で保存する（図の作成用）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {f"theta_{k}": v for k, v in r.theta.items()}
    arrays |= {k: getattr(r, k) for k in ("h_mean", "h_sd", "h_q025", "h_q975")}
    if r.elbo_trace is not None:
        arrays["elbo_trace"] = r.elbo_trace
    meta = {"method": r.method, "time_total": r.time_total, "time_compile": r.time_compile}
    meta["extra"] = r.extra
    np.savez_compressed(path, meta=json.dumps(meta, default=float), **arrays)


def load_summary(path: Path) -> PosteriorSummary:
    z = np.load(path)
    meta = json.loads(str(z["meta"]))
    theta = {k[6:]: z[k] for k in z.files if k.startswith("theta_")}
    return PosteriorSummary(
        method=meta["method"],
        theta=theta,
        h_mean=z["h_mean"],
        h_sd=z["h_sd"],
        h_q025=z["h_q025"],
        h_q975=z["h_q975"],
        time_total=meta["time_total"],
        time_compile=meta["time_compile"],
        elbo_trace=z["elbo_trace"] if "elbo_trace" in z.files else None,
        extra=meta["extra"],
    )


def write_env(out_dir: Path, args: argparse.Namespace, cfg: RunConfig, started: float) -> None:
    """乱数シード・設定・ライブラリのバージョン・実行時間を記録する。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    info = {
        "args": vars(args),
        "config": cfg.__dict__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "jax_devices": [str(d) for d in jax.devices()],
        "versions": {p: version(p) for p in ("jax", "numpyro", "optax", "numpy", "arviz")},
        "elapsed_sec": time.time() - started,
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out_dir / "env.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))


def data_key(seed: int, *ids: int) -> jax.Array:
    key = jax.random.PRNGKey(seed)
    for i in ids:
        key = jax.random.fold_in(key, i)
    return key
