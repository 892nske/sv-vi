"""全手法で共通に使う初期値。

- h の初期値：log(y_t^2 + c) + 1.27 を移動平均で平滑化したもの
  （log ε_t^2 の期待値が約 -1.27 なので、log y_t^2 ≈ h_t - 1.27）
- θ の初期値：φ=0.95, σ=0.2, μ=h の初期値の平均（ρ=0, ν=10）
- 潜在変数の精度行列の初期値：AR(1) 事前の精度 ＋ 尤度の曲率（観測 1 つあたり約 1/2）。
  θ を固定したときの p(h | y, θ) のラプラス近似を大雑把に真似たもの。三重対角になる。
"""

import jax.numpy as jnp
import numpy as np

from sv_vi import banded
from sv_vi.logdensity import ModelSpec


def initial_values(y: np.ndarray, spec: ModelSpec, window: int = 10) -> dict:
    """制約付き空間での初期値 {"mu", "phi", "sigma", ["rho"], ["nu"], latent_name}。"""
    y = np.asarray(y)
    c = 0.1 * np.mean(y**2)
    raw = np.log(y**2 + c) + 1.27
    pad = np.pad(raw, (window // 2, window - 1 - window // 2), mode="edge")
    h0 = np.convolve(pad, np.ones(window) / window, mode="valid")

    theta = {"mu": float(np.mean(h0)), "phi": 0.95, "sigma": 0.2}
    if spec.leverage:
        theta["rho"] = 0.0
    if spec.student_t:
        theta["nu"] = 10.0
    latent = h0 if spec.centered else (h0 - theta["mu"]) / theta["sigma"]
    init = {k: jnp.asarray(v) for k, v in theta.items()}
    init[spec.latent_name] = jnp.asarray(latent)
    return init


def initial_latent_factor(init: dict, spec: ModelSpec) -> tuple[jnp.ndarray, jnp.ndarray]:
    """潜在変数の三重対角精度の初期値を、コレスキー因子 (d, s) の形で返す。"""
    T = init[spec.latent_name].shape[0]
    phi, sigma = init["phi"], init["sigma"]
    if spec.centered:
        a, b = banded.ar1_precision(T, phi, sigma)
        a = a + 0.5
    else:  # x = (h - μ)/σ  →  尤度の曲率は σ^2 倍
        a, b = banded.ar1_precision(T, phi, 1.0)
        a = a + 0.5 * sigma**2
    return banded.tridiag_cholesky(a, b)
