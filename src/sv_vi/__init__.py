"""確率的ボラティリティ（SV）モデルの変分推論。

import 時に次の 2 つを設定する（JAX のバックエンドが初期化される前に行う必要がある）。

- NUTS の 4 チェーンや多銘柄の推定を CPU で並列実行するため、CPU を複数のデバイスに見せる
  （既定 4 つ。環境変数 SV_VI_HOST_DEVICES で変更できる）
- SV は exp(-h) を扱うので倍精度（float64）を有効にする
"""

import os

import numpyro

numpyro.set_host_device_count(
    int(os.environ.get("SV_VI_HOST_DEVICES", min(4, os.cpu_count() or 1)))
)
numpyro.enable_x64()

__all__ = ["numpyro"]
