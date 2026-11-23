"""確率的ボラティリティ（SV）モデルの変分推論。

import 時に次の 2 つを設定する（JAX のバックエンドが初期化される前に行う必要がある）。

- NUTS の 4 チェーンを CPU で並列実行するため、ホストデバイスを 4 つに見せる
- SV は exp(-h) を扱うので倍精度（float64）を有効にする
"""

import os

import numpyro

numpyro.set_host_device_count(min(4, os.cpu_count() or 1))
numpyro.enable_x64()

__all__ = ["numpyro"]
