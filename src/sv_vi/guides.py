"""NumPyro 側の変分族（ガイド）。

- "meanfield"  : AutoNormal（各変数独立のガウス）
- "lowrank"    : AutoLowRankMultivariateNormal（共分散 = W W^T + 対角）
- "fullrank"   : AutoMultivariateNormal（密な共分散。T が小さいときの参照用）
- "structured" : 大域パラメータは AutoMultivariateNormal、潜在変数は三重対角精度のガウス
"""

import jax.numpy as jnp
import numpyro
from numpyro import handlers
from numpyro.infer.autoguide import (
    AutoGuideList,
    AutoLowRankMultivariateNormal,
    AutoMultivariateNormal,
    AutoNormal,
)
from numpyro.infer.initialization import init_to_value

from sv_vi.distributions import BandedPrecisionNormal
from sv_vi.initialization import initial_latent_factor
from sv_vi.logdensity import ModelSpec

GUIDE_KINDS = ("meanfield", "lowrank", "fullrank", "structured")


class BandedLatentGuide:
    """潜在変数 1 サイトに対する q(v) = N(loc, (L L^T)^{-1})。AutoGuideList の部品として使う。"""

    def __init__(self, site: str, init_loc, init_diag, init_off):
        self.site = site
        self.init_loc, self.init_diag, self.init_off = init_loc, init_diag, init_off

    def __call__(self, *args, **kwargs):
        loc = numpyro.param(f"{self.site}_loc", self.init_loc)
        log_diag = numpyro.param(f"{self.site}_log_diag", jnp.log(self.init_diag))
        off = numpyro.param(f"{self.site}_off", self.init_off)
        value = numpyro.sample(self.site, BandedPrecisionNormal(loc, jnp.exp(log_diag), off))
        return {self.site: value}


def make_guide(kind: str, model, spec: ModelSpec, init: dict, init_scale: float = 0.1, rank=10):
    """model は y だけを引数に取る形（functools.partial(sv_model, spec=spec)）を想定。"""
    init_fn = init_to_value(values=init)
    if kind == "meanfield":
        return AutoNormal(model, init_loc_fn=init_fn, init_scale=init_scale)
    if kind == "lowrank":
        return AutoLowRankMultivariateNormal(
            model, init_loc_fn=init_fn, init_scale=init_scale, rank=rank
        )
    if kind == "fullrank":
        return AutoMultivariateNormal(model, init_loc_fn=init_fn, init_scale=init_scale)
    if kind == "structured":
        latent = spec.latent_name
        hidden = [latent] if spec.centered else [latent, "h"]  # 非中心化では決定論的サイト h も隠す
        guide = AutoGuideList(model)
        guide.append(
            AutoMultivariateNormal(
                handlers.block(model, hide=hidden), init_loc_fn=init_fn, init_scale=init_scale
            )
        )
        d0, s0 = initial_latent_factor(init, spec)
        guide.append(BandedLatentGuide(latent, init[latent], d0, s0))
        return guide
    raise ValueError(kind)
