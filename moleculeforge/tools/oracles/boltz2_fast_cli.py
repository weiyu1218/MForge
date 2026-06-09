#!/workspace/.mforge_boltz_env/bin/python
from __future__ import annotations

import os
import sys


def _set_thread_defaults() -> None:
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(key, "1")


def _patch_checkpoint_initializers() -> None:
    import torch
    from torch import nn

    torch.set_num_threads(int(os.environ.get("BOLTZ_TORCH_NUM_THREADS", "1")))
    try:
        torch.set_num_interop_threads(
            int(os.environ.get("BOLTZ_TORCH_INTEROP_THREADS", "1"))
        )
    except RuntimeError:
        pass

    def noop(*_args, **_kwargs) -> None:
        return None

    nn.Linear.reset_parameters = noop
    nn.Embedding.reset_parameters = noop
    nn.LayerNorm.reset_parameters = noop
    if hasattr(nn, "MultiheadAttention"):
        nn.MultiheadAttention._reset_parameters = noop

    from boltz.model.layers import initialize

    for name in (
        "bias_init_one_",
        "bias_init_zero_",
        "final_init_",
        "gating_init_",
        "glorot_uniform_init_",
        "he_normal_init_",
        "ipa_point_weights_init_",
        "lecun_normal_init_",
        "normal_init_",
        "trunc_normal_init_",
    ):
        setattr(initialize, name, noop)


def main() -> int:
    _set_thread_defaults()
    _patch_checkpoint_initializers()
    from boltz.main import cli

    if sys.argv[0].endswith("-script.pyw"):
        sys.argv[0] = sys.argv[0][:-11]
    elif sys.argv[0].endswith(".exe"):
        sys.argv[0] = sys.argv[0][:-4]
    result = cli()
    return int(result) if result is not None else 0


if __name__ == "__main__":
    raise SystemExit(main())
