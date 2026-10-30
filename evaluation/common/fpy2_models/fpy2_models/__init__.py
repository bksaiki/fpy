"""FPy's compiled MMA models, registered as torch.ops.fpy2_models.<op>, with
the interface of MMA-Sim's `dpa`; see generate_models.py."""

import torch as _torch  # Load LibTorch before the extension on platforms using rpath.

from . import _C as _C
