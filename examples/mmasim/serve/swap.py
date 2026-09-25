"""
Replace how a model's `nn.Linear` layers compute, one run at a time.

The runs (see `docs/todos/mmasim-serve.md`):

- `fp32`: the model in full precision, `x @ W.T` in FP32 (TF32 off).
- `bf16-exact`: `x` rounded to BF16, the dot product in FP64 -- exact far
  below FP32's precision, BF16 products being exact in it -- rounded once to
  FP32.
- a design (`kernels.BF16_DESIGNS`): `x` rounded to BF16, the design's
  kernel, its FP32 result.

Everything outside the linear layers is left as it is: load the model in FP32
so that is full precision in every run.
"""

from dataclasses import dataclass
from types import MethodType
from typing import Literal

import kernels
import torch
import torch.nn.functional as F

MODES = ('fp32', 'bf16-exact', *kernels.BF16_DESIGNS)

MODEL = 'Qwen/Qwen3-0.6B'
"""The default model; `Qwen/Qwen3.5-0.8B` also runs (text only)."""

_ROWS = 256
"""Rows per block of `bf16-exact`'s FP64 product."""


@dataclass
class Run:
    """How every patched `nn.Linear` computes; see :func:`patch`."""

    mode: str = 'fp32'
    split_k: int = 1
    combine: Literal['linear', 'tree'] = 'linear'

    def linear(self, x: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None) -> torch.Tensor:
        if self.mode == 'fp32':
            return F.linear(x, w, b)
        if self.mode == 'bf16-exact':
            # in blocks of rows: `lm_head` in FP64 is 2.5 GB at 2048 tokens
            wd = w.to(torch.bfloat16).double().T
            xd = x.reshape(-1, x.shape[-1]).to(torch.bfloat16)
            y = torch.empty(xd.shape[0], w.shape[0], device=x.device)
            for i in range(0, xd.shape[0], _ROWS):
                y[i:i + _ROWS] = xd[i:i + _ROWS].double() @ wd
            y = y.reshape(*x.shape[:-1], w.shape[0])
        elif self.mode in kernels.BF16_DESIGNS:
            y = kernels.linear(x, w, self.mode, split_k=self.split_k, combine=self.combine)
        else:
            raise ValueError(f'{self.mode!r} is not one of {MODES}')
        return y if b is None else y + b


def patch(model: torch.nn.Module) -> Run:
    """Route every `nn.Linear` of *model* through a :class:`Run`, returned:
    set its fields to change the run.  Starts at `fp32`, which computes as the
    unpatched model does."""
    torch.backends.cuda.matmul.allow_tf32 = False
    run = Run()
    for layer in model.modules():
        if isinstance(layer, torch.nn.Linear):
            layer.forward = MethodType(
                lambda self, x: run.linear(x, self.weight, self.bias), layer)
    return run


def load(name: str = MODEL) -> tuple[torch.nn.Module, Run]:
    """*name*'s causal LM from the Hub, in FP32 on the GPU and patched
    (:func:`patch`)."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).cuda().eval()
    return model, patch(model)
