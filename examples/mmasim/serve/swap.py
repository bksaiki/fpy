"""
Replace how a model's `nn.Linear` layers compute, one run at a time.

The runs:

- `fp32`: the model in full precision, `x @ W.T` in FP32 (TF32 off).
- `bf16-exact`: `x` and `W` rounded to BF16, the dot product in FP64 (BF16
  products are exact in it), rounded once to FP32.
- a design (`kernels.BF16_DESIGNS`): `x` and `W` rounded to BF16, the
  design's kernel, its FP32 result.

Everything outside the linear layers is left as it is: load the model in FP32
so that is full precision in every run.  A linear layer with a bias is
refused.
"""

import argparse
from dataclasses import dataclass, field
from types import MethodType
from typing import get_args

import kernels
import torch
import torch.nn.functional as F

MODES = ('fp32', 'bf16-exact', *kernels.BF16_DESIGNS)
RUNS = MODES[1:]
"""Every run but R0 (`fp32`)."""

MODEL = 'Qwen/Qwen3-0.6B'
"""The default model; `Qwen/Qwen3.5-0.8B` also runs (text only)."""

_ROWS = 256
"""Rows per block of `bf16-exact`'s FP64 product."""

_ELEMS = 1 << 24
"""FP64 weight elements per block of `bf16-exact`'s columns."""


def _exact(a: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """`a @ w.T` in FP64, rounded once to FP32, for `a` `[m, k]` and `w` `[n, k]`
    holding BF16 values."""
    m, k = a.shape
    y = torch.empty(m, w.shape[0], device=a.device)
    cols = max(1, _ELEMS // k)
    for j in range(0, w.shape[0], cols):
        wd = w[j:j + cols].double().T
        for i in range(0, m, _ROWS):
            y[i:i + _ROWS, j:j + cols] = a[i:i + _ROWS].double() @ wd
    return y


@dataclass
class Run:
    """How every patched `nn.Linear` computes; see :func:`patch`."""

    mode: str = 'fp32'
    split_k: int = 1
    combine: kernels.Combine = 'linear'
    _weights: dict[tuple[int, int, torch.dtype, int], tuple[torch.Tensor, torch.Tensor]] = field(
        default_factory=dict, compare=False, repr=False)
    """Each weight and it prepared (`kernels.prepare`), by `id`, version,
    storage and split; only the current split's are kept."""
    _input: tuple[torch.Tensor, int, torch.dtype, torch.Tensor] | None = field(
        default=None, compare=False, repr=False)
    """The last input, its version and storage, and it rounded."""

    def _weight(self, w: torch.Tensor, dtype: torch.dtype, split_k: int) -> torch.Tensor:
        key = (id(w), w._version, dtype, split_k)
        if key not in self._weights:
            self._weights = {k: v for k, v in self._weights.items()
                             if k[0] != id(w) and k[3] == split_k}
            self._weights[key] = (w, kernels.prepare(w, dtype, split_k))
        return self._weights[key][1]

    def _round(self, x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """*x* rounded (`kernels.round_input`), reused while the same tensor
        comes back unchanged: `q/k/v_proj` share one, as do `gate/up_proj`."""
        last = self._input
        if last is not None and last[0] is x and last[1] == x._version and last[2] == dtype:
            return last[3]
        a = kernels.round_input(x, dtype)
        self._input = (x, x._version, dtype, a)
        return a

    def linear(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        if self.mode == 'fp32':
            return F.linear(x, w)
        if self.mode == 'bf16-exact':
            y = _exact(self._round(x, torch.float32), self._weight(w, torch.float32, 1)[0])
        elif self.mode in kernels.BF16_DESIGNS:
            held_x, held_w = kernels.storage(self.mode)
            y = kernels.matmul(self._round(x, held_x), self._weight(w, held_w, self.split_k),
                               self.mode, self.combine)
        else:
            raise ValueError(f'{self.mode!r} is not one of {MODES}')
        return y.reshape(*x.shape[:-1], w.shape[0])


def patch(model: torch.nn.Module) -> Run:
    """Route every `nn.Linear` of *model* through a :class:`Run`, returned:
    set its fields to change the run.  Starts at `fp32`, which computes as the
    unpatched model does."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    run = Run()
    for name, layer in model.named_modules():
        if isinstance(layer, torch.nn.Linear):
            if layer.bias is not None:
                raise ValueError(f'{name} has a bias')
            layer.forward = MethodType(lambda self, x: run.linear(x, self.weight), layer)
    return run


def add_args(ap: argparse.ArgumentParser) -> None:
    """The options every script shares: the model, and how designs split `k`."""
    ap.add_argument('--model', default=MODEL)
    ap.add_argument('--split-k', type=int, default=1)
    ap.add_argument('--combine', choices=get_args(kernels.Combine), default='linear')


def load(name: str = MODEL, split_k: int = 1, combine: kernels.Combine = 'linear',
         ) -> tuple[torch.nn.Module, Run]:
    """*name*'s causal LM from the Hub, in FP32 on the GPU and patched
    (:func:`patch`), its designs splitting `k` as given."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).cuda().eval()
    run = patch(model)
    run.split_k, run.combine = split_k, combine
    return model, run
