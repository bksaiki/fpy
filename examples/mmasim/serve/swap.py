"""
Replace how a model's `nn.Linear` layers compute, one run at a time.

A run computes under a quantization scheme (`quant`, default `bf16`):

- `fp32`: the model in full precision, `x @ W.T` in FP32 (TF32 off).
- `<scheme>-exact` (`bf16-exact`, ...): `x` and `W` quantized by the
  scheme, their product in FP64 (exact for BF16 and FP8 products), rounded
  once to FP32.
- a design the scheme applies to (`kernels.designs`): the quantized
  elements through its kernel, the scheme's scales applied after it.

Everything outside the linear layers is left as it is: load the model in FP32
so that is full precision in every run.  A linear layer with a bias is
refused.
"""

import argparse
from dataclasses import dataclass, field
from types import MethodType
from typing import get_args

import kernels
import quant
import torch
import torch.nn.functional as F

BF16 = quant.SCHEMES['bf16']


def modes(scheme: quant.Scheme = BF16) -> tuple[str, ...]:
    """*scheme*'s runs: R0, its exact run, and every design it applies to."""
    return ('fp32', f'{scheme.name}-exact', *kernels.designs(scheme))


MODES = modes()
RUNS = MODES[1:]
"""Every `bf16` run but R0 (`fp32`)."""

MODEL = 'Qwen/Qwen3-0.6B'
"""The default model; `Qwen/Qwen3.5-0.8B` also runs (text only)."""

_ROWS = 256
"""Rows per block of an exact run's FP64 product."""

_ELEMS = 1 << 24
"""FP64 weight elements per block of an exact run's columns."""


def exact(qa: quant.Quantized, qw: quant.Quantized) -> torch.Tensor:
    """`qa @ qw.T` in FP64, rounded once to FP32."""
    m, k = qa.elements.shape
    n = qw.elements.shape[0]
    y = torch.empty(m, n, device=qa.elements.device)
    cols = max(1, _ELEMS // k)
    for j in range(0, n, cols):
        wd = qw.dequantize(slice(j, j + cols)).T
        for i in range(0, m, _ROWS):
            y[i:i + _ROWS, j:j + cols] = qa.dequantize(slice(i, i + _ROWS)) @ wd
    return y


def gemm(design: str, scheme: quant.Scheme, qa: quant.Quantized, qw: quant.Quantized,
         w: torch.Tensor, combine: kernels.Combine = 'linear') -> torch.Tensor:
    """`qa @ qw.T` by *design* under *scheme*, FP32: *w* is *qw*'s elements
    prepared for it (`kernels.prepare`), and the scales are applied after the
    kernel, `acc * s_x[i] * s_w[j]`."""
    y = kernels.matmul(qa.elements.to(kernels.storage(design)[0]), w, design, combine)
    return y * qa.scales * qw.scales.T if scheme.applied == 'epilogue' else y


@dataclass
class Run:
    """How every patched `nn.Linear` computes; see :func:`patch`."""

    mode: str = 'fp32'
    scheme: quant.Scheme = BF16
    split_k: int = 1
    combine: kernels.Combine = 'linear'
    _weights: dict[tuple[int, int, str], tuple[torch.Tensor, quant.Quantized]] = field(
        default_factory=dict, compare=False, repr=False)
    """Each weight and it quantized, by `id`, version and scheme."""
    _prepared: dict[tuple[int, int, str, torch.dtype, int], torch.Tensor] = field(
        default_factory=dict, compare=False, repr=False)
    """Each quantized weight prepared (`kernels.prepare`), by `id`, version,
    scheme, storage and split; only the current scheme, storage and split's
    are kept."""
    _input: tuple[torch.Tensor, int, str, quant.Quantized] | None = field(
        default=None, compare=False, repr=False)
    """The last input, its version and scheme, and it quantized."""

    def _weight(self, w: torch.Tensor) -> quant.Quantized:
        key = (id(w), w._version, self.scheme.name)
        if key not in self._weights:
            self._weights = {k: v for k, v in self._weights.items() if k[0] != id(w)}
            self._weights[key] = (w, quant.quantize(w, self.scheme.w))
        return self._weights[key][1]

    def _prepare(self, w: torch.Tensor, qw: quant.Quantized, dtype: torch.dtype) -> torch.Tensor:
        key = (id(w), w._version, self.scheme.name, dtype, self.split_k)
        if key not in self._prepared:
            self._prepared = {k: v for k, v in self._prepared.items()
                              if k[0] != id(w) and k[2:] == key[2:]}
            self._prepared[key] = kernels.prepare(qw.elements, dtype, self.split_k)
        return self._prepared[key]

    def _quantize(self, x: torch.Tensor) -> quant.Quantized:
        """*x* quantized, reused while the same tensor comes back unchanged:
        `q/k/v_proj` share one, as do `gate/up_proj`."""
        last = self._input
        if last is not None and last[0] is x and last[1:3] == (x._version, self.scheme.name):
            return last[3]
        qa = quant.quantize(x.reshape(-1, x.shape[-1]), self.scheme.x)
        self._input = (x, x._version, self.scheme.name, qa)
        return qa

    def linear(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        if self.mode == 'fp32':
            return F.linear(x, w)
        qa, qw = self._quantize(x), self._weight(w)
        if self.mode == f'{self.scheme.name}-exact':
            y = exact(qa, qw)
        elif self.mode in kernels.TILES and kernels.applicable(self.mode, self.scheme):
            held = kernels.storage(self.mode)[1]
            y = gemm(self.mode, self.scheme, qa, qw, self._prepare(w, qw, held), self.combine)
        else:
            raise ValueError(f'{self.mode!r} is not one of {modes(self.scheme)}')
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
