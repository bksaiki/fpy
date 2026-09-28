"""
Replace how a model's `nn.Linear` layers compute, one run at a time.

A run computes under a quantization scheme (`quant`, default `bf16`):

- `fp32`: the model in full precision, `x @ W.T` in FP32 (TF32 off).
- `<scheme>-exact` (`bf16-exact`, ...): `x` (from BF16) and `W` quantized
  by the scheme, their product in FP64, rounded once to FP32.
- a design the scheme applies to (`kernels.designs`): the quantized
  elements through its kernel, the scheme's scales applied after it, per
  block of `k`, or by its instructions (:func:`gemm`).

Everything outside the linear layers is left as it is: load the model in FP32
so that is full precision in every run.  A linear layer with a bias is
refused.
"""

from dataclasses import dataclass, field
from types import MethodType
from typing import Any

import torch
import torch.nn.functional as F

from . import kernels, quant

BF16 = quant.SCHEMES['bf16']


def modes(scheme: quant.Scheme = BF16) -> tuple[str, ...]:
    """*scheme*'s runs: R0, its exact run, and every design it applies to."""
    return ('fp32', f'{scheme.name}-exact', *kernels.designs(scheme))


MODES = modes()
"""Every `bf16` run."""

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


def slices(design: str, scheme: quant.Scheme, k: int, split_k: int) -> int:
    """How many slices of `k` to prepare a weight in: *split_k*, or one per
    scale block of `k` (`fp8-block`'s 128) or per instruction of a
    block-scaled *design*."""
    if scheme.applied not in ('k-blocks', 'instruction'):
        return split_k
    if split_k != 1:
        raise ValueError(f'{scheme.name} splits `k` into its own blocks, not {split_k}')
    if scheme.applied == 'k-blocks':
        return k // scheme.x.scaling.cols
    return k // kernels.compiled(design)[1]


def _per_call(q: quant.Quantized, design: str) -> torch.Tensor:
    """*q*'s block scales as a block-scaled *design*'s instructions take them,
    `[s, rows]` (one per call) or `[s, rows, g]` (one per group),
    each scale repeated over the groups its block covers."""
    k0, g = kernels.compiled(design)[1], kernels.group(design)
    r, k = q.elements.shape
    sc = q.scales.repeat_interleave(q.operand.scaling.cols // g, 1).view(r, k // k0, k0 // g)
    sc = sc.transpose(0, 1)
    return (sc[..., 0] if g == k0 else sc).contiguous().to(kernels.storage(design)[2])


def gemm(design: str, scheme: quant.Scheme, qa: quant.Quantized, qw: quant.Quantized,
         w: torch.Tensor, combine: kernels.Combine = 'linear') -> torch.Tensor:
    """`qa @ qw.T` by *design* under *scheme*, FP32; *w* is *qw*'s elements
    prepared in :func:`slices`.  Scales apply per `scheme.applied`: after
    the GEMM, `acc * s_x[i] * s_w[j]`; to each block of `k`'s partial; or in
    the instructions, then per tensor, `acc * (g_x * g_w)`."""
    a = qa.elements.to(kernels.storage(design)[0])
    if scheme.applied == 'instruction':
        y = kernels.matmul(a, w, design, scales=(_per_call(qa, design), _per_call(qw, design)))
        return y if qa.tensor is None else y.mul_(qa.tensor * qw.tensor)
    if scheme.applied == 'k-blocks':
        sw = qw.scales.repeat_interleave(qw.operand.scaling.rows, 0)[:w.shape[1]]
        return kernels.matmul(a, w, design, scales=(qa.scales, sw))
    y = kernels.matmul(a, w, design, combine)
    return y.mul_(qa.scales).mul_(qw.scales.T) if scheme.applied == 'epilogue' else y


@dataclass
class Given:
    """Weights given for a scheme, as a checkpoint stores them, and their
    layers' static activation scales, by the weights' `id`."""

    scheme: str
    weights: dict[int, quant.Quantized]
    inputs: dict[int, torch.Tensor]


@dataclass
class Run:
    """How every patched `nn.Linear` computes; see :func:`patch`.  A weight
    is quantized by the scheme's RTN unless `given` for it; a layer whose
    weight is in `ignore` computes as R0 in every run."""

    mode: str = 'fp32'
    scheme: quant.Scheme = BF16
    split_k: int = 1
    combine: kernels.Combine = 'linear'
    given: Given | None = None
    ignore: set[int] = field(default_factory=set)
    _weights: dict[tuple[int, int, str], tuple[torch.Tensor, quant.Quantized]] = field(
        default_factory=dict, compare=False, repr=False)
    """Each weight and its quantized form, by `id`, version and scheme."""
    _prepared: dict[int, tuple[quant.Quantized, torch.dtype, int, torch.Tensor]] = field(
        default_factory=dict, compare=False, repr=False)
    """By weight `id`, its last quantized form, storage, split, and those
    elements prepared (`kernels.prepare`)."""
    _input: tuple[torch.Tensor, tuple[Any, ...], quant.Quantized] | None = field(
        default=None, compare=False, repr=False)
    """The last input, its version, scheme and scales' ids, and its quantized form."""

    def _given(self) -> Given | None:
        return self.given if self.given is not None and self.given.scheme == self.scheme.name else None

    def weight(self, w: torch.Tensor) -> quant.Quantized:
        """*w* under the scheme: given, else quantized (once per version)."""
        given = self._given()
        if given is not None and id(w) in given.weights:
            return given.weights[id(w)]
        key = (id(w), w._version, self.scheme.name)
        if key not in self._weights:
            self._weights = {k: v for k, v in self._weights.items() if k[0] != id(w)}
            self._weights[key] = (w, quant.quantize(w, self.scheme.w))
        return self._weights[key][1]

    def _prepare(self, w: torch.Tensor, qw: quant.Quantized, dtype: torch.dtype) -> torch.Tensor:
        split = slices(self.mode, self.scheme, w.shape[1], self.split_k)
        got = self._prepared.get(id(w))
        if got is None or got[0] is not qw or got[1:3] != (dtype, split):
            got = self._prepared[id(w)] = (qw, dtype, split, kernels.prepare(qw.elements, dtype, split))
        return got[3]

    def quantize(self, x: torch.Tensor, w: torch.Tensor,
                 amax: torch.Tensor | None = None) -> quant.Quantized:
        """*x*, the input of the layer with weight *w*, rounded to BF16 and
        quantized (with its static scale, if given; else a per-tensor scale
        from *amax*, if given), reused while the same tensor comes back
        unchanged: `q/k/v_proj` share one, as do `gate/up_proj`."""
        given = self._given()
        scale = None if given is None else given.inputs.get(id(w))
        key = (x._version, self.scheme.name, id(scale), id(amax))
        last = self._input
        if last is not None and last[0] is x and last[1] == key:
            return last[2]
        a = x.reshape(-1, x.shape[-1]).to(torch.bfloat16).float()
        qa = quant.quantize(a, self.scheme.x, scale, amax)
        self._input = (x, key, qa)
        return qa

    def linear(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        if self.mode == 'fp32' or id(w) in self.ignore:
            return F.linear(x, w)
        qa, qw = self.quantize(x, w), self.weight(w)
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


def give(run: Run, model: torch.nn.Module, scheme: quant.Scheme,
         weights: dict[str, quant.Quantized], inputs: dict[str, torch.Tensor] | None = None,
         ignore: list[str] | tuple[str, ...] = ()) -> None:
    """Set *run*'s scheme, give it *weights* and static activation scales
    *inputs* for it, by layer name; layers in *ignore* compute as R0."""
    layers = dict(model.named_modules())
    run.scheme = scheme
    run.given = Given(scheme.name, {id(layers[n].weight): q for n, q in weights.items()},
                      {id(layers[n].weight): s for n, s in (inputs or {}).items()})
    run.ignore = {id(layers[n].weight) for n in ignore}


def load(name: str = MODEL, split_k: int = 1, combine: kernels.Combine = 'linear',
         ) -> tuple[torch.nn.Module, Run]:
    """*name*'s causal LM from the Hub, in FP32 on the GPU and patched
    (:func:`patch`), its designs splitting `k` as given."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).cuda().eval()
    run = patch(model)
    run.split_k, run.combine = split_k, combine
    return model, run
