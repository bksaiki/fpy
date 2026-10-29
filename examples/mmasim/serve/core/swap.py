"""
Replace how a model's `nn.Linear` layers compute, one run at a time.

A run computes under a quantization scheme (`quant`, default `bf16`):

- `fp32`: the model in full precision, `x @ W.T` in FP32 (TF32 off).
- `<scheme>-exact` (`bf16-exact`, ...): `x` (from BF16) and `W` quantized
  by the scheme, their product in FP64, rounded once to FP32.
- `<scheme>-exact-rtz`: the same, rounded toward zero: a perturbation of
  the exact run under an ulp.
- `<scheme>-exact-rand`: the exact run moved an ulp up or down, at random
  per element (by :meth:`Run.reseed`'s generator): a perturbation of it
  without bias.
- `<scheme>-exact-p<t>` (:func:`calibration`): the exact product rounded to
  nearest at `t` fraction bits, FP32's exponents: a known loss of
  precision, to read other runs' distances against.
- a design the scheme applies to (`kernels.designs`): the quantized
  elements through its kernel, the scheme's scales applied after it, per
  block of `k`, or by its instructions (:func:`gemm`).

Any run but `fp32` can also be *frozen* to a recorded exact run
(:class:`Frozen`): each layer multiplies the recorded quantized input, and
its own input's offset from the recorded one passes through the quantized
weight unrounded.  Its arithmetic's error propagates; no activation rounds
differently.

Everything outside the linear layers is left as it is: load the model in FP32
so that is full precision in every run.  A linear layer with a bias is
refused.
"""

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from types import MethodType
from typing import Any, Literal

import torch
import torch.nn.functional as F

from . import kernels, quant

BF16 = quant.SCHEMES['bf16']


def modes(scheme: quant.Scheme = BF16) -> tuple[str, ...]:
    """*scheme*'s runs: R0, its exact runs (`-exact`, `-rtz`, `-rand`), and
    every design it applies to."""
    return ('fp32', *(f'{scheme.name}-exact{s}' for s in _EXACT), *kernels.designs(scheme))


PRECISIONS = range(8, 23)
"""The calibration runs' fraction bits (FP32's 23 is the exact run)."""


def calibration(scheme: quant.Scheme = BF16) -> tuple[str, ...]:
    """*scheme*'s calibration runs, an exact run per precision in
    :data:`PRECISIONS`."""
    return tuple(f'{scheme.name}-exact-p{t}' for t in PRECISIONS)


MODEL = 'Qwen/Qwen3-0.6B'

MODELS = {
    MODEL: 'BF16 master',
    'Qwen/Qwen3.5-0.8B': 'BF16 master, text only',
    'RedHatAI/Qwen3-0.6B-FP8-dynamic': 'fp8-row checkpoint',
    'RedHatAI/Qwen3-0.6B-FP8-BLOCK': 'fp8-block checkpoint (baseline: Qwen/Qwen3-0.6B)',
    'kaitchup/Qwen3-0.6B-NVFP4': 'nvfp4 checkpoint',
}
"""The models these scripts have run, and what each holds."""

_ROWS = 256
"""Rows per block of an exact run's FP64 product."""

_ELEMS = 1 << 24
"""FP64 weight elements per block of an exact run's columns."""


Rounding = Literal['nearest', 'toward-zero', 'random-sign']
"""How an exact run rounds: to nearest; toward zero; or to nearest, then an
ulp up or down at random."""

_EXACT: dict[str, Rounding] = {'': 'nearest', '-rtz': 'toward-zero', '-rand': 'random-sign'}
"""Each exact run's rounding, by its name's suffix."""


def _exact_run(mode: str, scheme: quant.Scheme) -> tuple[Rounding, int] | None:
    """How *mode* rounds and to how many fraction bits, if an exact run."""
    suffix = mode.removeprefix(f'{scheme.name}-exact')
    if suffix == mode:
        return None
    if suffix in _EXACT:
        return _EXACT[suffix], 23
    t = suffix.removeprefix('-p')
    return ('nearest', int(t)) if t.isdigit() and int(t) in PRECISIONS else None


def exact(qa: quant.Quantized, qw: quant.Quantized, signs: torch.Generator,
          rounding: Rounding = 'nearest', bits: int = 23) -> torch.Tensor:
    """`qa @ qw.T` in FP64, rounded once to FP32 by *rounding*, at *bits*
    fraction bits (to nearest, below 23); a random sign drawn from *signs*."""
    m, k = qa.elements.shape
    n = qw.elements.shape[0]
    y = torch.empty(m, n, device=qa.elements.device)
    cols = max(1, _ELEMS // k)
    for j in range(0, n, cols):
        wd = qw.dequantize(slice(j, j + cols)).T
        for i in range(0, m, _ROWS):
            p = qa.dequantize(slice(i, i + _ROWS)) @ wd
            y[i:i + _ROWS, j:j + cols] = _round(_fraction(p, bits) if bits < 23 else p, rounding, signs)
    return y


def _fraction(p: torch.Tensor, bits: int) -> torch.Tensor:
    """FP64 *p* rounded to nearest, ties to even, at *bits* fraction bits;
    exact in FP32 for *bits* under 23 (in FP32's normal range)."""
    m, e = torch.frexp(p)
    return torch.ldexp(torch.round(m * 2.0 ** (bits + 1)), e - (bits + 1))


def _round(p: torch.Tensor, rounding: Rounding, signs: torch.Generator) -> torch.Tensor:
    """FP64 *p* rounded to FP32 by *rounding*, a random sign from *signs*."""
    f = p.float()
    if rounding == 'toward-zero':
        return torch.where(f.double().abs() > p.abs(), torch.nextafter(f, torch.zeros_like(f)), f)
    if rounding == 'random-sign':
        up = torch.rand(f.shape, device=f.device, generator=signs) < 0.5
        return torch.nextafter(f, torch.where(up, torch.inf, -torch.inf))
    return f


def slices(scheme: quant.Scheme, split_k: int) -> int:
    """How many slices of `k` to prepare a weight in: *split_k*, which a scaled
    scheme refuses (`kernels.matmul` takes its blocks whole)."""
    if split_k != 1 and scheme.applied in ('k-blocks', 'instruction'):
        raise ValueError(f'{scheme.name} splits `k` into its own blocks, not {split_k}')
    return split_k


def _per_group(q: quant.Quantized, design: str) -> torch.Tensor:
    """*q*'s block scales as a block-scaled *design* takes them: `[rows, k /
    g]`, one per group `g` of its instructions in order along `k`, each scale
    repeated over the groups its block covers."""
    g = kernels.group(design)
    return q.scales.repeat_interleave(q.operand.scaling.cols // g, 1).to(kernels.storage(design)[2])


def gemm(design: str, scheme: quant.Scheme, qa: quant.Quantized, qw: quant.Quantized,
         w: torch.Tensor, combine: kernels.Combine = 'linear') -> torch.Tensor:
    """`qa @ qw.T` by *design* under *scheme*, FP32; *w* is *qw*'s elements
    prepared in :func:`slices`.  Scales apply per `scheme.applied`: after
    the GEMM, `acc * s_x[i] * s_w[j]`; to each block of `k`'s partial; or in
    the instructions, then per tensor, `acc * (g_x * g_w)`."""
    a = qa.elements.to(kernels.storage(design)[0])
    if scheme.applied == 'instruction':
        y = kernels.matmul(a, w, design, scales=(_per_group(qa, design), _per_group(qw, design)))
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


_FP8 = (torch.float8_e4m3fn, torch.float8_e4m3fnuz, torch.float8_e5m2, torch.float8_e5m2fnuz)
"""Element types a recorded input is held in, losslessly, to save memory."""


@dataclass
class Frozen:
    """One forward pass's linear-layer inputs, raw and quantized, in call
    order; replayed, what the replaying run's own inputs would round
    differently."""

    inputs: list[tuple[torch.Tensor, quant.Quantized]] = field(default_factory=list)
    at: int = 0
    rerounded: torch.Tensor | int = 0
    """Elements the replaying run would round differently (on the GPU, read
    once a pass), of :attr:`elements`."""
    elements: int = 0
    rescaled: torch.Tensor | int = 0
    """Scales it would compute differently, of :attr:`scales`."""
    scales: int = 0
    _last: quant.Quantized | None = field(default=None, repr=False)

    def add(self, x: torch.Tensor, qa: quant.Quantized) -> None:
        """Record *x* and its quantized *qa*, once for layers sharing *qa*
        (`Run.quantize` reuses it)."""
        if qa is self._last:
            self.inputs.append(self.inputs[-1])
            return
        dtype = quant.DTYPES[qa.operand.elements]
        e = qa.elements.to(dtype) if dtype in _FP8 else qa.elements
        self.inputs.append((x.detach().reshape(-1, x.shape[-1]).clone(), replace(qa, elements=e)))
        self._last = qa

    def replay(self) -> 'Frozen':
        """These inputs, to replay from the first."""
        return Frozen(self.inputs)

    def next(self, own: Callable[[], quant.Quantized]) -> tuple[torch.Tensor, quant.Quantized]:
        """The next recorded input and its quantized form; counted against
        *own*, the replaying run's quantized input, once per recorded input."""
        entry = self.inputs[self.at]
        fresh = self.at == 0 or self.inputs[self.at - 1] is not entry
        self.at += 1
        x, qa = entry
        qa = replace(qa, elements=qa.elements.float())
        if fresh:
            mine = own()
            self.rerounded = self.rerounded + (mine.elements != qa.elements).sum()
            self.elements += qa.elements.numel()
            if qa.scales is not None:
                self.rescaled = self.rescaled + (mine.scales != qa.scales).sum()
                self.scales += qa.scales.numel()
        return x, qa


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
    record: Frozen | None = None
    """While set, each layer's input is recorded on it."""
    frozen: Frozen | None = None
    """While set, each layer replays it (:class:`Frozen`)."""
    signs: torch.Generator = field(default_factory=lambda: torch.Generator('cuda').manual_seed(0),
                                   compare=False, repr=False)
    """The random-sign rounding's generator (:meth:`reseed`)."""
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
        split = slices(self.scheme, self.split_k)
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
        # BF16 elements need no more rounding, nor quantize's check (a sync)
        qa = (quant.Quantized(self.scheme.x, a) if self.scheme.x == BF16.x
              else quant.quantize(a, self.scheme.x, scale, amax))
        self._input = (x, key, qa)
        return qa

    def linear(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        if self.mode == 'fp32' or id(w) in self.ignore:
            return F.linear(x, w)
        qw = self.weight(w)
        if self.frozen is not None:
            xr, qa = self.frozen.next(lambda: self.quantize(x, w))
            y = self._matmul(qa, qw, w) + (x.reshape(xr.shape) - xr) @ qw.dequantize(dtype=torch.float32).T
        else:
            qa = self.quantize(x, w)
            if self.record is not None:
                self.record.add(x, qa)
            y = self._matmul(qa, qw, w)
        return y.reshape(*x.shape[:-1], w.shape[0])

    def reseed(self, seed: int, sequence: int) -> None:
        """Draw the random-sign rounding afresh for *sequence* under *seed*:
        a pass over the same sequence draws the same signs, whatever ran
        before it."""
        self.signs.manual_seed(seed * 1_000_003 + sequence)

    def _matmul(self, qa: quant.Quantized, qw: quant.Quantized, w: torch.Tensor) -> torch.Tensor:
        if (r := _exact_run(self.mode, self.scheme)) is not None:
            return exact(qa, qw, self.signs, *r)
        if self.mode in kernels.TILES and kernels.applicable(self.mode, self.scheme):
            held = kernels.storage(self.mode)[1]
            return gemm(self.mode, self.scheme, qa, qw, self._prepare(w, qw, held), self.combine)
        raise ValueError(f'{self.mode!r} is not one of {modes(self.scheme)}')


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


def load(name: str = MODEL) -> tuple[torch.nn.Module, Run]:
    """*name*'s causal LM from the Hub, in FP32 on the GPU and patched
    (:func:`patch`)."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).cuda().eval()
    return model, patch(model)
