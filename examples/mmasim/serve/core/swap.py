"""
Replace how a model's `nn.Linear` layers compute, one run at a time.

A run computes under a quantization scheme (`quant`, default `bf16`):

- `fp32`: the model in full precision, `x @ W.T` in FP32 (TF32 off).
- `<scheme>-exact` (`bf16-exact`, ...): `x` (from BF16) and `W` quantized
  by the scheme, their product in FP64, rounded once to FP32.
- `<scheme>-exact-rand`: the exact run moved an ulp up or down, at random
  per element (by :meth:`Run.reseed`'s generator): a perturbation of it
  without bias, the plots' floor.
- a design the scheme applies to (`kernels.designs`): the quantized
  elements through its kernel, the scheme's scales applied after it or by
  its instructions (:func:`gemm`).

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
from typing import Literal

import torch
import torch.nn.functional as F

from . import kernels, quant

BF16 = quant.SCHEMES['bf16']


def modes(scheme: quant.Scheme = BF16) -> tuple[str, ...]:
    """*scheme*'s runs: R0, its exact runs (`-exact`, `-rand`), and every
    design it applies to."""
    return ('fp32', *(f'{scheme.name}-exact{s}' for s in _EXACT), *kernels.designs(scheme))


MODELS = ('Qwen/Qwen3.5-0.8B', 'Qwen/Qwen3-0.6B')
"""The models these scripts run, BF16 masters from the Hub."""

MODEL = MODELS[0]

_ROWS = 256
"""Rows per block of an exact run's FP64 product."""

_ELEMS = 1 << 24
"""FP64 weight elements per block of an exact run's columns."""


Rounding = Literal['nearest', 'random-sign']
"""How an exact run rounds: to nearest, or to nearest, then an ulp up or
down at random."""

_EXACT: dict[str, Rounding] = {'': 'nearest', '-rand': 'random-sign'}
"""Each exact run's rounding, by its name's suffix."""


def _exact_run(mode: str, scheme: quant.Scheme) -> Rounding | None:
    """How *mode* rounds, if an exact run."""
    prefix = f'{scheme.name}-exact'
    return _EXACT.get(mode.removeprefix(prefix)) if mode.startswith(prefix) else None


def exact(qa: quant.Quantized, qw: quant.Quantized, signs: torch.Generator,
          rounding: Rounding = 'nearest') -> torch.Tensor:
    """`qa @ qw.T` in FP64, rounded once to FP32 by *rounding*; a random sign
    drawn from *signs*."""
    m, k = qa.elements.shape
    n = qw.elements.shape[0]
    y = torch.empty(m, n, device=qa.elements.device)
    cols = max(1, _ELEMS // k)
    for j in range(0, n, cols):
        wd = qw.dequantize(slice(j, j + cols)).T
        for i in range(0, m, _ROWS):
            p = qa.dequantize(slice(i, i + _ROWS)) @ wd
            y[i:i + _ROWS, j:j + cols] = _round(p, rounding, signs)
    return y


def _round(p: torch.Tensor, rounding: Rounding, signs: torch.Generator) -> torch.Tensor:
    """FP64 *p* rounded to FP32 by *rounding*, a random sign from *signs*."""
    f = p.float()
    if rounding == 'random-sign':
        up = torch.rand(f.shape, device=f.device, generator=signs) < 0.5
        return torch.nextafter(f, torch.where(up, torch.inf, -torch.inf))
    return f


def _per_group(q: quant.Quantized, design: str) -> torch.Tensor:
    """*q*'s block scales as a block-scaled *design* takes them: `[rows, k /
    g]`, one per group `g` of its instructions in order along `k`, each scale
    repeated over the groups its block covers."""
    g = kernels.group(design)
    return q.scales.repeat_interleave(q.operand.scaling.cols // g, 1).to(kernels.storage(design)[2])


def gemm(design: str, scheme: quant.Scheme, qa: quant.Quantized, qw: quant.Quantized,
         w: torch.Tensor) -> torch.Tensor:
    """`qa @ qw.T` by *design* under *scheme*, FP32; *w* is *qw*'s elements
    in *design*'s storage (`kernels.storage`).  Scales apply per `scheme.applied`: after
    the GEMM, `acc * s_x[i] * s_w[j]`; or in the instructions, then per
    tensor, `acc * (g_x * g_w)`."""
    a = qa.elements.to(kernels.storage(design)[0])
    if scheme.applied == 'instruction':
        y = kernels.matmul(a, w, design, scales=(_per_group(qa, design), _per_group(qw, design)))
        return y if qa.tensor is None else y.mul_(qa.tensor * qw.tensor)
    y = kernels.matmul(a, w, design)
    return y.mul_(qa.scales).mul_(qw.scales.T) if scheme.applied == 'epilogue' else y


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
    is quantized by the scheme's RTN; a layer whose weight is in `ignore`
    computes as R0 in every run."""

    mode: str = 'fp32'
    scheme: quant.Scheme = BF16
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
    _prepared: dict[int, tuple[quant.Quantized, torch.dtype, torch.Tensor]] = field(
        default_factory=dict, compare=False, repr=False)
    """By weight `id`, its last quantized form, storage, and those elements
    in it."""
    _input: tuple[torch.Tensor, tuple[int, str], quant.Quantized] | None = field(
        default=None, compare=False, repr=False)
    """The last input, its version and scheme, and its quantized form."""

    def weight(self, w: torch.Tensor) -> quant.Quantized:
        """*w* quantized under the scheme (once per version)."""
        key = (id(w), w._version, self.scheme.name)
        if key not in self._weights:
            self._weights = {k: v for k, v in self._weights.items() if k[0] != id(w)}
            self._weights[key] = (w, quant.quantize(w, self.scheme.w))
        return self._weights[key][1]

    def _prepare(self, w: torch.Tensor, qw: quant.Quantized, dtype: torch.dtype) -> torch.Tensor:
        got = self._prepared.get(id(w))
        if got is None or got[0] is not qw or got[1] != dtype:
            got = self._prepared[id(w)] = (qw, dtype, qw.elements.to(dtype))
        return got[2]

    def quantize(self, x: torch.Tensor) -> quant.Quantized:
        """*x*, a layer's input, rounded to BF16 and quantized, reused while the
        same tensor comes back unchanged: `q/k/v_proj` share one, as do
        `gate/up_proj`."""
        key = (x._version, self.scheme.name)
        last = self._input
        if last is not None and last[0] is x and last[1] == key:
            return last[2]
        a = x.reshape(-1, x.shape[-1]).to(torch.bfloat16).float()
        # BF16 elements need no more rounding, nor quantize's check (a sync)
        qa = (quant.Quantized(self.scheme.x, a) if self.scheme.x == BF16.x
              else quant.quantize(a, self.scheme.x))
        self._input = (x, key, qa)
        return qa

    def linear(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        if self.mode == 'fp32' or id(w) in self.ignore:
            return F.linear(x, w)
        qw = self.weight(w)
        if self.frozen is not None:
            xr, qa = self.frozen.next(lambda: self.quantize(x))
            y = self._matmul(qa, qw, w) + (x.reshape(xr.shape) - xr) @ qw.dequantize(dtype=torch.float32).T
        else:
            qa = self.quantize(x)
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
            return exact(qa, qw, self.signs, r)
        if self.mode in kernels.TILES and kernels.applicable(self.mode, self.scheme):
            held = kernels.storage(self.mode)[1]
            return gemm(self.mode, self.scheme, qa, qw, self._prepare(w, qw, held))
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


def use(run: Run, model: torch.nn.Module, scheme: quant.Scheme) -> None:
    """Set *run*'s scheme; under a quantizing one, `lm_head` computes as R0."""
    run.scheme = scheme
    run.ignore = set() if scheme.name == BF16.name else {id(dict(model.named_modules())['lm_head'].weight)}


def load(name: str = MODEL, scheme: quant.Scheme = BF16) -> tuple[torch.nn.Module, Run]:
    """*name*'s causal LM from the Hub, in FP32 on the GPU, patched
    (:func:`patch`) and under *scheme* (:func:`use`)."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).cuda().eval()
    run = patch(model)
    use(run, model, scheme)
    return model, run
