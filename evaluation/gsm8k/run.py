"""
Times GSM8K on Qwen3.5-0.8B with every linear layer computed by a design:
by FPy's kernel (`fpy`) and by MMA-Sim (`mmasim`), against FP32 (`torch`).

    python evaluation/gsm8k/run.py                  # every design, 32 problems: about 3 h
    python evaluation/gsm8k/run.py --mmasim full nv.hopper.bf16.f32   # MMA-Sim measured: about 3.6 h
    python evaluation/gsm8k/run.py --smoke          # 2 problems, 8 tokens
    python evaluation/gsm8k/run.py -n 1319 --backends torch fpy

FPy's kernels compile first, so every timing is of the compiled model.

Replaces the results directory (`evaluation/results/gsm8k/`, or `-o DIR`)
with `torch.json`, `<design>-<backend>.json` and `report.py`'s tables.
MMA-Sim's time is predicted per layer by default (`sample.py`), and also
sampled in the model for `--sampled` designs, to check the prediction.
Exits 1 if `fpy` and `mmasim` emit different tokens, or a call MMA-Sim ran
differs from FPy's kernel.
"""

import argparse
import contextlib
import dataclasses
import json
import platform
import shutil
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(_ROOT / 'evaluation' / 'common' / 'mma'), str(_ROOT / 'examples' / 'mmasim')]
sys.path.append(str(_ROOT / 'evaluation'))  # last: `config`, not shadowing this `run`

import batched
import config
import gsm8k
import mmasim_matmul
import report
import sample
import torch
from core import generate, kernels, swap

RESULTS = Path(__file__).resolve().parents[1] / 'results' / 'gsm8k'
BACKENDS = ('torch', 'fpy', 'mmasim')


def mode(backend: str, design: str | None) -> str:
    """*backend*'s `Run` mode for *design*."""
    return {'torch': 'fp32', 'fpy': design, 'mmasim': mmasim_matmul.PREFIX + str(design)}[backend]


def encode(tok: Any, problems: list[gsm8k.Problem]) -> list[list[int]]:
    """*problems*' prompts' tokens."""
    return [generate.encode(tok, gsm8k.messages(p.question))[0].tolist() for p in problems]


def measure(model: torch.nn.Module, run: mmasim_matmul.Run, tok: Any,
            problems: list[gsm8k.Problem], backend: str, design: str | None,
            max_new: int, batch: int) -> dict[str, Any]:
    """*backend*'s record on *problems*, in batches of *batch*."""
    run.mode = mode(backend, design)
    eos = generate.stop_tokens(model, tok)
    prompts = encode(tok, problems)
    warmup = time.perf_counter()
    # a tuned MMA-Sim's first decode step captures its graphs, inside the timing: prefill
    # then runs without them resident, as in its prediction
    decoding = (mmasim_matmul.replaying if backend == 'mmasim' and mmasim_matmul.TUNED
                else contextlib.nullcontext)
    if backend != 'mmasim':   # Triton's JIT and CUDA's start-up, outside the timing
        batched.generate(model, prompts[:batch], 2, eos)
    warmup = time.perf_counter() - warmup
    tokens: list[list[int]] = []
    batches = []
    for i in range(0, len(prompts), batch):
        g = batched.generate(model, prompts[i:i + batch], max_new, eos, decoding)
        tokens += g.tokens
        batches.append({'first': i, 'shared': batched.shared(prompts[i:i + batch]),
                        **{k: v for k, v in dataclasses.asdict(g).items() if k != 'tokens'}})
        print(f'  {backend}: {len(tokens)}/{len(prompts)}  '
              f'prefill {sum(b["prefill"] for b in batches):.1f}s  '
              f'decode {sum(b["decode"] for b in batches):.1f}s', flush=True)
    replies = [tok.decode(t, skip_special_tokens=True) for t in tokens]
    return {
        'backend': backend, 'design': design, 'model': model.config.name_or_path,
        'scheme': None if design is None else run.scheme.name,
        'baseline': _baseline() if backend == 'mmasim' else None,
        'problems': len(problems), 'max_new': max_new, 'batch': batch,
        'prefill': sum(b['prefill'] for b in batches), 'decode': sum(b['decode'] for b in batches),
        'warmup': warmup,
        'compile': kernels.compile_seconds(design) if backend == 'fpy' else None,
        'batches': batches, 'questions': [p.question for p in problems],
        'prompt_tokens': [len(p) for p in prompts], 'tokens': tokens, 'replies': replies,
        'answers': {m: list(map(f, replies)) for m, f in gsm8k.METRICS.items()},
        'targets': [p.target for p in problems],
    }


def mmasim(model: torch.nn.Module, run: mmasim_matmul.Run, tok: Any,
           problems: list[gsm8k.Problem], design: str, fpy: dict[str, Any], method: str,
           max_new: int, batch: int, sampled: bool) -> dict[str, Any]:
    """MMA-Sim's record for *design*: measured end to end (*method* `full`),
    or predicted per layer (`layers`); with the per-layer prediction either
    way, and the in-model sample if *sampled*.  *fpy* is FPy's record."""
    if method == 'full':
        r = measure(model, run, tok, problems, 'mmasim', design, max_new, batch)
        r['graphless'] = sample.graphless(design)
    else:
        r = {'backend': 'mmasim', 'design': design, 'model': model.config.name_or_path,
             'scheme': run.scheme.name, 'baseline': _baseline(), 'problems': fpy['problems'],
             'max_new': fpy['max_new'], 'batch': fpy['batch']}
    p = sample.prefilled(model, run, encode(tok, problems), design, fpy)
    lay = sample.layers(model, run, design, fpy, p)
    r |= {'method': method, 'layers': lay, 'check': lay['check']}
    if method == 'layers':
        r |= {k: lay[k] for k in ('prefix', 'prefill', 'decode')}
    print(f"  mmasim: predicted {lay['prefill'] + lay['decode']:.0f}s  check {lay['check']}", flush=True)
    if sampled:
        r['in_model'] = sample.in_model(model, run, design, fpy, p)
        print(f"  mmasim: in the model {r['in_model']['prefill'] + r['in_model']['decode']:.0f}s",
              flush=True)
    return r


def _baseline() -> str:
    return 'tuned' if mmasim_matmul.TUNED else 'released'


def _git(*args: str) -> str:
    return subprocess.run(['git', *args], capture_output=True, text=True, check=False,
                          cwd=Path(__file__).parent).stdout.strip()


def environment() -> dict[str, Any]:
    """The GPU, package versions, commit and command line."""
    versions = {}
    for d in ('torch', 'triton', 'transformers', 'datasets', 'mmasim'):
        try:
            versions[d] = metadata.version(d)
        except metadata.PackageNotFoundError:
            versions[d] = None
    return {'gpu': torch.cuda.get_device_name(), 'python': platform.python_version(),
            'versions': versions, 'commit': _git('rev-parse', 'HEAD'),
            'dirty': bool(_git('status', '--porcelain')), 'argv': sys.argv}


def _foreign(out: Path) -> list[str]:
    """What in *out* a run did not write, and so must not be deleted."""
    if not out.exists():
        return []
    return sorted(p.name for p in out.iterdir()
                  if not (p.suffix == '.json' or p.name.startswith(report.PREFIX)))


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument('designs', nargs='*', default=list(config.GSM8K_DESIGNS or mmasim_matmul.DESIGNS),
                    choices=mmasim_matmul.DESIGNS, metavar='DESIGN',
                    help='designs (default: every design both backends compute, `mmasim_matmul.DESIGNS`)')
    ap.add_argument('-o', '--out', metavar='DIR', type=Path, default=RESULTS,
                    help='the results directory, replaced (default %(default)s)')
    ap.add_argument('-n', '--problems', type=int, default=config.GSM8K_PROBLEMS,
                    help='test problems (default %(default)s)')
    ap.add_argument('--max-new', type=int, default=config.GSM8K_MAX_NEW,
                    help='tokens per reply at most (default %(default)s)')
    ap.add_argument('--batch', type=int, default=config.GSM8K_BATCH,
                    help='problems per batch (default %(default)s)')
    ap.add_argument('--backends', nargs='+', choices=BACKENDS, default=list(BACKENDS),
                    help='default: all')
    ap.add_argument('--model', default=config.MODEL, help='default %(default)s')
    ap.add_argument('--mmasim', choices=('layers', 'full'), default='layers',
                    help="MMA-Sim's time: predicted per layer, or measured over the whole "
                         'run (default %(default)s)')
    ap.add_argument('--mmasim-as', choices=('tuned', 'released'), default='tuned',
                    help='MMA-Sim tuned (no host waits, CUDA graphs; same bits), or as released '
                         '(default %(default)s)')
    ap.add_argument('--reuse', metavar='DIR', type=Path,
                    help="take `torch`'s and `fpy`'s records from DIR where there, rather than run them")
    ap.add_argument('--sampled', nargs='*', default=list(config.GSM8K_SAMPLED), metavar='DESIGN',
                    help='designs also sampled in the model, checking the prediction '
                         '(default %(default)s)')
    ap.add_argument('-j', '--jobs', type=int, default=1,
                    help='processes compiling the designs; above 1, their compile times '
                         'contend (default %(default)s)')
    ap.add_argument('--smoke', action='store_true', help='2 problems, 8 tokens')
    args = ap.parse_args(argv)
    if args.smoke:
        args.problems, args.max_new = 2, 8
    if 'mmasim' in args.backends and 'fpy' not in args.backends:
        ap.error("MMA-Sim's prediction starts from FPy's run: add --backends fpy")
    if foreign := _foreign(args.out):
        ap.error(f'{args.out} holds files a run did not write: {", ".join(foreign)}')
    shutil.rmtree(args.out, ignore_errors=True)
    args.out.mkdir(parents=True)

    mmasim_matmul.tune(args.mmasim_as == 'tuned')
    try:
        return _run(args)
    finally:
        mmasim_matmul.tune(False)


def _run(args: argparse.Namespace) -> int:
    """*args*' runs, records and report; whether every design agrees (0)."""
    fpy = args.designs if 'fpy' in args.backends else []
    if refused := kernels.compile_designs(fpy, args.jobs):
        sys.exit(f'refused: {refused}')

    from transformers import AutoTokenizer

    model, run = mmasim_matmul.load(args.model)
    tok = AutoTokenizer.from_pretrained(args.model)
    problems = gsm8k.problems(args.problems)
    runs = [('torch', None)] if 'torch' in args.backends else []
    runs += [(b, d) for d in args.designs for b in ('fpy', 'mmasim') if b in args.backends]
    env = environment()
    records = {}
    for backend, design in runs:
        name = backend if design is None else f'{design}-{backend}'
        print(f'-- {name}', flush=True)
        start = time.perf_counter()
        if design is not None and backend == 'fpy':
            run.release()
            swap.use(run, model, kernels.scheme(design))
            print(f'  {torch.cuda.memory_allocated() / 2**30:.2f} GiB allocated', flush=True)
        reused = None if args.reuse is None or backend == 'mmasim' else args.reuse / f'{name}.json'
        if reused is not None and reused.exists():
            r = json.loads(reused.read_text())
            print(f'  reused {reused}', flush=True)
        elif backend == 'mmasim':
            r = mmasim(model, run, tok, problems, design, records['fpy', design], args.mmasim,
                       args.max_new, args.batch, design in args.sampled)
        else:
            r = measure(model, run, tok, problems, backend, design, args.max_new, args.batch)
        records[backend, design] = r
        if reused is None or not reused.exists():
            r |= {'seconds': time.perf_counter() - start, **env}
        (args.out / f'{name}.json').write_text(json.dumps(r, indent=1))

    print('-- report')
    report.main([str(args.out)])
    rs = report.rows(list(records.values()))
    return 1 if any(r.disagrees for r in rs) else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
