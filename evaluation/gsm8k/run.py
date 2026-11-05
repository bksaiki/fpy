"""
Times GSM8K on Qwen3.5-0.8B with every linear layer computed by a design:
by FPy's kernel (`fpy`) and by MMA-Sim (`mmasim`), against FP32 (`torch`).

    python evaluation/gsm8k/run.py                  # Hopper BF16, 32 problems, about 3.5 h
    python evaluation/gsm8k/run.py --smoke          # 2 problems, 8 tokens
    python evaluation/gsm8k/run.py -n 1319 --backends torch fpy

Replaces the results directory (`evaluation/results/gsm8k/`, or `-o DIR`)
with `torch.json`, `<design>-<backend>.json` and `report.py`'s tables.
Exits 1 if `fpy` and `mmasim` emit different tokens.
"""

import argparse
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
import torch
from core import generate, kernels, swap

RESULTS = Path(__file__).resolve().parents[1] / 'results' / 'gsm8k'
BACKENDS = ('torch', 'fpy', 'mmasim')


def mode(backend: str, design: str | None) -> str:
    """*backend*'s `Run` mode for *design*."""
    return {'torch': 'fp32', 'fpy': design, 'mmasim': mmasim_matmul.PREFIX + str(design)}[backend]


def measure(model: torch.nn.Module, run: mmasim_matmul.Run, tok: Any,
            problems: list[gsm8k.Problem], backend: str, design: str | None,
            max_new: int, batch: int) -> dict[str, Any]:
    """*backend*'s record on *problems*, in batches of *batch*."""
    run.mode = mode(backend, design)
    eos = generate.stop_tokens(model, tok)
    prompts = [generate.encode(tok, gsm8k.messages(p.question))[0].tolist() for p in problems]
    warmup = time.perf_counter()
    if backend != 'mmasim':   # compiles and loads kernels outside the timing
        batched.generate(model, prompts[:batch], 2, eos)
    warmup = time.perf_counter() - warmup
    tokens: list[list[int]] = []
    batches = []
    for i in range(0, len(prompts), batch):
        g = batched.generate(model, prompts[i:i + batch], max_new, eos)
        tokens += g.tokens
        batches.append({'first': i, 'shared': batched.shared(prompts[i:i + batch]),
                        **{k: v for k, v in dataclasses.asdict(g).items() if k != 'tokens'}})
        print(f'  {backend}: {len(tokens)}/{len(prompts)}  '
              f'prefill {sum(b["prefill"] for b in batches):.1f}s  '
              f'decode {sum(b["decode"] for b in batches):.1f}s', flush=True)
    replies = [tok.decode(t, skip_special_tokens=True) for t in tokens]
    return {
        'backend': backend, 'design': design, 'model': model.config.name_or_path,
        'problems': len(problems), 'max_new': max_new, 'batch': batch,
        'prefill': sum(b['prefill'] for b in batches), 'decode': sum(b['decode'] for b in batches),
        'warmup': warmup,
        'compile': kernels.compile_seconds(design) if backend == 'fpy' else None,
        'batches': batches, 'questions': [p.question for p in problems],
        'prompt_tokens': [len(p) for p in prompts], 'tokens': tokens, 'replies': replies,
        'answers': {m: list(map(f, replies)) for m, f in gsm8k.METRICS.items()},
        'targets': [p.target for p in problems],
    }


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
    ap.add_argument('designs', nargs='*', default=list(config.GSM8K_DESIGNS),
                    choices=sorted(mmasim_matmul.OPERATIONS), metavar='DESIGN',
                    help=f'designs (default {" ".join(config.GSM8K_DESIGNS)}; of {", ".join(mmasim_matmul.OPERATIONS)})')
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
    ap.add_argument('--smoke', action='store_true', help='2 problems, 8 tokens')
    args = ap.parse_args(argv)
    if args.smoke:
        args.problems, args.max_new = 2, 8
    if foreign := _foreign(args.out):
        ap.error(f'{args.out} holds files a run did not write: {", ".join(foreign)}')
    shutil.rmtree(args.out, ignore_errors=True)
    args.out.mkdir(parents=True)

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
        records[backend, design] = r = measure(model, run, tok, problems, backend, design,
                                               args.max_new, args.batch)
        r |= {'seconds': time.perf_counter() - start, **env}
        (args.out / f'{name}.json').write_text(json.dumps(r, indent=1))

    print('-- report')
    report.main([str(args.out)])
    rs = report.rows(list(records.values()))
    return 1 if any(r.same is not None and r.same < r.problems for r in rs) else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
