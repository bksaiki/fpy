"""
The tabulars of the paper's table of MMA-Sim's units, marking which FPy
compiles, from a compilation manifest, with the lines of FPy each unit's
model takes.  The `table` around them, and its caption, are the paper's.

    python evaluation/common/mma/designs_table.py                # to stdout
    python evaluation/common/mma/designs_table.py -o designs.tex
    python evaluation/common/mma/designs_table.py MANIFEST

The manifest is `generate_models.py`'s (`models` and `unsupported`) or
`compile.py -o`'s (`models` and `refused`); by default, the one the
`fpy2_models` build leaves.  A row is marked compiled when every design it
lists does, refused when any is; a design the manifest does not mention is
an error, as is one it mentions that no row lists.

A row's lines of code are those of every FPy function its designs reach
from their entry, each counted once: from the `def` line to the function's
end, without its decorator, docstring, comments or blank lines.  The Python
around them, which computes their constants, is not counted.
"""

import argparse
import ast
import io
import json
import sys
import tokenize
from functools import cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'examples' / 'mmasim'))

from compile import DESIGNS

import fpy2 as fp

MANIFEST = (Path(__file__).resolve().parents[1]
            / 'fpy2_models' / 'build' / 'generated_models' / 'manifest.json')
"""The manifest the `fpy2_models` build writes."""

_FP8 = ['e4m3', 'e5m2', 'e4m3.e5m2']
_F8F6F4 = _FP8 + ['e3m2', 'e2m3', 'e3m2.e2m3', 'e2m1']
_CDNA3_FP8 = ['fp8', 'bf8', 'fp8.bf8']

_FP64 = ('FP64', 'FP64', 1, ['fp64 (fma)'])
_FP32 = ('FP32', 'FP32', 1, ['fp32 (fma)'])

# Hopper's, also Blackwell's and RTX Blackwell's
_HOPPER = [
    _FP64,
    ('TF32', 'FP32', 8, ['nv.hopper.tf32.f32']),
    ('BF16/FP16', 'FP32', 16, ['nv.hopper.bf16.f32', 'nv.hopper.f16.f32']),
    ('FP16', 'FP16', 16, ['nv.hopper.f16.f16']),
]
# Blackwell's, also RTX Blackwell's
_BLACKWELL = [
    *_HOPPER,
    ('FP8/6/4', 'FP32', 32, [f'nv.blackwell.{t}.f32' for t in _F8F6F4]),
    ('FP8/6/4', 'FP16', 32, [f'nv.blackwell.{t}.f16' for t in _F8F6F4]),
    ('MXFP8/6/4', 'FP32', 32, [f'nv.blackwell.mx.{t}' for t in _F8F6F4]),
    ('MXFP4/NVFP4', 'FP32', 64, ['nv.blackwell.mxfp4', 'nv.blackwell.nvfp4']),
]

# (architecture, rows): each row (input, output, width, the designs in
# `compile.DESIGNS` that model it); a design shared by several
# architectures is listed under each
ARCHS: list[tuple[str, list[tuple[str, str, int, list[str]]]]] = [
    ('Volta', [
        ('FP16', 'FP32', 4, ['nv.volta.f16.f32']),
        ('FP16', 'FP16', 4, ['nv.volta.f16.f16']),
    ]),
    ('Turing', [
        ('FP16', 'FP32', 8, ['nv.turing.f16.f32']),
        ('FP16', 'FP16', 8, ['nv.turing.f16.f16']),
    ]),
    ('Ampere', [
        _FP64,
        ('TF32', 'FP32', 4, ['nv.ampere.tf32.f32']),
        ('BF16', 'FP32', 8, ['nv.ampere.bf16.f32']),
        ('FP16', 'FP32', 8, ['nv.turing.f16.f32']),
        ('FP16', 'FP16', 8, ['nv.turing.f16.f16']),
    ]),
    ('Ada Lovelace', [
        _FP64,
        ('TF32', 'FP32', 4, ['nv.ampere.tf32.f32']),
        ('BF16/FP16', 'FP32', 8, ['nv.ampere.bf16.f32', 'nv.turing.f16.f32']),
        ('FP16', 'FP16', 8, ['nv.turing.f16.f16']),
        ('FP8', 'FP32', 16, [f'nv.ada.{t}.f32' for t in _FP8]),
        ('FP8', 'FP16', 16, [f'nv.ada.{t}.f16' for t in _FP8]),
    ]),
    ('Hopper', [
        *_HOPPER,
        ('FP8', 'FP32', 32, [f'nv.hopper.{t}.f32' for t in _FP8]),
        ('FP8', 'FP16', 32, [f'nv.hopper.{t}.f16' for t in _FP8]),
    ]),
    ('Blackwell', _BLACKWELL),
    ('RTX Blackwell', _BLACKWELL),
    ('CDNA1', [
        _FP32,
        ('BF16', 'FP32', 2, ['amd.cdna1.bf16']),
        ('FP16', 'FP32', 4, ['amd.cdna1.f16']),
    ]),
    ('CDNA2', [
        _FP64,
        _FP32,
        ('BF16', 'FP32', 2, ['amd.cdna2.bf16']),
        ('FP16', 'FP32', 4, ['amd.cdna2.f16']),
        (r'BF16 (\texttt{\_1k})', 'FP32', 4, ['amd.cdna2.bf16_1k']),
    ]),
    ('CDNA3', [
        _FP64,
        _FP32,
        ('TF32', 'FP32', 4, ['amd.cdna3.tf32']),
        ('BF16/FP16', 'FP32', 8, ['amd.cdna3.bf16', 'amd.cdna3.f16']),
        ('FP8', 'FP32', 16, [f'amd.cdna3.{t}' for t in _CDNA3_FP8]),
    ]),
]

SPLIT = 'RTX Blackwell'
"""The architecture the table's second column starts at."""

_YES, _NO = r'\ding{51}', r'\xmark{}'


def load(path: Path) -> tuple[set[str], set[str]]:
    """The designs *path* lists as compiled, and as refused."""
    manifest = json.loads(path.read_text())
    refused = manifest.get('unsupported', manifest.get('refused'))
    if refused is None:
        raise ValueError(f'{path}: neither `unsupported` nor `refused`: not a manifest')
    return {m['name'] for m in manifest['models']}, set(refused)


@cache
def _functions(path: str) -> dict[int, int]:
    """The lines of code of each function in *path*, by every line from its
    first decorator to its `def`: from the `def` to its end, without blank
    lines, comments or docstrings."""
    src = Path(path).read_text()
    skip = (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT,
            tokenize.DEDENT, tokenize.ENDMARKER)
    code = {line for t in tokenize.generate_tokens(io.StringIO(src).readline)
            if t.type not in skip for line in range(t.start[0], t.end[0] + 1)}
    nodes = [n for n in ast.walk(ast.parse(src))
             if isinstance(n, (ast.Module, ast.FunctionDef))]
    for n in nodes:
        if ast.get_docstring(n, clean=False) is not None:
            code -= set(range(n.body[0].lineno, n.body[0].end_lineno + 1))
    loc = {}
    for n in nodes:
        if isinstance(n, ast.FunctionDef):
            count = sum(n.lineno <= line <= n.end_lineno for line in code)
            start = n.decorator_list[0].lineno if n.decorator_list else n.lineno
            loc.update(dict.fromkeys(range(start, n.lineno + 1), count))
    return loc


@cache
def _sources(name: str) -> frozenset[tuple[str, int]]:
    """Where each FPy function design *name* reaches is defined."""
    module = fp.Module()
    module.add(dict(DESIGNS)[name]()[0])
    return frozenset((f.ast.loc.source, f.ast.loc.start_line) for f in module.functions())


def lines_of_code(designs: list[str]) -> int:
    """The lines of FPy *designs* take together, a function they share
    counted once."""
    return sum(_functions(path)[line]
               for path, line in set().union(*map(_sources, designs)))


def check(compiled: set[str], refused: set[str]) -> None:
    """Raises if a row's design is missing from the manifest, or a design in
    it is in no row."""
    listed = {d for _, rows in ARCHS for *_, ds in rows for d in ds}
    missing = sorted(listed - compiled - refused)
    unlisted = sorted((compiled | refused) - listed)
    if missing:
        raise ValueError(f'not in the manifest: {", ".join(missing)}')
    if unlisted:
        raise ValueError(f'in no row of the table: {", ".join(unlisted)}')


def tabular(archs: list, compiled: set[str]) -> list[str]:
    lines = [r'\begin{tabular}[t]{@{}lllrrc@{}}', r'\toprule',
             r'Arch. & Input & Output & L & LoC & \name \\']
    for arch, rows in archs:
        lines.append(r'\midrule')
        for i, (a, c, width, designs) in enumerate(rows):
            mark = _YES if all(d in compiled for d in designs) else _NO
            loc = lines_of_code(designs)
            lines.append(f'{arch if i == 0 else "":14} & {a} & {c} & {width} & {loc} & {mark} \\\\')
    return [*lines, r'\bottomrule', r'\end{tabular}']


def latex(compiled: set[str]) -> str:
    at = next(i for i, (arch, _) in enumerate(ARCHS) if arch == SPLIT)
    return '\n'.join([
        *tabular(ARCHS[:at], compiled), r'\hfill', *tabular(ARCHS[at:], compiled), ''])


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument('manifest', nargs='?', type=Path, default=MANIFEST,
                    help='the compilation manifest (default: the fpy2_models build\'s)')
    ap.add_argument('-o', '--out', type=Path, help='write the table here, not to stdout')
    args = ap.parse_args(argv)
    if not args.manifest.exists():
        ap.error(f'no manifest at {args.manifest}: build fpy2_models, or run compile.py -o')

    compiled, refused = load(args.manifest)
    try:
        check(compiled, refused)
    except ValueError as ex:
        print(f'{args.manifest}: {ex}', file=sys.stderr)
        return 1
    tex = latex(compiled)
    if args.out is None:
        sys.stdout.write(tex)
    else:
        args.out.write_text(tex)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
