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
from fpy2.ast.visitor import DefaultVisitor

MANIFEST = (Path(__file__).resolve().parents[1]
            / 'fpy2_models' / 'build' / 'generated_models' / 'manifest.json')
"""The manifest the `fpy2_models` build writes."""

_FP8 = ['e4m3', 'e5m2', 'e4m3.e5m2']
_F8F4 = _FP8 + ['e2m1', 'e4m3.e2m1', 'e5m2.e2m1']
_CDNA3_FP8 = ['fp8', 'bf8', 'fp8.bf8']

_FP64 = ('FP64', 'FP64', 1, ['fp64 (fma)'])
_FP32 = ('FP32', 'FP32', 1, ['fp32 (fma)'])

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
        _FP64,
        ('TF32', 'FP32', 8, ['nv.hopper.tf32.f32']),
        ('BF16/FP16', 'FP32', 16, ['nv.hopper.bf16.f32', 'nv.hopper.f16.f32']),
        ('FP16', 'FP16', 16, ['nv.hopper.f16.f16.wgmma', 'nv.hopper.f16.f16.mma']),
        ('FP8', 'FP32', 32, [f'nv.hopper.{t}.f32' for t in _FP8]),
        ('FP8', 'FP16', 32, [f'nv.hopper.{t}.f16' for t in _FP8]),
    ]),
    ('Blackwell', [
        _FP64,
        ('TF32', 'FP32', 8, ['nv.hopper.tf32.f32']),
        ('BF16/FP16', 'FP32', 16, ['nv.hopper.bf16.f32', 'nv.hopper.f16.f32']),
        ('FP16', 'FP16', 16, ['nv.hopper.f16.f16.wgmma', 'nv.hopper.f16.f16.mma']),
        ('FP8/4', 'FP32', 32, [f'nv.blackwell.{t}.f32' for t in _F8F4]),
        ('FP8/4', 'FP16', 32, [f'nv.blackwell.{t}.f16.tcgen05' for t in _F8F4]),
        ('MXFP8/4', 'FP32', 32, [f'nv.blackwell.mx.{t}' for t in _F8F4]),
        ('MXFP4/NVFP4', 'FP32', 64, ['nv.blackwell.mxfp4', 'nv.blackwell.nvfp4']),
    ]),
    ('RTX Blackwell', [
        _FP64,
        ('TF32', 'FP32', 8, ['nv.hopper.tf32.f32']),
        ('BF16/FP16', 'FP32', 16, ['nv.hopper.bf16.f32', 'nv.hopper.f16.f32']),
        ('FP16', 'FP16', 16, ['nv.hopper.f16.f16.mma']),
        ('FP8/4', 'FP32', 32, [f'nv.blackwell.{t}.f32' for t in _F8F4]),
        ('FP8/4', 'FP16', 32, [f'nv.rtx_blackwell.{t}.f16.mma' for t in _F8F4]),
        ('MXFP8/4', 'FP32', 32, [f'nv.blackwell.mx.{t}' for t in _F8F4]),
        ('MXFP4/NVFP4', 'FP32', 64, ['nv.blackwell.mxfp4', 'nv.blackwell.nvfp4']),
    ]),
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


class _Callees(DefaultVisitor):
    """The FPy functions a function calls directly."""

    def __init__(self):
        self.fns: list[fp.Function] = []

    def _visit_call(self, e, ctx):
        if isinstance(e.fn, fp.Function):
            self.fns.append(e.fn)
        super()._visit_call(e, ctx)


@cache
def _code_lines(path: str) -> tuple[frozenset[int], ast.Module]:
    """The lines of *path* holding code, not a comment, docstring or blank,
    and its syntax tree."""
    src = Path(path).read_text()
    tree = ast.parse(src)
    docs = set()
    for node in ast.walk(tree):
        if (isinstance(node, (ast.Module, ast.FunctionDef)) and node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)):
            docs.update(range(node.body[0].lineno, node.body[0].end_lineno + 1))
    skip = (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT,
            tokenize.DEDENT, tokenize.ENDMARKER)
    code = {line for t in tokenize.generate_tokens(io.StringIO(src).readline)
            if t.type not in skip for line in range(t.start[0], t.end[0] + 1)}
    return frozenset(code - docs), tree


def _function_lines(path: str, line: int) -> frozenset[int]:
    """The code lines of the function defined, decorator first, at *line*."""
    code, tree = _code_lines(path)
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        start = node.decorator_list[0].lineno if node.decorator_list else node.lineno
        if start <= line <= node.lineno:
            return frozenset(l for l in code if node.lineno <= l <= node.end_lineno)
    raise LookupError(f'no function at {path}:{line}')


@cache
def _sources(name: str) -> frozenset[tuple[str, int]]:
    """Where each FPy function design *name* reaches is defined."""
    seen: set[tuple[str, int]] = set()
    todo = [dict(DESIGNS)[name]()[0]]
    while todo:
        fn = todo.pop()
        where = (fn.ast.loc.source, fn.ast.loc.start_line)
        if where in seen:
            continue
        seen.add(where)
        callees = _Callees()
        callees._visit_function(fn.ast, None)
        todo.extend(callees.fns)
    return frozenset(seen)


def lines_of_code(designs: list[str]) -> int:
    """The lines of FPy *designs* take together, a function they share
    counted once."""
    return sum(len(_function_lines(*where))
               for where in set().union(*map(_sources, designs)))


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
