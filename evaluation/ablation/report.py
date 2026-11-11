"""
The ablation's table, from `run.py`'s results; `run.py` calls it last.

    python evaluation/ablation/report.py evaluation/results/ablation

Writes `ablation-table.tex`, `ablation-paper.json` and `ablation-summary.txt`
beside the data, overwriting only those.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ANALYSES = {'fi': 'FI', 'fi+db': 'FI+DB'}
"""Each analysis configuration's label."""

_SPLIT, _UNFOLD = r'\Cref{ssec:rewrites:split}', r'\Cref{ssec:rewrites:unfold}'

REWRITES = {'none': 'None', 'split': _SPLIT, 'unfold': _UNFOLD, 'split+unfold': f'{_SPLIT} + {_UNFOLD}'}
"""Each rewrite configuration's column heading: the paper's sections."""

REASONS = {
    'datatype': 'storage selection failed',
    'operator': 'no matching signature',
    # the attribute is a rounding mode, for a context whose position is
    # computed: `with fp.MPFixedContext(n, rm)`
    'context': 'does not handle Attribute',
}
"""Why a cell refuses a design, and the compiler's words for it: no datatype
holds a value, no operator computes an operation, or a rounding context is
built at run time."""


def load(out: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """The run's settings, and every design record in *out* in the order the
    run measured them."""
    run = json.loads((out / 'ablation-run.json').read_text())
    by_name = {}
    for p in out.glob('*.json'):
        if not p.name.startswith('ablation-'):
            rec = json.loads(p.read_text())
            by_name[rec['design']] = rec
    return run, [by_name[n] for n in run['designs'] if n in by_name]


def verdict(cell: dict[str, Any]) -> str:
    """`ok`, or why the cell did not compile: one of `REASONS`, `other`,
    `timeout` or `error`."""
    if 'cpp_sha256' in cell:
        return 'ok'
    if 'refused' not in cell:
        return 'timeout' if 'timeout' in cell else 'error'
    return next((r for r, words in REASONS.items() if words in cell['refused']), 'other')


def tally(run: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Counter[str]]:
    """Each cell's verdicts, counted over *records*."""
    cells = [f'{a}/{r}' for a in run['analyses'] for r in run['rewrites']]
    return {c: Counter(verdict(rec['cells'][c]) for rec in records) for c in cells}


def tex_table(run: dict[str, Any], counts: dict[str, Counter[str]]) -> str:
    """Designs compiled per cell: a row per analysis, a column per rewrite set."""
    rows = [
        f'{ANALYSES[a]} & ' + ' & '.join(str(counts[f"{a}/{r}"]["ok"]) for r in run['rewrites']) + r' \\'
        for a in run['analyses']
    ]
    return '\n'.join([
        r'\begin{tabular}{l' + 'r' * len(run['rewrites']) + '}',
        r'\toprule',
        'Analyses & ' + ' & '.join(REWRITES[r] for r in run['rewrites']) + r' \\',
        r'\midrule',
        *rows,
        r'\bottomrule',
        r'\end{tabular}',
    ]) + '\n'


def summary(records: list[dict[str, Any]], counts: dict[str, Counter[str]]) -> str:
    """Each cell's designs compiled and refused, by reason, as text; and how
    many designs' C++ depends on the cell."""
    lines = [f'Designs compiled, of {len(records)}', f'{"":10}' + ''.join(f'{c:>20}' for c in counts)]
    for v in ('ok', *REASONS, 'other', 'timeout', 'error'):
        if any(n[v] for n in counts.values()):
            lines.append(f'{v:10}' + ''.join(f'{n[v]:>20}' for n in counts.values()))
    differ = sum(len({c['cpp_sha256'] for c in rec['cells'].values() if 'cpp_sha256' in c}) > 1
                 for rec in records)
    lines += ['', f'{differ} designs compile to different C++ in different cells']
    return '\n'.join(lines) + '\n'


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument('out', type=Path, help="`run.py`'s results directory")
    args = ap.parse_args(argv)
    run, records = load(args.out)
    counts = tally(run, records)
    (args.out / 'ablation-table.tex').write_text(tex_table(run, counts))
    (args.out / 'ablation-paper.json').write_text(json.dumps(
        {'designs': len(records), 'cells': {c: dict(n) for c, n in counts.items()}}, indent=1))
    text = summary(records, counts)
    (args.out / 'ablation-summary.txt').write_text(text)
    print(text, end='')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
