"""
Tables of `run.py`'s records.

    python evaluation/gsm8k/report.py evaluation/results/gsm8k

Writes, beside the records:

- `gsm8k-speed.tex`: per backend, time, throughput, accuracy (`gsm8k_cot`'s
  two metrics, scored from the replies), agreement with MMA-Sim and speedup
  over it: `torch`, then each design whose MMA-Sim run was measured;
- `gsm8k-designs.tex`: per design, FPy's time (measured), MMA-Sim's
  (predicted per layer, or measured), the speedup, and the bit check of the
  calls MMA-Sim ran; and the speedup's minimum, median and maximum;
- `gsm8k-check.tex`: the per-layer prediction against MMA-Sim's in-model
  sample, or its whole run;
- `gsm8k-speed.json`: every number in them;
- `gsm8k-summary.txt`: the same, as text.
"""

import argparse
import dataclasses
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gsm8k

PREFIX = 'gsm8k-'
"""This report's outputs' prefix."""

LABELS = {'torch': 'PyTorch (FP32)', 'fpy': 'FPy', 'mmasim': 'MMA-Sim'}
_ORDER = list(LABELS)


@dataclass(frozen=True)
class Row:
    design: str | None
    backend: str
    problems: int
    tokens: int | None
    """`None` if predicted."""
    prefill: float
    decode: float
    correct: dict[str, int] | None
    """Problems correct by each of `gsm8k.METRICS`; `None` if predicted."""
    same: int | None
    """Problems whose tokens equal MMA-Sim's, for the design's `fpy`, if
    MMA-Sim ran in full."""
    check: dict[str, int] | None
    """The bit check of the calls MMA-Sim ran (`calls`, `outputs`,
    `differing`, `finite`), for the design's `fpy`."""
    speedup: float | None
    """MMA-Sim's total time over this backend's, for the design's `fpy`."""
    predicted: bool = False

    @property
    def seconds(self) -> float:
        return self.prefill + self.decode

    @property
    def rate(self) -> float | None:
        """Tokens per second, prefill included."""
        return None if self.tokens is None else self.tokens / self.seconds

    @property
    def disagrees(self) -> bool:
        """Whether MMA-Sim's tokens, or the calls it ran, differ from this row's."""
        return ((self.same is not None and self.same < self.problems)
                or (self.check is not None and self.check['differing'] > 0))


def load(results: Path) -> list[dict[str, Any]]:
    """`run.py`'s records in *results*."""
    return [r for p in sorted(results.glob('*.json')) if not p.name.startswith(PREFIX)
            if 'backend' in (r := json.loads(p.read_text()))]


def _key(design: str | None, backend: str) -> tuple[bool, str, int]:
    return design is not None, design or '', _ORDER.index(backend)


def rows(records: list[dict[str, Any]]) -> list[Row]:
    """One row per record: `torch` first, then by design, `fpy` before
    `mmasim`."""
    by = {(r['design'], r['backend']): r for r in records}
    out = []
    for (design, backend), r in sorted(by.items(), key=lambda kv: _key(*kv[0])):
        ref = by.get((design, 'mmasim')) if backend == 'fpy' else None
        same = check = speedup = None
        if ref is not None:
            check = ref.get('check')
            if 'tokens' in ref:
                same = sum(a == b for a, b in zip(r['tokens'], ref['tokens'], strict=True))
            speedup = (ref['prefill'] + ref['decode']) / (r['prefill'] + r['decode'])
        ran = 'tokens' in r
        out.append(Row(design, backend, r['problems'], sum(map(len, r['tokens'])) if ran else None,
                       r['prefill'], r['decode'], gsm8k.scores(r['replies'], r['targets']) if ran else None,
                       same, check, speedup, predicted=not ran))
    return out


def measured(rs: list[Row]) -> list[Row]:
    """*rs*' rows for the main table: `torch`, and each design whose MMA-Sim
    run was measured, or absent."""
    predicted = {r.design for r in rs if r.predicted}
    return [r for r in rs if r.design not in predicted]


@dataclass(frozen=True)
class Design:
    """A design's FPy and MMA-Sim times, one row of the per-design table."""

    design: str
    scheme: str
    fpy: tuple[float, float]
    """FPy's prefill and decode seconds, measured."""
    mmasim: tuple[float, float]
    """MMA-Sim's, predicted per layer or measured."""
    measured: bool
    check: dict[str, int]

    @property
    def speedup(self) -> float:
        return sum(self.mmasim) / sum(self.fpy)


def designs(records: list[dict[str, Any]]) -> list[Design]:
    """A row per design with both records, by design."""
    by = {(r['design'], r['backend']): r for r in records}
    return [Design(d, f['scheme'], (f['prefill'], f['decode']), (m['prefill'], m['decode']),
                   'tokens' in m, m['check'])
            for (d, b), f in sorted(by.items(), key=lambda kv: kv[0][0] or '')
            if b == 'fpy' and (m := by.get((d, 'mmasim'))) is not None]


def summary(ds: list[Design]) -> dict[str, Any]:
    """The speedup's minimum, median and maximum over *ds*, and where."""
    if not ds:
        return {}
    by = sorted(ds, key=lambda d: d.speedup)
    return {'designs': len(ds), 'min': (by[0].speedup, by[0].design),
            'median': statistics.median(d.speedup for d in ds),
            'max': (by[-1].speedup, by[-1].design),
            'differing': sum(d.check['differing'] for d in ds)}


@dataclass(frozen=True)
class Check:
    """The per-layer prediction against a measurement of MMA-Sim."""

    design: str
    against: str
    """`in-model sample` or `full run`."""
    predicted: tuple[float, float]
    """Prefill and decode seconds, per layer."""
    reference: tuple[float, float]

    @property
    def errors(self) -> tuple[float, float, float]:
        """Prefill's, decode's and the total's, relative to the reference."""
        p, r = self.predicted, self.reference
        return p[0] / r[0] - 1, p[1] / r[1] - 1, sum(p) / sum(r) - 1


def checks(records: list[dict[str, Any]]) -> list[Check]:
    """A row per MMA-Sim record with a measurement to check its per-layer
    prediction against."""
    out = []
    for m in sorted((r for r in records if r['backend'] == 'mmasim'), key=lambda r: r['design']):
        lay = (m['layers']['prefill'], m['layers']['decode'])
        if 'tokens' in m:
            out.append(Check(m['design'], 'full run', lay, (m['prefill'], m['decode'])))
        if 'in_model' in m:
            out.append(Check(m['design'], 'in-model sample', lay,
                             (m['in_model']['prefill'], m['in_model']['decode'])))
    return out



def _num(v: float) -> str:
    """*v* to about three significant digits, grouped by thousands."""
    return f'{v:,.0f}' if v >= 100 else f'{v:.3g}'


def _agrees(r: Row) -> str:
    if r.same is not None:
        return f'{r.same}/{r.problems}'
    if r.check is not None:
        return 'bit-exact' if r.check['differing'] == 0 else f"{r.check['differing']:,} differ"
    return ''


def _cells(r: Row) -> list[str]:
    return [str(r.problems), '' if r.tokens is None else f'{r.tokens:,}', _num(r.prefill),
            _num(r.decode), '' if r.rate is None else _num(r.rate),
            *('' if r.correct is None else f'{100 * r.correct[m] / r.problems:.1f}' for m in gsm8k.METRICS),
            _agrees(r), '' if r.speedup is None else _num(r.speedup)]


def _bits(c: dict[str, int]) -> list[str]:
    return [f"{c['differing']:,}/{c['outputs']:,}", f"{100 * c['finite'] / c['outputs']:.1f}"]


DESIGN_HEADER = ['Scheme', 'FPy prefill (s)', 'FPy decode (s)', 'FPy (s)', 'MMA-Sim prefill (s)',
                 'MMA-Sim decode (s)', 'MMA-Sim (s)', 'Speedup', 'Differing outputs', 'Finite (\\%)']


def _design_cells(d: Design) -> list[str]:
    star = '' if d.measured else '*'
    return [d.scheme, _num(d.fpy[0]), _num(d.fpy[1]), _num(sum(d.fpy)), _num(d.mmasim[0]) + star,
            _num(d.mmasim[1]) + star, _num(sum(d.mmasim)) + star, _num(d.speedup), *_bits(d.check)]


CHECK_HEADER = ['Against', 'Predicted (s)', 'Measured (s)', 'Prefill error (\\%)',
                'Decode error (\\%)', 'Total error (\\%)']


def _check_cells(c: Check) -> list[str]:
    return [c.against, _num(sum(c.predicted)), _num(sum(c.reference)),
            *(f'{100 * e:+.1f}' for e in c.errors)]


def _tabular(header: list[str], rows: list[list[str]], note: str = '') -> str:
    lines = [r'\begin{tabular}{l' + 'r' * len(header) + '}', r'\toprule',
             ' & '.join(['Design', *header]) + r' \\', r'\midrule',
             *(' & '.join([rf'\texttt{{{r[0]}}}', *r[1:]]) + r' \\' for r in rows),
             r'\bottomrule', r'\end{tabular}']
    return '\n'.join([*lines, *([f'% {note}'] if note else []), ''])


def _text(header: list[str], rows: list[list[str]]) -> str:
    table = [['Design', *(h.replace('\\', '') for h in header)], *rows]
    widths = [max(len(row[i]) for row in table) for i in range(len(table[0]))]
    return '\n'.join('  '.join(c.ljust(w) if i == 0 else c.rjust(w)
                               for i, (c, w) in enumerate(zip(row, widths, strict=True))).rstrip()
                     for row in table)


def _label(r: Row) -> str:
    return LABELS[r.backend] + (' (predicted)' if r.predicted else '')


HEADER = ['Problems', 'Tokens', 'Prefill (s)', 'Decode (s)', 'Tokens/s', 'Strict (\\%)',
          'Flexible (\\%)', 'Same as MMA-Sim', 'Speedup']


def latex(rs: list[Row]) -> str:
    lines = [r'\begin{tabular}{ll' + 'r' * len(HEADER) + '}', r'\toprule',
             ' & '.join(['Design', 'Backend', *HEADER]) + r' \\']
    last: object = ()
    for r in rs:
        if r.design != last:
            lines.append(r'\midrule')
            last = r.design
            design = '---' if r.design is None else rf'\texttt{{{r.design}}}'
        else:
            design = ''
        lines.append(' & '.join([design, _label(r), *_cells(r)]) + r' \\')
    return '\n'.join([*lines, r'\bottomrule', r'\end{tabular}', ''])


def text(rs: list[Row]) -> str:
    table = [['Design', 'Backend', *(h.replace('\\', '') for h in HEADER)],
             *([r.design or '-', r.backend + ('*' if r.predicted else ''), *_cells(r)] for r in rs)]
    widths = [max(len(row[i]) for row in table) for i in range(len(table[0]))]
    note = ['* time predicted from a sample (sample.py)'] if any(r.predicted for r in rs) else []
    return '\n'.join(['  '.join(c.ljust(w) if i < 2 else c.rjust(w)
                                for i, (c, w) in enumerate(zip(row, widths, strict=True))).rstrip()
                      for row in table] + note)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument('results', type=Path, help="run.py's output directory")
    out = ap.parse_args(argv).results
    records = load(out)
    rs = rows(records)
    if not rs:
        ap.error(f'no records in {out}')
    main_rows, ds, cs = measured(rs), designs(records), checks(records)
    stats = summary(ds)
    star = '* MMA-Sim predicted per layer (sample.py)'
    design_rows = [[d.design, *_design_cells(d)] for d in ds]
    check_rows = [[c.design, *_check_cells(c)] for c in cs]
    (out / f'{PREFIX}speed.tex').write_text(latex(main_rows))
    (out / f'{PREFIX}designs.tex').write_text(_tabular(DESIGN_HEADER, design_rows, star))
    (out / f'{PREFIX}check.tex').write_text(_tabular(CHECK_HEADER, check_rows))
    nums = {
        'main': [dataclasses.asdict(r) | {'seconds': r.seconds, 'rate': r.rate} for r in main_rows],
        'designs': [dataclasses.asdict(d) | {'speedup': d.speedup} for d in ds],
        'summary': stats,
        'check': [dataclasses.asdict(c) | {'errors': c.errors} for c in cs],
    }
    (out / f'{PREFIX}speed.json').write_text(json.dumps(nums, indent=1) + '\n')
    parts = [text(main_rows)]
    if ds:
        parts += [_text(DESIGN_HEADER, design_rows), star]
    if stats:
        parts.append(f"Speedup over {stats['designs']} designs: min {_num(stats['min'][0])} "
                     f"({stats['min'][1]}), median {_num(stats['median'])}, "
                     f"max {_num(stats['max'][0])} ({stats['max'][1]})")
    if cs:
        parts.append(_text(CHECK_HEADER, check_rows))
    report = '\n\n'.join(parts)
    (out / f'{PREFIX}summary.txt').write_text(report + '\n')
    print(report)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
