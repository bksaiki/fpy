"""
Tables of `run.py`'s records: per backend and design, time, throughput,
accuracy (`gsm8k_cot`'s two metrics, scored from the replies), agreement
with MMA-Sim and speedup over it.  A sampled MMA-Sim run (`sample.py`)
gives its predicted time, and agreement is its sampled calls' bit check.

    python evaluation/gsm8k/report.py evaluation/results/gsm8k

Writes, beside the records:

- `gsm8k-speed.tex`: the table, `torch` first, then each design's backends;
- `gsm8k-speed.json`: every number in it;
- `gsm8k-summary.txt`: the same, as text.
"""

import argparse
import dataclasses
import json
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
    """MMA-Sim's sampled calls' bit check (`calls`, `outputs`, `differing`),
    for the design's `fpy`, if MMA-Sim was sampled."""
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
        """Whether MMA-Sim's tokens or sampled outputs differ from this row's."""
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
            if ref.get('sampled'):
                check = ref['check']
            else:
                same = sum(a == b for a, b in zip(r['tokens'], ref['tokens'], strict=True))
            speedup = (ref['prefill'] + ref['decode']) / (r['prefill'] + r['decode'])
        ran = 'tokens' in r
        out.append(Row(design, backend, r['problems'], sum(map(len, r['tokens'])) if ran else None,
                       r['prefill'], r['decode'], gsm8k.scores(r['replies'], r['targets']) if ran else None,
                       same, check, speedup, predicted=bool(r.get('sampled'))))
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
    rs = rows(load(out))
    if not rs:
        ap.error(f'no records in {out}')
    (out / f'{PREFIX}speed.tex').write_text(latex(rs))
    nums = [dataclasses.asdict(r) | {'seconds': r.seconds, 'rate': r.rate} for r in rs]
    (out / f'{PREFIX}speed.json').write_text(json.dumps(nums, indent=1) + '\n')
    summary = text(rs)
    (out / f'{PREFIX}summary.txt').write_text(summary + '\n')
    print(summary)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
