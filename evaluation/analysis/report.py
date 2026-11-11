"""
The precision evaluation's tables, figure and summary, from `run.py`'s results;
`run.py` calls it last.

    python evaluation/analysis/report.py evaluation/results/analysis

Writes into the results directory:

- `precision-table.tex`: the paper's Table 1, a column per architecture: the
  designs, sites and sites bounded only by R* from format inference, then
  each parameter's mean tightness;
- `precision-tops.tex`: sites bounded only by R*, per architecture, without
  and with digit bounds;
- `precision-tightness.tex`: the fraction of each inferred format's
  parameters observed values reach, per architecture;
- both again grouped by vendor (`-vendor`);
- `precision-slack.{pdf,png}`: each bounded site's slack with digit bounds;
- `precision-paper.json`: every table and number the evaluation cites,
  including the random inputs alone and the `anf/` site counts;
- `precision-summary.txt`: the same, as text.
"""

import argparse
import json
import math
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from observe import SLACK, Analysis

sys.path.append(str(Path(__file__).resolve().parents[1]))

from common.plot import ACCENT, LARGE, ROW, save, tidy

ARCHS = {
    'volta': 'Volta', 'turing': 'Turing', 'ampere': 'Ampere', 'ada': 'Ada',
    'hopper': 'Hopper', 'blackwell': 'Blackwell', 'rtx_blackwell': 'Blackwell',
    'cdna1': 'CDNA1', 'cdna2': 'CDNA2', 'cdna3': 'CDNA3', 'fma': 'FMA chain',
}
"""Each architecture's tag in a design's name, and its row label, in row order;
RTX Blackwell shares Blackwell's row."""

VENDORS = {tag: 'AMD' if tag.startswith('cdna') else 'FMA chain' if tag == 'fma' else 'NVIDIA'
           for tag in ARCHS}
"""Each architecture's tag, and its vendor's row label."""

DIMENSIONS = {'digits': 'Significant digits', 'lsb': 'Lowest digit', 'mag': 'Magnitude'}
"""Each slack reported, and its panel title.  The highest digit, `msb`, is
left out for the magnitude: `log2 b - log2 max|v|`, which sees a bound `b`
that is not a power of two."""

BINS = 8
"""Slacks of this many bits or more share the last bin."""



@dataclass
class Row:
    label: str
    designs: int = 0
    sites: int = 0
    top_format: int = 0
    top_digit_bound: int = 0


def arch(design: str) -> str:
    """The architecture tag of *design*: `nv.hopper.bf16.f32` -> `hopper`."""
    return 'fma' if '(fma)' in design else design.split('.')[1]


def _result(site: dict[str, Any], a: Analysis) -> dict[str, Any] | None:
    return site.get(a.value)


def _allocation(site: dict[str, Any]) -> bool:
    """Whether *site* allocates a list (`zs = empty(n)`): bounded, by the empty
    set, so with no abstract format, under both analyses.  The formalism has
    no allocations, so these are not sites."""
    return all(r is not None and not r['top'] and r['static'] is None
               for r in (_result(site, a) for a in Analysis))


def load(results: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """The records in *results*, list allocations left out, and the designs
    that failed."""
    records = [r for p in sorted(results.glob('*.json'))
               if 'design' in (r := json.loads(p.read_text()))]   # not this report's own JSON
    for r in records:
        if 'sites' in r:
            r['sites'] = [s for s in r['sites'] if not _allocation(s)]
    return [r for r in records if 'error' not in r], [r['design'] for r in records if 'error' in r]


def table(records: Iterable[dict[str, Any]], groups: dict[str, str] = ARCHS) -> list[Row]:
    """One row per group of architectures present (*groups* maps each
    architecture's tag to its row label), in order, then the total."""
    rows = {label: Row(label) for label in dict.fromkeys(groups.values())}
    total = Row('Total')
    for rec in records:
        for row in (rows[groups[arch(rec['design'])]], total):
            row.designs += 1
            for site in rec['sites']:
                f, d = _result(site, Analysis.FORMAT), _result(site, Analysis.DIGIT_BOUND)
                row.sites += 1
                row.top_format += bool(f and f['top'])
                row.top_digit_bound += bool(d and d['top'])
    return [r for r in rows.values() if r.designs] + [total]


def latex(rows: list[Row]) -> str:
    lines = [
        r'\begin{tabular}{lrrrr}', r'\toprule',
        r'Architecture & Designs & Sites & $\mathbb{R}^*$ (FI) & $\mathbb{R}^*$ (FI+DB) \\',
        r'\midrule',
    ]
    for r in rows:
        if r.label == 'Total':
            lines.append(r'\midrule')
        lines.append(f'{r.label} & {r.designs} & {r.sites} & {r.top_format} & '
                     f'{r.top_digit_bound} \\\\')
    return '\n'.join([*lines, r'\bottomrule', r'\end{tabular}', ''])


def _measured(
    rec: dict[str, Any], observed: str = 'observed',
) -> Iterable[tuple[dict[str, Any], dict[str, Any]]]:
    """The static and observed parameters of each of *rec*'s sites that digit
    bounds leave bounded and that ran soundly with a finite nonzero value;
    *observed* is `observed_random` for the random inputs alone."""
    for site in rec['sites']:
        r = _result(site, Analysis.DIGIT_BOUND)
        if r and not r['top'] and r['sound'] and r['static'] and r.get(observed):
            yield r['static'], r[observed]


def slacks(records: Iterable[dict[str, Any]], observed: str = 'observed') -> dict[str, list[int | float]]:
    """Each dimension's slack at every measured site."""
    out: dict[str, list[int | float]] = {d: [] for d in DIMENSIONS}
    for rec in records:
        for st, seen in _measured(rec, observed):
            for d in DIMENSIONS:
                out[d].append(SLACK[d](st[d], seen[d]))
    return out


TIGHTNESS = {
    'digits': ('Precision', lambda st, seen: seen['digits'] / st['digits']),
    'lsb': ('Min.\\ quantum', lambda st, seen: 2.0 ** (st['lsb'] - seen['lsb'])),
    'mag': ('Bound', lambda st, seen: 2.0 ** (seen['mag'] - st['mag'])),
}
"""Each parameter's tightness -- the fraction of an inferred format A(p, exp, b)
that observed values reach, in (0, 1], or 0 where it is unbounded -- by its
column title: p' / p, 2^exp / 2^exp', and b' / b."""


def tightness(
    records: Iterable[dict[str, Any]], groups: dict[str, str] = ARCHS, observed: str = 'observed',
) -> list[tuple[str, int, dict[str, float]]]:
    """Each group's mean tightness over its measured sites, grouped as in
    `table`, then the total: `(label, sites, {parameter: mean})`."""
    seen: dict[str, list[dict[str, float]]] = {label: [] for label in dict.fromkeys(groups.values())}
    for rec in records:
        seen[groups[arch(rec['design'])]] += [{c: f(st, ob) for c, (_, f) in TIGHTNESS.items()}
                                             for st, ob in _measured(rec, observed)]
    rows = [(label, cs) for label, cs in seen.items() if cs]
    rows.append(('Total', [c for cs in seen.values() for c in cs]))
    return [(label, len(cs), {c: sum(x[c] for x in cs) / max(len(cs), 1) for c in TIGHTNESS})
            for label, cs in rows]


def tightness_latex(rows: list[tuple[str, int, dict[str, float]]]) -> str:
    titles = ' & '.join(title for title, _ in TIGHTNESS.values())
    lines = [r'\begin{tabular}{lr' + 'r' * len(TIGHTNESS) + '}', r'\toprule',
             f'Architecture & Sites & {titles} \\\\', r'\midrule']
    for label, n, means in rows:
        if label == 'Total':
            lines.append(r'\midrule')
        cells = ' & '.join(f'{100 * means[c]:.0f}\\%' for c in TIGHTNESS)
        lines.append(f'{label} & {n} & {cells} \\\\')
    return '\n'.join([*lines, r'\bottomrule', r'\end{tabular}', ''])


PAPER = {'digits': "$p' / p$", 'lsb': "$2^{exp} / 2^{exp'}$", 'mag': "$b' / b$"}
"""Each parameter's row in the paper's table."""


def paper_latex(rows: list[Row], tight: list[tuple[str, int, dict[str, float]]]) -> str:
    """The paper's Table 1 from `table`'s *rows* and `tightness`' *tight*:
    a column per architecture, then the total, set apart (`\\R` is the
    paper's macro for the reals)."""
    means = {label: m for label, _, m in tight}

    def pct(label: str, c: str) -> str:
        return f'{100 * means[label][c]:.0f}\\%' if label in means else '--'

    body = [('Designs', [str(r.designs) for r in rows]), ('Assignments', [str(r.sites) for r in rows]),
            (r'Only $\R^*$ (FI)', [str(r.top_format) for r in rows]),
            *((head, [pct(r.label, c) for r in rows]) for c, head in PAPER.items())]
    heads = ['FMA' if r.label == 'FMA chain' else r.label for r in rows]
    return '\n'.join([
        r'\begin{tabular}{@{}l*{' + str(len(rows) - 1) + r'}{r}@{\hspace{1em}}r@{}}', r'\toprule',
        ' & ' + ' & '.join(heads) + r' \\', r'\midrule',
        *(f'{head} & {" & ".join(cells)} \\\\' for head, cells in body),
        r'\bottomrule', r'\end{tabular}', ''])


def _bin(v: float) -> int:
    """Bin 0 holds exact bounds, bin k slacks in (k - 1, k] bits."""
    return BINS if math.isinf(v) else min(math.ceil(v), BINS)


def figure(data: dict[str, list[int | float]], out: Path) -> None:
    fig, axes = plt.subplots(1, len(DIMENSIONS), figsize=ROW, sharey=True, layout='constrained')
    labels = [str(b) for b in range(BINS)] + [f'≥{BINS}']
    for ax, (d, title) in zip(axes, DIMENSIONS.items(), strict=True):
        counts = [0] * (BINS + 1)
        for v in data[d]:
            counts[_bin(v)] += 1
        ax.bar(range(BINS + 1), counts, width=0.8, color=ACCENT, zorder=2)
        ax.set_title(title, loc='left', **LARGE)
        ax.set_xticks(range(BINS + 1), labels)
        ax.set_xlabel('slack (bits)', **LARGE)
        tidy(ax)
    axes[0].set_ylabel('sites', **LARGE)
    save(fig, out)


def _excluded(records: list[dict[str, Any]]) -> dict[str, int]:
    """Why each site digit bounds do not measure is left out."""
    out = {'top': 0, 'not_run': 0, 'unsound': 0, 'zero_or_special_only': 0}
    for rec in records:
        for site in rec['sites']:
            r = _result(site, Analysis.DIGIT_BOUND)
            if r['top']:
                out['top'] += 1
            elif not r['executed']:
                out['not_run'] += 1
            elif not r['sound']:
                out['unsound'] += 1
            elif r['observed'] is None:
                out['zero_or_special_only'] += 1
    return out


_NAMES = {'digits': 'precision', 'lsb': 'min_quantum', 'mag': 'bound'}
"""Each tightness parameter's name in `numbers`."""


def _tops(rows: list[Row]) -> list[dict[str, Any]]:
    return [{'label': r.label, 'designs': r.designs, 'sites': r.sites,
             'top_format': r.top_format, 'top_digit_bound': r.top_digit_bound} for r in rows]


def _tightness(rows: list[tuple[str, int, dict[str, float]]]) -> list[dict[str, Any]]:
    return [{'label': label, 'sites': n, **{_NAMES[c]: means[c] for c in TIGHTNESS}}
            for label, n, means in rows]


def numbers(records: list[dict[str, Any]], anf: list[dict[str, Any]], failed: list[str]) -> dict[str, Any]:
    """Every table and number the evaluation cites.  Tightness is a fraction
    in [0, 1]; the tables print it as a rounded percentage."""
    data = slacks(records)
    n = len(data['digits'])
    first = records[0]
    out: dict[str, Any] = {
        'run': {
            'designs': len(records), 'failed': failed,
            'inputs_per_design': first['n'], 'random_inputs_per_design': first.get('random'),
            'input_source': first['inputs'], 'seed': first.get('seed'),
            'executions': sum(r['n'] for r in records),
            'unsound': sum(not r['sound'] for rec in records for site in rec['sites']
                           for a in Analysis if (r := _result(site, a))),
        },
        'sites': {'total': table(records)[-1].sites, 'measured': n, 'excluded': _excluded(records)},
        'tops': {'by_architecture': _tops(table(records)),
                 'by_vendor': _tops(table(records, VENDORS))},
        'tightness': {'by_architecture': _tightness(tightness(records)),
                      'by_vendor': _tightness(tightness(records, VENDORS)),
                      'random_inputs_alone': _tightness(tightness(records, observed='observed_random'))[-1]},
        'exactly_attained': {
            **{_NAMES[d]: sum(v == 0 for v in data[d]) / n for d in DIMENSIONS},
            'all_three': sum(all(data[d][i] == 0 for d in DIMENSIONS) for i in range(n)) / n,
        },
        'lowest_digit_slack_at_least': {
            'bits': BINS,
            'all_inputs': sum(v >= BINS for v in data['lsb']),
            'random_inputs_alone': sum(v >= BINS for v in slacks(records, 'observed_random')['lsb']),
        },
    }
    if anf:
        out['anf'] = _tops(table(anf))[-1]
    return out


def summary(nums: dict[str, Any]) -> str:
    """*nums* as text."""
    run, sites = nums['run'], nums['sites']
    total, alone = nums['tops']['by_architecture'][-1], nums['tightness']['random_inputs_alone']

    def tight(row: dict[str, Any]) -> str:
        return ', '.join(f'{k} {100 * row[k]:.0f}%' for k in _NAMES.values())

    deep = nums['lowest_digit_slack_at_least']
    lines = [
        (f'{run["designs"]} designs, {run["inputs_per_design"]} inputs each ({run["input_source"]}, '
         f'{run["random_inputs_per_design"]} random), {run["executions"]} executions, '
         f'{run["unsound"]} unsound results'),
        f'sites: {sites["total"]}; R* without digit bounds {total["top_format"]}, with {total["top_digit_bound"]}',
        f'measured: {sites["measured"]}; left out: {sites["excluded"]}',
        f'tightness, all inputs: {tight(nums["tightness"]["by_architecture"][-1])}',
        f'tightness, random inputs alone: {tight(alone)}',
        (f'exactly attained: {tight(nums["exactly_attained"])}, all three '
         f'{100 * nums["exactly_attained"]["all_three"]:.0f}%'),
        (f'sites with >= {deep["bits"]} bits of lowest-digit slack: {deep["all_inputs"]} '
         f'(random inputs alone {deep["random_inputs_alone"]})'),
    ]
    if 'anf' in nums:
        a = nums['anf']
        lines.append(f'anf: {a["sites"]} sites; R* without digit bounds {a["top_format"]}, '
                     f'with {a["top_digit_bound"]}')
    if run['failed']:
        lines.append(f'failed, left out: {", ".join(run["failed"])}')
    return '\n'.join(lines)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument('results', type=Path, help="run.py's output directory")
    args = ap.parse_args(argv)
    out = args.results

    records, failed = load(out)
    if not records:
        ap.error(f'no results in {out}')
    anf, _ = load(out / 'anf')
    (out / 'precision-table.tex').write_text(paper_latex(table(records), tightness(records)))
    for suffix, groups in (('', ARCHS), ('-vendor', VENDORS)):
        (out / f'precision-tops{suffix}.tex').write_text(latex(table(records, groups)))
        (out / f'precision-tightness{suffix}.tex').write_text(
            tightness_latex(tightness(records, groups)))
    figure(slacks(records), out / 'precision-slack')
    nums = numbers(records, anf, failed)
    (out / 'precision-paper.json').write_text(json.dumps(nums, indent=1) + '\n')
    text = summary(nums)
    (out / 'precision-summary.txt').write_text(text + '\n')
    print(text)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
