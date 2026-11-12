import report


def test_families():
    assert report.family('fp32.fma') == 'FMA chain'
    assert report.family('amd.cdna3.fp8') == 'CDNA3'
    assert report.family('nv.turing.f16.f32') == 'Volta / Turing'
    assert report.family('nv.rtx_blackwell.e4m3.f16.mma') == 'Blackwell'
    assert report.family('nv.ada.e4m3.f32') == 'Ada'
    assert set(map(report.family, ['nv.hopper.bf16.f32', 'nv.ampere.tf32.f32'])) <= set(report.FAMILIES)


def test_speedups_and_table():
    rows = [{'design': d, 'implementation': i, 'batch_size': 8, 'dot_length': 64, 'dots_per_second': r}
            for d, i, r in [('fp32.fma', 'fpy', 8.0), ('fp32.fma', 'mmasim-tuned', 2.0),
                            ('amd.cdna3.fp8', 'fpy', 3.0), ('amd.cdna3.fp8', 'mmasim-tuned', 3.0)]]
    ups = report.speedups(rows)
    assert ups == {('fp32.fma', 8, 64): 4.0, ('amd.cdna3.fp8', 8, 64): 1.0}
    t = report.table({d: x for (d, _, _), x in ups.items()})
    assert [r['family'] for r in t] == ['CDNA3', 'FMA chain', 'All']
    assert t[-1]['geomean'] == 2.0
    assert '2.00$\\times$' in report.latex(t)


def test_macros():
    tex = report.macros({'fp32.fma': 4.0, 'amd.cdna3.fp8': 1.0}, 8, 64, 5)
    assert '\\newcommand{\\EvalCPUPerfSpeedupMin}{1.00$\\times$\\xspace}' in tex
    assert '\\newcommand{\\EvalCPUPerfSpeedupMax}{4.00$\\times$\\xspace}' in tex
    assert '\\newcommand{\\EvalCPUPerfSpeedupMean}{2.00$\\times$\\xspace}' in tex
    assert '\\newcommand{\\EvalCPUPerfThreads}{5\\xspace}' in tex


def test_tuning_speedup_and_macro():
    rows = [{'design': 'fp32.fma', 'implementation': i, 'batch_size': 8, 'dot_length': 64, 'dots_per_second': r}
            for i, r in [('fpy', 8.0), ('mmasim-tuned', 2.0), ('mmasim-released', 1.0)]]
    tuning = report.headline(rows, impl=report.BASELINE, base=report.RELEASED)
    assert tuning == {'fp32.fma': 2.0}
    assert '\\newcommand{\\EvalCPUPerfOptRatio}{2.00$\\times$\\xspace}' in report.macros({'fp32.fma': 4.0}, 8, 64, 1, tuning)
    assert 'OptRatio' not in report.macros({'fp32.fma': 4.0}, 8, 64, 1)
