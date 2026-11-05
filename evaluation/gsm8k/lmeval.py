"""
GSM8K accuracy as lm-evaluation-harness reports it: its `gsm8k_cot` task
(8-shot chain of thought, chat template, few-shot as turns, thinking off)
on Qwen3.5-0.8B, with every linear layer computed by a backend.

    python evaluation/gsm8k/lmeval.py                      # torch and fpy, all 1319
    python evaluation/gsm8k/lmeval.py --against evaluation/results/gsm8k

Replaces the output directory (`results-lmeval/` beside this file, or
`-o DIR`) with `<backend>.json`: lm-eval's metrics, every sample, and the
time.  With `--against`, also `lmeval-compare.json`: per backend, how many of
`run.py`'s problems lm-eval answered with the same reply, both scores on
them, and how many of lm-eval's replies `gsm8k.METRICS` judge as lm-eval does.
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(_ROOT / 'evaluation' / 'common' / 'mma'), str(_ROOT / 'examples' / 'mmasim')]
sys.path.append(str(_ROOT / 'evaluation'))  # last: `config`, not shadowing this `run`

import config
import gsm8k
import mmasim_matmul
import report
import run
from core import swap

RESULTS = Path(__file__).resolve().parent / 'results-lmeval'
TASK = 'gsm8k_cot'


def evaluate(model: Any, tok: Any, batch: int, limit: int | None) -> dict[str, Any]:
    """lm-eval's results for :data:`TASK` on *model*, with its samples."""
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    lm = HFLM(pretrained=model, tokenizer=tok, batch_size=batch, enable_thinking=False)
    return lm_eval.simple_evaluate(model=lm, tasks=[TASK], apply_chat_template=True,
                                   fewshot_as_multiturn=True, log_samples=True, limit=limit)


def _samples(out: dict[str, Any]) -> dict[int, dict[str, Any]]:
    """Each problem's sample, by `doc_id`, every filter's answer and score
    merged."""
    merged: dict[int, dict[str, Any]] = {}
    for s in out['samples'][TASK]:
        m = merged.setdefault(s['doc_id'], {'doc_id': s['doc_id'], 'target': s['target'],
                                            'reply': s['resps'][0][0], 'answers': {}, 'correct': {}})
        m['answers'][s['filter']] = s['filtered_resps'][0]
        m['correct'][s['filter']] = bool(s['exact_match'])
    return merged


def compare(lmeval: dict[str, Any], ours: dict[str, Any]) -> dict[str, Any]:
    """How lm-eval's samples match `run.py`'s record *ours* on its problems."""
    samples = {s['doc_id']: s for s in lmeval['samples']}
    n = ours['problems']
    mine = [r.split(gsm8k.STOP)[0] for r in ours['replies']]
    theirs = [samples[i]['reply'] for i in range(n)]
    return {
        'problems': n,
        'same_reply': sum(a.strip() == b.strip() for a, b in zip(mine, theirs, strict=True)),
        'ours': gsm8k.scores(ours['replies'], ours['targets']),
        'lmeval': {m: sum(samples[i]['correct'][m] for i in range(n)) for m in gsm8k.METRICS},
        'same_verdict': {m: sum((f(samples[i]['reply']) == t) == samples[i]['correct'][m]
                                for i, t in enumerate(ours['targets']))
                         for m, f in gsm8k.METRICS.items()},
    }


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument('--backends', nargs='+', choices=('torch', 'fpy'), default=['torch', 'fpy'])
    ap.add_argument('--design', default=config.GSM8K_DESIGNS[0], help='for fpy (default %(default)s)')
    ap.add_argument('--limit', type=int, help='the first LIMIT problems only (default all)')
    ap.add_argument('--batch', type=int, default=config.GSM8K_BATCH, help='default %(default)s')
    ap.add_argument('--model', default=config.MODEL, help='default %(default)s')
    ap.add_argument('-o', '--out', metavar='DIR', type=Path, default=RESULTS,
                    help='the output directory, replaced (default results-lmeval/ here)')
    ap.add_argument('--against', metavar='DIR', type=Path, help="run.py's results to compare with")
    args = ap.parse_args(argv)
    if args.out.exists() and any(p.suffix != '.json' for p in args.out.iterdir()):
        ap.error(f'{args.out} holds files a run did not write')
    shutil.rmtree(args.out, ignore_errors=True)
    args.out.mkdir(parents=True)

    from transformers import AutoTokenizer

    model, r = mmasim_matmul.load(args.model)
    tok = AutoTokenizer.from_pretrained(args.model)
    env = run.environment()
    records = {}
    for backend in args.backends:
        design = args.design if backend == 'fpy' else None
        r.mode = run.mode(backend, design)
        print(f'-- {backend}', flush=True)
        start = time.perf_counter()
        out = evaluate(model, tok, args.batch, args.limit)
        records[backend] = rec = {
            'backend': backend, 'design': design, 'task': TASK, 'limit': args.limit,
            'batch': args.batch, 'seconds': time.perf_counter() - start,
            'results': out['results'][TASK], 'config': out['configs'][TASK],
            'samples': sorted(_samples(out).values(), key=lambda s: s['doc_id']), **env}
        (args.out / f'{backend}.json').write_text(json.dumps(rec, indent=1, default=str))
        print(f'  {json.dumps({k: v for k, v in rec["results"].items() if "exact" in k})}', flush=True)

    if args.against is not None:
        ours = {(x['backend'], x['design']): x for x in report.load(args.against)}
        cmp = {b: compare(rec, ours[b, rec['design']]) for b, rec in records.items()
               if (b, rec['design']) in ours}
        (args.out / 'lmeval-compare.json').write_text(json.dumps(cmp, indent=1))
        print(json.dumps(cmp, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
