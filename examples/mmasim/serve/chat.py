"""
Chat with the model in the terminal, its linear layers through a run.

Greedy decoding, thinking off, the conversation re-encoded each turn by the
model's chat template; the reply streams as it is decoded.  `--model` is a
master (Qwen/Qwen3-0.6B, Qwen/Qwen3.5-0.8B) or a quantized checkpoint
(RedHatAI/Qwen3-0.6B-FP8-dynamic, RedHatAI/Qwen3-0.6B-FP8-BLOCK,
kaitchup/Qwen3-0.6B-NVFP4), run under `--scheme` as the other scripts do
(`checkpoints.for_scheme`).  Commands:

    /run <mode>   switch run (fp32, the scheme's exact run, or a design);
                  the history stays
    /reset        forget the conversation
    /quit         leave (or end of input)

    python serve/chat.py                               # fp32
    python serve/chat.py -r amd.cdna2.bf16 --model Qwen/Qwen3.5-0.8B
    python serve/chat.py --scheme fp8-row -r nv.ada.e4m3.f32
    python serve/chat.py --model kaitchup/Qwen3-0.6B-NVFP4 --scheme nvfp4 -r nv.blackwell.nvfp4
"""

import argparse
import sys
import time

from core import checkpoints, cli, generate, swap


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    cli.add_args(ap, seed=False)
    cli.add_scheme_args(ap)
    ap.add_argument('-r', '--run', default='fp32', help="fp32 (default), or one of the scheme's runs")
    ap.add_argument('--max-new', type=int, default=1024, help='tokens per reply at most')
    args = ap.parse_args(argv)
    modes = swap.modes(args.scheme)
    if args.run not in modes:
        ap.error(f'{args.scheme.name} has no run {args.run} (only {", ".join(modes)})')

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    model, run, about = checkpoints.for_scheme(
        args.model, args.scheme, requantize=args.requantize, master=args.master,
        split_k=args.split_k, combine=args.combine)
    run.mode = args.run
    eos = generate.stop_tokens(model, tok)
    messages: list[dict[str, str]] = []
    print(f'{args.model} under {args.scheme.name} (weights {about["source"]}), run {run.mode}.  '
          '/run <mode>, /reset, /quit.', file=sys.stderr)

    while True:
        try:
            line = input(f'[{run.mode}] > ').strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            continue
        cmd, _, arg = line.partition(' ')
        if cmd == '/quit':
            return 0
        if cmd == '/reset':
            messages.clear()
            continue
        if cmd == '/run':
            if arg.strip() in modes:
                run.mode = arg.strip()
            else:
                print(f'runs: {", ".join(modes)}', file=sys.stderr)
            continue

        messages.append({'role': 'user', 'content': line})
        prompt = generate.encode(tok, messages)
        out: list[int] = []
        shown = ''
        start = time.perf_counter()
        try:
            for t in generate.stream(model, prompt, args.max_new, eos):
                out.append(t)
                # decode the whole reply, holding back a character still incomplete
                text = tok.decode(out, skip_special_tokens=True)
                if not text.endswith('\ufffd'):
                    print(text[len(shown):], end='', flush=True)
                    shown = text
        except KeyboardInterrupt:
            pass
        seconds = time.perf_counter() - start
        text = tok.decode(out, skip_special_tokens=True)
        print(text[len(shown):], flush=True)
        print(f'({len(out)} tokens, {len(out) / seconds:.1f} tokens/s)', file=sys.stderr)
        messages.append({'role': 'assistant', 'content': text})


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
