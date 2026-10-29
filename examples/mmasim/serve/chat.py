"""
Chat with the model in the terminal, its linear layers through a matmul.

Greedy decoding, thinking off, the conversation re-encoded each turn by the
model's chat template; the reply streams as it is decoded.  `--model` runs
under `--scheme`, its weights quantized by round-to-nearest.  Commands:

    /matmul <name>  switch matmul (fp32, the scheme's exact matmul, or a
                    design); the history stays
    /reset          forget the conversation
    /quit           leave (or end of input)

    python serve/chat.py                               # fp32
    python serve/chat.py --matmul amd.cdna2.bf16
    python serve/chat.py --scheme fp8-row --matmul nv.ada.e4m3.f32
    python serve/chat.py --scheme mxfp8 --matmul nv.blackwell.mx.e4m3
"""

import argparse
import sys
import time

from core import cli, generate, swap


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    cli.add_args(ap, seed=False)
    cli.add_scheme_args(ap)
    ap.add_argument('--matmul', dest='run', default='fp32',
                    help="fp32, or one of the scheme's matmuls (default: %(default)s)")
    ap.add_argument('--max-new', type=int, default=1024,
                    help='tokens per reply at most (default: %(default)s)')
    args = cli.parse(ap, argv)
    modes = swap.modes(args.scheme)
    if args.run not in modes:
        ap.error(f'{args.scheme.name} has no matmul {args.run} (only {", ".join(modes)})')

    import transformers
    from transformers import AutoTokenizer

    # its notes (e.g. a kernel falling back to PyTorch) would land mid-reply
    transformers.logging.set_verbosity_error()
    tok = AutoTokenizer.from_pretrained(args.model)
    model, run = cli.load(args)
    run.mode = args.run
    eos = generate.stop_tokens(model, tok)
    messages: list[dict[str, str]] = []
    print(f'{args.model} under {args.scheme.name}, '
          f'matmul {run.mode}.  /matmul <name>, /reset, /quit.', file=sys.stderr)

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
        if cmd == '/matmul':
            if arg.strip() in modes:
                run.mode = arg.strip()
            else:
                print('\n'.join(cli.matmuls(args.scheme)), file=sys.stderr)
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
