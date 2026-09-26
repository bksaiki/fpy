"""
Chat with the model in the terminal, its linear layers through a run.

Greedy decoding, thinking off, the conversation re-encoded each turn by the
model's chat template; the reply streams as it is decoded.  Commands:

    /run <mode>   switch run (fp32, bf16-exact, or a design); the history stays
    /reset        forget the conversation
    /quit         leave (or end of input)

    python serve/chat.py                               # fp32
    python serve/chat.py -r amd.cdna2.bf16 --model Qwen/Qwen3.5-0.8B
"""

import argparse
import sys
import time

import decode
import swap


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    swap.add_args(ap)
    ap.add_argument('-r', '--run', choices=swap.MODES, default='fp32')
    ap.add_argument('--max-new', type=int, default=1024, help='tokens per reply at most')
    args = ap.parse_args(argv)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    model, run = swap.load(args.model, args.split_k, args.combine)
    run.mode = args.run
    eos = decode.stop_tokens(model, tok)
    messages: list[dict[str, str]] = []
    print(f'{args.model}, run {run.mode}.  /run <mode>, /reset, /quit.', file=sys.stderr)

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
            if arg.strip() in swap.MODES:
                run.mode = arg.strip()
            else:
                print(f'runs: {", ".join(swap.MODES)}', file=sys.stderr)
            continue

        messages.append({'role': 'user', 'content': line})
        prompt = decode.encode(tok, messages)
        out: list[int] = []
        shown = ''
        start = time.perf_counter()
        try:
            for t in decode.stream(model, prompt, args.max_new, eos):
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
