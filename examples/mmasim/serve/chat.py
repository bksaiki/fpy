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
    ap.add_argument('--model', default=swap.MODEL)
    ap.add_argument('-r', '--run', choices=swap.MODES, default='fp32')
    ap.add_argument('--max-new', type=int, default=1024, help='tokens per reply at most')
    ap.add_argument('--split-k', type=int, default=1)
    ap.add_argument('--combine', choices=['linear', 'tree'], default='linear')
    args = ap.parse_args(argv)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    model, run = swap.load(args.model)
    run.mode, run.split_k, run.combine = args.run, args.split_k, args.combine
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
        prompt = tok.apply_chat_template(
            messages, add_generation_prompt=True, enable_thinking=False,
            return_dict=True, return_tensors='pt')['input_ids'].cuda()
        out: list[int] = []
        shown = ''
        start = time.perf_counter()
        try:
            for t in decode.stream(model, prompt, args.max_new, eos):
                out.append(t)
                # decode the whole reply: a token can be part of a character
                text = tok.decode(out, skip_special_tokens=True)
                print(text[len(shown):], end='', flush=True)
                shown = text
        except KeyboardInterrupt:
            pass
        seconds = time.perf_counter() - start
        print(f'\n({len(out)} tokens, {len(out) / seconds:.1f} tokens/s)', file=sys.stderr)
        messages.append({'role': 'assistant', 'content': shown})


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
