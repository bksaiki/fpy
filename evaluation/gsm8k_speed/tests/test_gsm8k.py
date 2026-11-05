"""
GSM8K answer extraction, and batched generation against one prompt at a
time.  The latter needs a GPU and Qwen3.5-0.8B.

    pytest -n 0 -q evaluation/gsm8k_speed/tests/test_gsm8k.py
"""

import batched
import gsm8k
import pytest
import torch
from core import generate, swap

QUESTIONS = [
    'Tom has 3 apples and buys 4 more. How many apples does he have?',
    ('A train travels 60 miles per hour for 2.5 hours, then 40 miles per hour for 1 hour. '
     'How many miles does it travel in total?'),
]


@pytest.mark.parametrize('reply, strict, flexible', [
    ('So 3 + 4 = 7. The answer is 7.', '7', '7'),
    ('The answer is 1,250.', '1250', '1250'),
    ('The answer is -3.5.', '-3.5', '-3.5'),
    ('The answer is $18.', None, '18'),
    ('The answer is 4. Check: 2 + 2 = 4, not 5.', '4', '5'),
    ('First 2, then 10, so 12', None, '12'),
    ('The answer is 7.\n\nQ: what is 8?', '7', '7'),   # scored up to the stop sequence
    ('No numbers here.', None, None),
])
def test_metrics(reply: str, strict: str | None, flexible: str | None):
    assert (gsm8k.strict(reply), gsm8k.flexible(reply)) == (strict, flexible)


def test_scores():
    targets = [gsm8k.normalize(' 1,000'), '5']
    replies = ['The answer is 1000.', 'I get $5']
    assert gsm8k.scores(replies, targets) == {'strict-match': 1, 'flexible-extract': 2}


def test_shared():
    assert batched.shared([[1, 2, 3, 4], [1, 2, 5]]) == 2
    assert batched.shared([[1, 2, 3], [1, 2, 3, 4]]) == 2   # one token each of its own
    assert batched.shared([[1, 2, 3]]) == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs a GPU')
def test_batched_decodes_as_one_at_a_time():
    from transformers import AutoTokenizer

    model, _ = swap.load()   # fp32: the unpatched model
    tok = AutoTokenizer.from_pretrained(swap.MODEL)
    eos = generate.stop_tokens(model, tok)
    prompts = [generate.encode(tok, gsm8k.messages(q)) for q in QUESTIONS]
    alone = [generate.greedy(model, p, 24, eos) for p in prompts]
    got = batched.generate(model, [p[0].tolist() for p in prompts], 24, eos)
    assert batched.shared([p[0].tolist() for p in prompts]) > 500
    assert got.tokens == alone
