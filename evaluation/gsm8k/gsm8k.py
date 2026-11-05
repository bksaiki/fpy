"""
GSM8K, 8-shot chain of thought, as lm-evaluation-harness's `gsm8k_cot` with
`--apply_chat_template --fewshot_as_multiturn`.
"""

import re
from dataclasses import dataclass

EXEMPLARS = [
    (('There are 15 trees in the grove. Grove workers will plant trees in the grove today. After '
      'they are done, there will be 21 trees. How many trees did the grove workers plant today?'),
     ('There are 15 trees originally. Then there were 21 trees after some more were planted. So '
      'there must have been 21 - 15 = 6. The answer is 6.')),
    (('If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the '
      'parking lot?'),
      'There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.'),
    (('Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have '
      'left in total?'),
     ('Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. '
      'After eating 35, they had 74 - 35 = 39. The answer is 39.')),
    (('Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many '
      'lollipops did Jason give to Denny?'),
     ('Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave '
      'Denny 20 - 12 = 8. The answer is 8.')),
    (('Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many '
      'toys does he have now?'),
     ('Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more '
      'toys. 5 + 4 = 9. The answer is 9.')),
    (('There were nine computers in the server room. Five more computers were installed each day, '
      'from monday to thursday. How many computers are now in the server room?'),
     ('There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * '
      '4 = 20 computers were added. 9 + 20 is 29. The answer is 29.')),
    (('Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. '
      'How many golf balls did he have at the end of wednesday?'),
     ('Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After '
      'losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33.')),
    ('Olivia has $23. She bought five bagels for $3 each. How much money does she have left?',
     ('Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has '
      '23 - 15 dollars left. 23 - 15 is 8. The answer is 8.')),
]
"""Wei et al.'s exemplars, `(question, answer)`."""

STOP = 'Q:'
"""`gsm8k_cot`'s stop sequence besides the end of turn: a reply is scored up to it."""

_STRICT = re.compile(r'The answer is (\-?[0-9\.\,]+).')
_FLEXIBLE = re.compile(r'(-?[$0-9.,]{2,})|(-?[0-9]+)')


@dataclass(frozen=True)
class Problem:
    question: str
    target: str
    """The final answer, normalized (:func:`normalize`)."""


def problems(n: int | None = None) -> list[Problem]:
    """The test split's first *n* problems (all, by default)."""
    from datasets import load_dataset

    split = load_dataset('openai/gsm8k', 'main', split='test')
    rows = split if n is None else split.select(range(n))
    return [Problem(r['question'], normalize(r['answer'].split('####')[-1])) for r in rows]


def _turn(question: str) -> dict[str, str]:
    return {'role': 'user', 'content': f'Q: {question}\nA:'}


def messages(question: str) -> list[dict[str, str]]:
    """The exemplars as alternating turns, then *question*."""
    shots = [m for q, a in EXEMPLARS for m in (_turn(q), {'role': 'assistant', 'content': a})]
    return [*shots, _turn(question)]


def normalize(s: str) -> str:
    """*s* without commas, dollar signs, surrounding space or a final period."""
    return re.sub(r'\.$', '', s.replace(',', '').replace('$', '').strip())


def strict(reply: str) -> str | None:
    """*reply*'s strict-match answer: the first number after "The answer
    is", normalized."""
    m = _STRICT.search(reply.split(STOP)[0])
    return normalize(m.group(1)) if m else None


def flexible(reply: str) -> str | None:
    """*reply*'s flexible-extract answer: its last number, normalized."""
    found = _FLEXIBLE.findall(reply.split(STOP)[0])
    return normalize(next(g for g in found[-1] if g)) if found else None


METRICS = {'strict-match': strict, 'flexible-extract': flexible}
"""`gsm8k_cot`'s metrics, each an exact match of its answer with the target."""


def scores(replies: list[str], targets: list[str]) -> dict[str, int]:
    """How many of *replies* each metric counts correct."""
    return {m: sum(f(r) == t for r, t in zip(replies, targets, strict=True))
            for m, f in METRICS.items()}
