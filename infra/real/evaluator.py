from .config import Config
from .load import load_funs
from .sample import sample

from fpy2 import *

def _run_one(fun: Function, rt: Interpreter, num_samples: int):
    # sample N points
    pts = sample(fun, num_samples)
    # evaluate over each point
    for pt in pts:
        rt.eval(fun, pt)


def run_eval_real(config: Config):
    rt = TitanicInterpreter()
    funs = load_funs(config.input_paths)

    print(f'testing over {len(funs)} functions')
    for fun in funs:
        _run_one(fun, rt, config.num_samples)

