import random

from argparse import ArgumentParser
from pathlib import Path
from typing import Optional

from .config import Config, EvalMode, ReferenceMode
from .evaluator import run_eval_real
from .func_profiler import run_func_profiler
from .expr_profiler import run_expr_profiler

DEFAULT_NUM_SAMPLES = 10

def _resolve_path(s: str):
    return Path(s).resolve()

def _run_eval(config: Config):
    match config.mode:
        case EvalMode.REAL:
            run_eval_real(config)
        case EvalMode.FUN_PROFILE:
            run_func_profiler(config)
        case EvalMode.EXPR_PROFILE:
            run_expr_profiler(config)
        case _:
            raise ValueError(f"invalid mode: {config}")

def _option_string(strs: list[str]):
    return ', '.join(map(lambda s: f'\'{s}\'', strs))

_eval_options = EvalMode.options()
_ref_options = ReferenceMode.options()
_mode_help_str = f'evaluation mode (one of {_option_string(_eval_options)})'
_ref_help_str = f'reference interpreter to use (default: \'real\', one of {_option_string(_ref_options)})'

parser = ArgumentParser(
    prog='python3 -m infra.real',
    description='evaluation of fpy2 real evaluation'
)

parser.add_argument('mode', help=_mode_help_str, type=str)
parser.add_argument('-i', '--input', help='input paths', type=str, action='append', default=[])
parser.add_argument('-n', '--num-samples', help='number of samples', type=int, default=DEFAULT_NUM_SAMPLES)
parser.add_argument('--ref', help=_ref_help_str, type=str, default='real')
parser.add_argument('--seed', help='random seed', type=int, default=None)
args = parser.parse_args()

input_strs: list[str] = args.input
mode_str: str = args.mode
num_samples: int = args.num_samples
ref_str: str = args.ref
seed: Optional[int] = args.seed

input_paths = [_resolve_path(s) for s in input_strs]
mode = EvalMode.from_str(mode_str)
ref_mode = ReferenceMode.from_str(ref_str)
config = Config(mode, input_paths, num_samples, ref_mode, seed)

_run_eval(config)
