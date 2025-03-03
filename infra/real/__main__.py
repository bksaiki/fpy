from argparse import ArgumentParser
from pathlib import Path

from .config import EvalMode, Config
from .evaluator import run_eval_real
from .func_profiler import run_func_profiler

DEFAULT_NUM_SAMPLES = 10

def _resolve_path(s: str):
    return Path(s).resolve()

def _run_eval(config: Config):
    match config.mode:
        case EvalMode.REAL:
            run_eval_real(config)
        case EvalMode.FUN_PROFILE:
            run_func_profiler(config)
        case _:
            raise ValueError(f"invalid mode: {config.mode}")

parser = ArgumentParser(
    prog='python3 -m infra.real',
    description='evaluation of fpy2 real evaluation'
)

parser.add_argument('mode', help='evaluation mode', type=str)
parser.add_argument('-i', '--input', help='input paths', type=str, action='append', default=[])
parser.add_argument('-n', '--num-samples', help='number of samples', type=int, default=DEFAULT_NUM_SAMPLES)
args = parser.parse_args()

input_strs: list[str] = args.input
mode_str: str = args.mode
num_samples: int = args.num_samples

input_paths = [_resolve_path(s) for s in input_strs]
mode = EvalMode.from_str(mode_str)
config = Config(mode, input_paths, num_samples)

_run_eval(config)
