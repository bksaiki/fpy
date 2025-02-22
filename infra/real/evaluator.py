
from .config import Config
from .load import load_funs

def run_eval_real(config: Config):
    funs = load_funs(config.input_paths)
    print(f'found {len(funs)} functions')
