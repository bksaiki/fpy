from fpy2 import RealInterpreter, DefaultInterpreter
from titanfp.arithmetic.ieee754 import ieee_ctx

from .config import ReferenceMode

_disabled = [
    # Too hard
    'Rocket_Trajectory',
    'Eigenvalue_Computation',
    'Pendulum',
    # 'Flower',
    'arclength_of_a_wiggly_function',
    'arclength_of_a_wiggly_function__u40_old_version_u41_',
    'Jacobi_u39_s_Method',
    'PID',
    # Infinite loops
    'Euler_Oscillator',
    'Filter',
    'Circle',
    # tensor
    'Symplectic_Oscillator',
    'Flower',
    'Arrow_Hurwicz'
]

def select_interpreter(mode: ReferenceMode):
    match mode:
        case ReferenceMode.REAL:
            return RealInterpreter()
        case ReferenceMode.FLOAT_1K:
            ctx = ieee_ctx(19, 1024)
            return DefaultInterpreter(ctx=ctx)
        case ReferenceMode.FLOAT_2K:
            ctx = ieee_ctx(19, 2048)
            return DefaultInterpreter(ctx=ctx)
        case ReferenceMode.FLOAT_4K:
            ctx = ieee_ctx(19, 4096)
            return DefaultInterpreter(ctx=ctx)
        case _:
            raise NotImplementedError(mode)

def disabled_tests():
    return list(_disabled)
