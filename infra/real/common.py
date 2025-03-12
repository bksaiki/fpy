from .config import Config, ReferenceMode

_disabled = [
    # Too hard
    'Rocket_Trajectory',
    'Eigenvalue_Computation',
    'Pendulum',
    'Flower',
    'arclength_of_a_wiggly_function',
    'arclength_of_a_wiggly_function__u40_old_version_u41_',
    'Jacobi_u39_s_Method',
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
    raise NotImplementedError(mode)

def disabled_tests():
    return list(_disabled)
