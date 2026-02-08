import fpy2 as fp

from dataclasses import dataclass


@dataclass(frozen=True)
class Benchmark:
    """Benchmark configuration."""
    func: fp.Function
    ctxs: tuple[fp.Context | None, ...]
    num_inputs: int
    vector_size: int
