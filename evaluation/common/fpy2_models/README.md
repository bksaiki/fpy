# fpy2_models

FPy's compiled MMA-Sim designs as Torch operators. Each design becomes
`torch.ops.fpy2_models.<op>(a, b, c[, alphas, betas])`, with the interface of
MMA-Sim's `dpa`: operands of shape `(..., K)`, one accumulator per row in `c`,
and, for block-scaled designs, scales of shape `(..., S)`.

## Build

From the repository root:

```sh
(cd evaluation/common/fpy2_models && ../../../.venv/bin/python setup.py build_ext --inplace)
```

This builds `fpy2_models/_C.abi3.so` in place; scripts import the package by
adding this directory to `sys.path`. Rebuild after changing a design or the
C++ backend.

The build runs `generate_models.py`, which:

1. runs `examples/mmasim/compile.py -o build/generated_models/models`, which
   compiles every design it can to C++ and lists their signatures, and the
   designs it refuses, in `models/manifest.json`;
2. writes `build/generated_models/<op>.cpp` for each design, registering its
   model as an operator through the runners in `fpy2_models/csrc/runners.h`;
3. writes `build/generated_models/manifest.json`, the models and refused
   designs, for the benchmark.

The extension then compiles every `<op>.cpp` with `-O3`.
