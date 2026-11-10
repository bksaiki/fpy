# MMA designs

Shared by the evaluations: the MMA designs, built from the FPy models in
`examples/mmasim/models`, and the harness that runs them.

```
compile.py        every design (DESIGNS) to C++; reports where each one stops
compile_triton.py every design to a Triton kernel; -r runs and compares
generator/        designs from parameters; catalog.py, DESIGNS as parameters
core/             runs a model's linear layers through a design's kernels
bench/            runtime of the Triton kernels, and a hand-written reference
no_sync.py        MMA-Sim's arithmetic without host waits, and torch.compile fixes
tests/            the generator and Triton kernels, against the interpreter
core/tests/       the harness (GPU; run serially)
```

```sh
python evaluation/common/mma/compile.py               # -v for the failures, -o DIR to emit
python evaluation/common/mma/compile_triton.py -r 64  # compared on 64 draws (GPU)
pytest -n 8 evaluation/common/mma/tests
pytest evaluation/common/mma/core/tests
```
