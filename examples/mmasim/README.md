# MMA-Sim

Bit-accurate FPy models of NVIDIA Tensor Core and AMD Matrix Core MMA
arithmetic, following [MMA-Sim](https://github.com/microsoft/MMA-Sim)
(Xie et al., [arXiv:2511.10909](https://arxiv.org/abs/2511.10909)).

```
models/nv.py      NVIDIA: make_t_fdpa, make_t_fdpa_chain, make_st_fdpa, make_gst_fdpa
models/amd.py     AMD:    make_e_fdpa, make_ftz_addmul, make_tr_fdpa, make_gtr_fdpa
models/utils.py   shared helpers, and make_fma_dpa
compile.py        compile every design to C++ and report where each one stops
tests/            validation against the reference implementation
```

Each factory takes per-instruction parameters (formats as contexts, `F`,
`rho`, ...) and returns an FPy dot-product-accumulate.

## Running

Everything runs from this directory. The models are a package, so run them
with `-m`:

```sh
python -m models.nv          # Table 8 demo (or models.amd)
python compile.py            # every design; -v for the failures, -o DIR to emit
```

## Tests

The directed checks need only `fpy2`, and are what `pytest` runs:

```sh
pytest tests
```

Running a test file directly adds a differential sweep against the reference
implementation, which needs PyTorch:

```sh
python tests/test_nv.py                # sweep skipped, with a note
pip install -r requirements.txt
python tests/test_nv.py                # + bit-exact differential sweep
python tests/test_amd.py
```

`--trials N` and `--seed N` set the sweep's size and seed.
