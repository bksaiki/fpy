# MMA-Sim

Bit-accurate FPy models of NVIDIA Tensor Core MMA arithmetic, following
[MMA-Sim](https://github.com/microsoft/MMA-Sim) (Xie et al.,
[arXiv:2511.10909](https://arxiv.org/abs/2511.10909)).

- [`nv.py`](nv.py) — the models: `make_fma_dpa`, `make_t_fdpa`,
  `make_t_fdpa_chain`, `make_st_fdpa`, `make_gst_fdpa`. Each factory takes
  per-instruction parameters (formats, F, rho, e_zero) and returns an FPy
  dot-product-accumulate. Running it reproduces the paper's Table 8.
- [`validate_nv.py`](validate_nv.py) — validation against the MMA-Sim
  reference implementation.

```sh
python nv.py             # Table 8 demo

python validate_nv.py    # directed checks (fpy2 only)
pip install -r requirements.txt
python validate_nv.py    # + bit-exact differential sweep
```
