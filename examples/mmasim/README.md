# MMA-Sim

Bit-accurate FPy models of NVIDIA Tensor Core and AMD Matrix Core MMA
arithmetic, following [MMA-Sim](https://github.com/microsoft/MMA-Sim)
(Xie et al., [arXiv:2511.10909](https://arxiv.org/abs/2511.10909)).

- [`nv.py`](nv.py) — NVIDIA models: `make_t_fdpa`, `make_t_fdpa_chain`,
  `make_st_fdpa`, `make_gst_fdpa`.
- [`amd.py`](amd.py) — AMD models: `make_e_fdpa`, `make_ftz_addmul`,
  `make_tr_fdpa`, `make_gtr_fdpa`.
- [`utils.py`](utils.py) — shared helpers and `make_fma_dpa`.
- [`validate_nv.py`](validate_nv.py) / [`validate_amd.py`](validate_amd.py)
  — validation against the MMA-Sim reference implementation.

Each factory takes per-instruction parameters (formats as contexts, F,
rho, ...) and returns an FPy dot-product-accumulate. Running a model
file reproduces its rows of the paper's Table 8.

```sh
python nv.py                # Table 8 demo (or amd.py)

python validate_nv.py       # directed checks (fpy2 only)
pip install -r requirements.txt
python validate_nv.py       # + bit-exact differential sweep
python validate_amd.py
```
