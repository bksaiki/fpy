# FPy

An embedded Python DSL for specifying and simulating numerical algorithms.

Important links:
 - PyPI package: [fpy2](https://pypi.org/project/fpy2/)
 - Documentation: [fpy.readthedocs.io](https://fpy.readthedocs.io/)
 - GitHub: [fpy](https://github.com/bksaiki/fpy)
 - Guide: [USAGE.md](docs/USAGE.md)

## Installation

The recommended way to install FPy is through `pip`.
FPy can also be installed from source for development.
The following instructions assume a `bash`-like shell.

### Installing with `pip`

Requirements:
 - Python 3.11 or later

To install the latest stable release of FPy, run:
```bash
pip install fpy2
```

### Installing from source

Requirements:
 - Python 3.11 or later
 - `make`

#### With `pip`

If you do not have a Python virtual environment,
create one using
```bash
python3 -m venv .env/
```
and activate it using using
```bash
source .env/bin/activate
```
To install an instance of FPy for development, run:
```bash
pip install -e .[dev]
```
or with `make`, run
```bash
make install-dev
```

To uninstall FPy, run:
```bash
pip uninstall fpy2
```

#### With `uv`

[uv](https://docs.astral.sh/uv/) handles the virtual environment and
dependency installation in one step:
```bash
uv sync
```
This creates `.venv/` and installs FPy in editable mode along with the
`dev` dependency group.  Activate the environment with
```bash
source .venv/bin/activate
```
or prefix individual commands with `uv run` (e.g. `uv run pytest tests/unit`).

### Testing

There are a number of tests that can be run through
the `Makefile` including
```bash
make lint
```
to ensure formatting and type safety;
```bash
make unittest
```
to run the unit tests;
```bash
make infratest
```
to run the infrastructure tests.
