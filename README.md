# FPy

An embedded Python DSL for specifying and simulating numerical algorithms.

Important links:
 - PyPI package: [fpy2](https://pypi.org/project/fpy2/)
 - Documentation: [fpy.readthedocs.io](https://fpy.readthedocs.io/)
 - GitHub: [fpy](https://github.com/bksaiki/fpy)
 - Guide: [USAGE.md](docs/USAGE.md)

## Installation

FPy can be installed from PyPI with either `uv` or `pip`, or built from
source for development.  The following instructions assume a `bash`-like
shell.

### Installing from PyPI

Requirements:
 - Python 3.11 or later

To install the latest stable release of FPy, run:
```bash
uv pip install fpy2
```
or, with `pip`:
```bash
pip install fpy2
```

### Installing from source

Requirements:
 - Python 3.11 or later
 - `make`

#### With `uv` (preferred)

[uv](https://docs.astral.sh/uv/) is the recommended development
workflow — it handles the virtual environment and dependency
installation in a single step:
```bash
uv sync
```
This creates `.venv/` and installs FPy in editable mode along with the
`dev` dependency group.  Activate the environment with
```bash
source .venv/bin/activate
```
or prefix individual commands with `uv run` (e.g. `uv run pytest tests/unit`).

#### With `pip` (legacy)

This path is preserved for compatibility with existing tooling; new
contributors should prefer `uv` above.

If you do not have a Python virtual environment,
create one using
```bash
python3 -m venv .venv/
```
and activate it using
```bash
source .venv/bin/activate
```
To install an instance of FPy for development, run:
```bash
pip install -e .[dev]
```

To uninstall FPy, run:
```bash
pip uninstall fpy2
```

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
