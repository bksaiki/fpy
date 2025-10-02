# FPy

An embedded Python DSL for specifying and simulating numerical algorithms.

Important links:
 - PyPI package: [fpy2](https://pypi.org/project/fpy2/)
 - Documentation: [fpy.readthedocs.io](https://fpy.readthedocs.io/)
 - GitHub: [fpy](https://github.com/bksaiki/fpy)

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

### Installation

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
