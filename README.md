# FPy

An embedded Python DSL for specifying and simulating numerical algorithms.

Important links:
 - PyPI package: [fpy2](https://pypi.org/project/fpy2/)
 - Documentation: [fpy.readthedocs.io](https://fpy.readthedocs.io/)
 - GitHub: [fpy](https://github.com/bksaiki/fpy)

## Installation

Requirements:
 - Python 3.11 or later

The following instructions assume a `bash`-like shell.
If you do not have a Python virtual environment,
create one using
```bash
python3 -m venv .env/
```
and activate it using using
```bash
source .env/bin/activate
```

To install a _frozen_ instance of FPy, run:
```bash
pip install .
```
or with `make`, run
```bash
make install
```
Note that this will not install the necessary dependencies for
development and installs a copy of the `fpy2` package.
If you checkout a different commit or branch, you will
need to reinstall FPy to overwrite the previous version.

To uninstall FPy, run:
```bash
pip uninstall fpy2
```

## Development

Developers of FPy should read this section since
installing FPy is actually different.

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
