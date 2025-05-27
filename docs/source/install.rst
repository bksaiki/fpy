Installation
==================

.. manually documented: keep in sync with README.md

Requirements:

* Python 3.11 or later

The following instructions assume a `bash`-like shell.
If you do not have a Python virtual environment,
create one using::

    python3 -m venv .env/

and activate it using using::

    source .env/bin/activate

To install a _frozen_ instance of FPy, run::

    pip install .

or with `make`, run::

    make install

Note that this will not install the necessary dependencies for
development and installs a copy of the `fpy2` package.

.. note::

    If you checkout a different commit or branch, you will
    need to reinstall FPy to overwrite the previous version.

    To uninstall FPy, run::

        pip uninstall fpy2
