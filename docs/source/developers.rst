Developers of FPy
==================

.. manually documented: keep in sync with README.md

Developers of FPy should read this section since
installing FPy is actually different.

Requirements:
 
* Python 3.11 or later
* `make`

Installation
------------------

If you do not have a Python virtual environment,
create one using::

    python3 -m venv .env/

and activate it using using::

    source .env/bin/activate

To install an instance of FPy for development, run::

    pip install -e .[dev]

or with `make`, run::

    make install-dev

Testing
------------------

There are a number of tests that can be run through
the `Makefile` including::

    make lint

to ensure formatting and type safety::

    make unittest

to run the unit tests::

    make infratest

to run the infrastructure tests.

Documentation
------------------

Documentation is generated using `sphinx` and is located in the `docs/` directory.
To build the documentation, run::

    make docs

Internal
------------------

.. toctree::
   :maxdepth: 1
   :caption: Table of Contents

   ast
   types
