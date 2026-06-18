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

With ``uv`` (preferred)
^^^^^^^^^^^^^^^^^^^^^^^^^^

`uv <https://docs.astral.sh/uv/>`_ is the recommended development
workflow.  It handles the virtual environment and dependency
installation in a single step::

    uv sync

This creates ``.venv/`` and installs FPy in editable mode along with
the ``dev`` dependency group.  Activate the environment with::

    source .venv/bin/activate

or prefix individual commands with ``uv run`` (e.g. ``uv run pytest tests/unit``).

With ``pip`` (legacy)
^^^^^^^^^^^^^^^^^^^^^^^^^^

This path is preserved for compatibility with existing tooling; new
contributors should prefer ``uv`` above.

If you do not have a Python virtual environment, create one using::

    python3 -m venv .venv/

and activate it using::

    source .venv/bin/activate

To install an instance of FPy for development, run::

    pip install -e .[dev]

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
   semantics
