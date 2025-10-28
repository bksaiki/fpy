Language
==================

Decorators
------------------

FPy programs are declared using the :py:deco:`fpy2.fpy` decorator.

.. autofunction:: fpy2.fpy

FPy primitive are declared using the :py:deco:`fpy2.fpy_primitive` decorator.

.. autofunction:: fpy2.fpy_primitive

Primitives are arbitrary Python functions that can be called with FPy programs.
They must have type annotations for all arguments and the return value.

Runtime
------------------

FPy programs are represented by the :py:class:`fpy2.Function` class.

.. autoclass:: fpy2.Function
   :members:
   :show-inheritance:

.. autoclass:: fpy2.ForeignEnv
   :members:
   :show-inheritance:

Interpreters
------------------

FPy programs are interpreted by any implementation of the :py:class:`fpy2.Interpreter` class.

.. autoclass:: fpy2.Interpreter
   :members:
   :show-inheritance:

FPy provides a single interpreter:

.. autoclass:: fpy2.DefaultInterpreter
   :members:
   :show-inheritance:

Calling a :py:class:`fpy2.Function` object invokes the default interpreter.
A user can set or get the default interpreter using the following functions:

.. autofunction:: fpy2.set_default_interpreter
.. autofunction:: fpy2.get_default_interpreter


Compatability
------------------

FPy is compatabile with FPCore, a minimal functional language for specifying
numerical algorithms.

.. autoclass:: fpy2.FPCoreContext
   :members:
   :show-inheritance:
