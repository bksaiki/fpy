Language
==================

Decorators
------------------

FPy programs are declared using the :py:deco:`fpy2.fpy` decorator.

.. autofunction:: fpy2.fpy

Runtime
------------------

FPy programs are represented by the :py:class:`fpy2.Function` class.

.. autoclass:: fpy2.Function
   :members:
   :show-inheritance:

Interpreters
------------------

FPy programs are interpreted by any implementor of the :py:class:`fpy2.Interpreter` class.

.. autoclass:: fpy2.Interpreter
   :members:
   :show-inheritance:

FPy provides a number of concrete interpreters.

.. autoclass:: fpy2.DefaultInterpreter
   :members:
   :show-inheritance:

.. autoclass:: fpy2.PythonInterpreter
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
