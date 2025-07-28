FAQ
====

This page contains answers to frequently asked questions,
common issues, and other useful information for the FPy library.

I'm getting a `PicklingError` when using FPy.
-----------------------------------------------

Python's `pickle` module does not play nice with decorators,
specifically decorators that produce objects rather than functions.
There are a few sources explaining this issue such as this
`blog on pickling gotchas <https://gael-varoquaux.info/programming/decoration-in-python-done-right-decorating-and-pickling.html>`_.
The problem is that `pickle` imposes restrictions on what can be pickled,
and deserialzies objects in a particular manner.

The `dill` package handles more complex objects, including those created by decorators.
If you wish to serialize an object, use `dill` instead of `pickle`.
The `multiprocessing` module in Python uses `pickle` by default.
Use `pathos` instead, which uses `dill` for serialization.
