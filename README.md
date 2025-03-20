# FPy

An embedded Python DSL for numerical computing.

## FPyDebug

This branch is an archived version of FPyDebug,
  a project for the software engineering course
  at the University of Washington.
FPyDebug is a numerical accuracy debugging tool that provides:
  - a function profiler for detecting numerical error in FPy functions
  - an expression profiler for ranking expressions by how
    likely they contribute to the function's numerical error
  - a real number evaluator that extends the Rival interval library
    to evaluate over statements and complex control flow.

## Building

Create a virtual environment:
```bash
python3 -m venv .env/
```
and activate it using using
```bash
source .env/bin/activate
```
Then, install the dependencies:
```bash
pip install .[dev]
```
