# AGENTS.md — FPy Repository Guide

This file provides an orientation for AI agents (and new contributors) working in this repository.

## What is FPy?

FPy is an embedded Python DSL for specifying and simulating numerical algorithms.
It consists of two main components:

1. **Language** — a Python-like syntax for writing numerical algorithms, and
2. **Runtime** — an interpreter and compiler infrastructure for executing FPy code under configurable number systems.

## Documentation

| Document | Location | Contents |
|---|---|---|
| Usage guide | [docs/USAGE.md](docs/USAGE.md) | Language overview, syntax, and examples |
| README | [README.md](README.md) | Installation, testing, and quick links |
| Developer docs | [docs/source/developers.rst](docs/source/developers.rst) | Internal design documentation |
| Online docs | [fpy.readthedocs.io](https://fpy.readthedocs.io/) | Full built documentation |

## Repository Layout

```
fpy/
├── fpy2/               # Main Python package (source code)
├── tests/              # Tests
├── docs/               # Documentation source (Sphinx)
├── examples/           # Example FPy programs
├── exploration/        # Research scripts and experiments
├── infra/              # Infrastructure utilities (e.g., FPBench integration)
├── pyproject.toml      # Package metadata and dependencies
└── Makefile            # Common development tasks
```

## `fpy2/` Package Structure

The main package lives in `fpy2/`. Its top-level modules and subpackages are:

| Path | Description |
|---|---|
| `decorator.py` | The `@fpy` decorator that marks FPy functions |
| `ops.py` | Built-in FPy operations (arithmetic, transcendental, etc.) |
| `types.py` | Type definitions used throughout the package |
| `function.py` | Representation of a compiled FPy function |
| `runner.py` | Entry point for executing FPy functions |
| `env.py` | Execution environment |
| `primitive.py` | Primitive value definitions |
| `ast/` | Abstract syntax tree (AST) node definitions and visitor infrastructure |
| `frontend/` | Parsing, syntax checking, and AST code generation |
| `analysis/` | Compiler analyses (type inference, liveness, reachability, purity, etc.) |
| `transform/` | AST-level transformations (constant folding, inlining, loop unrolling, etc.) |
| `interpret/` | FPy interpreters (bytecode, etc.) |
| `backend/` | Code generation from FPy AST (e.g., FPCore, C++) |
| `rewrite/` | Pattern-based rewriting infrastructure |
| `strategies/` | High-level rewriting strategies |
| `number/` | Number system library (formats, rounding, contexts) |
| `libraries/` | Built-in FPy library functions |
| `lut/` | Look-up table backend utilities |
| `utils/` | Shared utility functions |

## Tests

Tests live in `tests/` and are split into two suites:

- `tests/unit/` — Unit tests, run with `pytest`.
- `tests/infra/` — Infrastructure tests (e.g., FPBench integration).

Run tests via `make`:

```bash
make unittest      # Run unit tests
make infratest     # Run infrastructure tests
make tests         # Run all tests (lint + infratest + unittest)
```

## Linting and Type Checking

The project uses `mypy` for type checking and `ruff` for linting, both scoped to `fpy2/`.

```bash
make lint    # Run mypy and ruff
make mypy    # Type check only
make ruff    # Lint only
```

## Development Setup

```bash
python3 -m venv .venv/
source .venv/bin/activate
make install-dev   # Installs fpy2 in editable mode with dev dependencies
```

## Examples

The `examples/` directory contains runnable FPy programs organized by topic:

- `examples/fpbench/` — FPBench benchmark implementations
- `examples/graphics/` — Graphics-related numerical algorithms
- `examples/misc/` — Miscellaneous examples

## Key Concepts to Know

- **Rounding contexts** control how every operation is rounded. See [docs/USAGE.md](docs/USAGE.md) for details.
- **FPy AST** is the internal representation produced by the frontend and consumed by analyses, transforms, and backends.
- The `@fpy` decorator (in `decorator.py`) triggers parsing and compilation of the decorated function at definition time.
