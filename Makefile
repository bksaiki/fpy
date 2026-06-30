# Makefile for running tests and linting

UNITTEST_PATTERN ?= ''


default: help;

build:
	@echo "Building fpy2..."
	python3 -m build

docs:
	@echo "Building documentation..."
	make html -C docs/

lint:
	@echo "Running linters..."
	$(MAKE) mypy
	$(MAKE) ruff

mypy:
	@echo "Running mypy..."
	mypy fpy2

ruff:
	@echo "Running ruff..."
	ruff check fpy2

tests: 
	@echo "Running all tests..."
	$(MAKE) lint
	$(MAKE) infratest
	$(MAKE) unittest

infratest:
	@echo "Running infrastructure tests..."
	python3 -m tests.infra
	python3 -m tests.infra.fpcore

unittest:
	@echo "Running unit tests..."
	python3 -m pytest -v tests/unit -k $(UNITTEST_PATTERN)

cpptest:
	@echo "Running C++ backend tests (compile + execute, bit-compare vs interpreter)..."
	python3 -m tests.infra.backend.cpp --mode run

clean: clean-docs
	@echo "Cleaning build artifacts..."
	rm -rf build/ dist/ *.egg-info

clean-docs:
	@echo "Cleaning documentation..."
	make clean -C docs/


help:
	@echo "FPy Make targets"
	@echo ""
	@echo "Testing"
	@echo "  make tests         Run all tests"
	@echo "  make infratest     Run infrastructure testing"
	@echo "  make unittest      Run unit tests"
	@echo "  make cpptest       Compile + execute the C++ backend corpus (needs a C++ compiler)"
	@echo "  make lint          Run linters"
	@echo "   - make mypy       Run mypy type checker"
	@echo "   - make ruff       Run ruff linter"
	@echo ""
	@echo "Build"
	@echo "  make build         Build the fpy2 package (sdist + wheel)"
	@echo ""
	@echo "Documentation"
	@echo "  make docs          Build the documentation"
	@echo ""
	@echo "Miscellaneous"
	@echo "  make help          Show this help message"
	@echo "  make clean         Clean build artifacts"
	@echo "  make clean-docs    Clean documentation build artifacts"
	@echo ""

.PHONY: docs tests cpptest
