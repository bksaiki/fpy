# Makefile for running tests and linting

UNITTEST_PATTERN ?= ''


default: help;

build:
	@echo "Building fpy2..."
	python3 -m build

docs:
	@echo "Building documentation..."
	make html -C docs/

clean-docs:
	@echo "Cleaning documentation..."
	make clean -C docs/

install:
	@echo "Installing fpy2..."
	pip install .

install-dev:
	@echo "Installing fpy2 in development mode..."
	pip install -e .[dev]

lint:
	@echo "Running linters..."
	mypy fpy2
	ruff check fpy2

tests: 
	@echo "Running all tests..."
	$(MAKE) lint
	$(MAKE) infratest
	$(MAKE) unittest

infratest:
	@echo "Running infrastructure tests..."
	python3 -m tests.infra.unit
	python3 -m tests.infra.fpbench

unittest:
	@echo "Running unit tests..."
	python3 -m unittest -v -k $(UNITTEST_PATTERN)

help:
	@echo "FPy Make targets"
	@echo ""
	@echo "Testing"
	@echo "  make tests         Run all tests"
	@echo "  make infratest     Run infrastructure testing"
	@echo "  make unittest      Run unit tests"
	@echo "  make lint          Run linters"
	@echo ""
	@echo "Install / Build"
	@echo "  make build         Build the fpy2 package"
	@echo "  make install       Install the fpy2 package"
	@echo "  make install-dev   Install the fpy2 package in development mode"
	@echo ""
	@echo "Documentation"
	@echo ""
	@echo "  make docs          Build the documentation"
	@echo "  make clean-docs    Clean the HTML documentation"
	@echo ""
	@echo "Miscellaneous"
	@echo "  make help          Show this help message"
	@echo

.PHONY: docs tests
