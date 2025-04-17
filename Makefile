# Makefile for running tests and linting

help:
	@echo "Makefile for running tests and linting"
	@echo "Usage:"
	@echo "  make infratest: 	Run infrastructure testing"
	@echo "  make unittest      Run unit tests"
	@echo "  make lint         	Run linters"

lint:
	mypy fpy2
	ruff check fpy2

infratest:
	python3 -m tests.unit
	python3 -m tests.fpbench

unittest:
	python3 -m unittest -v 

.PHONY: infratest unittest
