SHELL := /bin/bash

PY_IMPORT = vendetect

# Optionally overridden by the user in the `test` target.
TESTS :=

# If the user selects a specific test pattern to run, set `pytest` to fail fast
# and only run tests that match the pattern.
# Otherwise, run all tests and enable coverage assertions, since we expect
# complete test coverage.
ifneq ($(TESTS),)
	TEST_ARGS := -x -k $(TESTS)
	COV_ARGS :=
else
	TEST_ARGS :=
	# COV_ARGS := --fail-under 100
	COV_ARGS :=
endif

.PHONY: all
all:
	@echo "Run my targets individually!"

.PHONY: run
run:
	uv run vendetect $(ARGS)

.PHONY: lint
lint:
	uv sync --group lint
	uv run ruff format --check && \
		uv run ruff check && \
		uv run ty check

.PHONY: format
format:
	uv sync --group lint
	uv run ruff format && \
		uv run ruff check --fix

.PHONY: test
test:
	uv sync --group test
	uv run pytest -svv --timeout=300 --cov=$(PY_IMPORT) $(T) $(TEST_ARGS)
	uv run coverage report -m $(COV_ARGS)

.PHONY: doc
doc:
	uv run pdoc -o html $(PY_IMPORT)

.PHONY: build
build:
	uv build
