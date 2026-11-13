# Repository Guidelines

## Project structure and module organization

Vendetect is a Python CLI for detecting vendored or copied code between repositories.
Source code lives in `src/vendetect/`: `_cli.py` handles arguments and output,
`repo.py` handles repositories and history, `detector.py` coordinates detection,
and `comparison.py`, `copydetect.py`, `diffing.py`, and `metrics.py` implement
comparison behavior. Add tests under `test/`, alongside the corresponding
`test_<module>.py` files. CI workflows live in `.github/workflows/`.
Generated documentation (`html/`) and build artifacts (`dist/`) are ignored.

## Build, test, and development commands

Use `uv` to manage dependencies. Develop with Python 3.13 while preserving support
for Python 3.11 and later; CI tests Python 3.11–3.13.

- `make dev`: install all development dependency groups.
- `make run ARGS="./test-repo ./source-repo --type py"`: compare local repositories.
- `make test`: run pytest with coverage and a 300-second per-test timeout.
- `make test T=test/test_metrics.py`: run one test file.
- `make test TESTS=metric`: select tests by name and stop at the first failure.
- `make lint`: check Ruff formatting, Ruff lint rules, and types with `ty`.
- `make format`: apply Ruff formatting and automatic lint fixes.
- `make build`: build distributions with `uv build`.
- `make doc`: generate API documentation with pdoc in `html/`.

## Coding style and naming conventions

Use four-space indentation, `snake_case` for functions and modules, `PascalCase`
for classes, and `UPPER_SNAKE_CASE` for constants. Use absolute imports, annotate
new APIs, and write Google-style docstrings for non-trivial public APIs. Keep new
lines within 100 characters; Ruff currently permits 120. Resolve lint and type
errors before submitting.

## Testing guidelines

Use pytest with descriptive `test_*` functions or existing `unittest.TestCase`
classes. Test observable behavior, boundary values, and error paths. Add regression
tests for fixes; metric changes should cover ranking, thresholds, and output formats.
Repository-history tests require Git and a full clone. Coverage excludes `_cli.py`,
and no minimum coverage percentage is currently enforced.

## Commit and pull request guidelines

History generally uses short, imperative subjects such as `Fix repository commit
regression` and `Bump ...`. Keep subjects within 72 characters and commits focused.
Submit feature branches through pull requests. Describe the resulting behavior,
link relevant issues, and report validation commands. Run relevant tests and
`make lint`; update `README.md` when CLI behavior changes.
