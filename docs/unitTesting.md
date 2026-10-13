# Unit Testing

The add-on tests are run with `pytest`. Some template tests still use Python's standard `unittest` framework.

## Running Tests Locally

To run the unit test suite locally using `uv`:

``` bash
uv run pytest -q tests
```

This runs the complete test suite used by CI, including the template tests and the add-on's tests in `tests/test_*.py`.

Or execute tests for a specific file:

``` bash
uv run python -m unittest -v tests/unit/template/test_sanity.py
```
