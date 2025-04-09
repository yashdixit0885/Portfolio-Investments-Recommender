# Portfolio Investments Recommender Tests

This directory contains tests for the Portfolio Investments Recommender application.

## Test Structure

The tests are organized by component:

- `test_investment_analyzer.py`: Tests for the Investment Analyzer
- `test_research_analyzer.py`: Tests for the Research Analyzer
- `test_trade_analyzer.py`: Tests for the Trade Analyzer
- `test_main.py`: Tests for the main application

## Running Tests

### Running All Tests

To run all tests:

```bash
pytest
```

### Running Critical Tests

To run only the critical tests:

```bash
pytest -m critical
```

### Running Tests for a Specific Component

To run tests for a specific component:

```bash
pytest tests/test_investment_analyzer.py
pytest tests/test_research_analyzer.py
pytest tests/test_trade_analyzer.py
pytest tests/test_main.py
```

### Running Tests with Coverage

To run tests with coverage reporting:

```bash
pytest --cov=src tests/ --cov-report=term-missing
```

## Test Markers

The tests use the following markers:

- `critical`: Marks tests as critical (run before each commit)
- `slow`: Marks tests as slow (can be skipped with `-m "not slow"`)

## Pre-commit Hook

The repository includes a pre-commit hook that runs the critical tests before each commit. To install the pre-commit hook:

```bash
pip install pre-commit
pre-commit install
```

## GitHub Actions

The repository includes a GitHub Actions workflow that runs the tests on each push and pull request. The workflow:

1. Runs the critical tests first
2. Runs all tests
3. Uploads the coverage report to Codecov 