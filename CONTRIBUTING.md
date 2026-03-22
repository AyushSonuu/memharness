# Contributing to memharness

Thank you for your interest in contributing to memharness! This document provides guidelines and instructions for contributing.

## Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/memharness.git
   cd memharness
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install development dependencies**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Install pre-commit hooks**
   ```bash
   pre-commit install
   ```

## Code Style

We use [Ruff](https://github.com/astral-sh/ruff) for linting and formatting. The pre-commit hooks will automatically check and fix issues.

- Run linting manually: `ruff check src/`
- Run formatting manually: `ruff format src/`

## Testing

We use [pytest](https://pytest.org/) for testing.

- Run all tests: `pytest tests/ -v`
- Run with coverage: `pytest tests/ -v --cov=memharness --cov-report=html`
- View coverage report: Open `htmlcov/index.html` in your browser

### Writing Tests

- Place tests in the `tests/` directory
- Name test files with `test_` prefix
- Name test functions with `test_` prefix
- Use descriptive test names that explain what is being tested

## Pull Request Process

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write tests for new functionality
   - Ensure all tests pass
   - Follow the code style guidelines

3. **Commit your changes**
   - Use clear, descriptive commit messages
   - Pre-commit hooks will run automatically

4. **Push and create a PR**
   ```bash
   git push origin feature/your-feature-name
   ```
   Then create a Pull Request on GitHub.

5. **Code Review**
   - Address any feedback from reviewers
   - Ensure CI checks pass

## Reporting Issues

When reporting issues, please include:

- A clear description of the problem
- Steps to reproduce the issue
- Expected vs actual behavior
- Python version and OS
- Relevant error messages or logs

## Feature Requests

We welcome feature requests! Please:

- Check if the feature has already been requested
- Clearly describe the use case
- Explain why this feature would be valuable

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Help others learn and grow

## Questions?

If you have questions, feel free to:

- Open a GitHub issue
- Start a discussion in the repository

Thank you for contributing!
