# Contributing to EvoFusion

Thank you for your interest in contributing! We welcome improvements, bug fixes, and new features.

## How to Contribute
- Fork the repository and create a feature branch.
- Keep changes focused and include tests when possible.
- Run tests locally: `python -m unittest discover -s tests -v`.
- Open a Pull Request with a clear description and rationale.

## Coding Guidelines
- Follow existing code style and structure.
- Prefer small, cohesive changes over large refactors.
- Avoid introducing unrelated changes.

## Reporting Issues
- Provide steps to reproduce, expected vs actual behavior, and environment info.
- Include logs or minimal examples when helpful.

## Development Setup
- Python 3.10
- Install dependencies: `pip install -r requirements.txt`
- CPU-only PyTorch is sufficient for tests.

## Release Process
- Update `README.md` and relevant docs.
- Ensure CI is green.
- Create a tag, e.g., `v0.1.0` and draft release notes.