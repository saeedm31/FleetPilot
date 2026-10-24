# Contributing

1. `make install` (installs dev tools and the pre-commit hooks: ruff lint + format, YAML check, secret detection).
2. Create a branch, make the change, add or update tests.
3. `make lint test` must pass. CI runs lint, tests, an offline evaluation of the copilot and a full Docker Compose run.
4. Open a pull request with: what changed, why, and how it was tested. One reviewer approves.

Conventions: type hints, small pure functions, settings from environment variables, plain-English docstrings.
Architecture decisions go in `docs/adr/` (one short file per decision).
