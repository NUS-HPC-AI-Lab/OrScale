# Contributing to OrScale

Thanks for your interest in improving OrScale. This project is research software, so contributions should keep the core optimizer behavior easy to inspect and test.

## Development Setup

```bash
python -m pip install -e ".[dev]"
```

Install optional extras when working on a specific area:

```bash
python -m pip install -e ".[vision,data,eval,analysis,wandb,dev]"
```

## Tests

Run the default test suite before opening a pull request:

```bash
pytest tests/ -v
```

Some downstream evaluation tests are skipped automatically unless `lm-eval` is installed.

## Pull Requests

- Keep changes focused and include tests for optimizer, routing, data-loading, or scheduler behavior changes.
- Prefer small, readable implementations over broad refactors.
- Do not commit checkpoints, datasets, generated reports, W&B runs, or machine-specific logs.
- Use relative paths or documented command-line overrides in configs and examples.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
