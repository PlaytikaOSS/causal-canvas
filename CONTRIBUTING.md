# Contributing to Causal Canvas

Thanks for your interest in improving Causal Canvas. This guide covers the development setup, the checks CI runs, and the conventions we follow for commits and pull requests.

## Development setup

Causal Canvas targets **Python 3.10 – 3.12** and uses [Poetry](https://python-poetry.org/) for dependency management.

```bash
git clone https://github.com/PlaytikaOSS/causal-canvas.git
cd causal_canvas
poetry config virtualenvs.in-project true
poetry install
```

The `.pre-commit-config.yaml` wires up `ruff`, `mypy`, `pytest`, and a handful of hygiene hooks. Activate it once per clone:

```bash
poetry run pre-commit install
```

## Running the checks

Two tox environments cover everything CI runs:

```bash
tox -e test          # pytest + coverage
tox -e lint          # pre-commit run --all-files (ruff + mypy + pytest)
tox -e audit         # pip-audit (non-blocking; see docs/SECURITY_AUDIT.md)
```

### Fast feedback during development

If you want a tighter loop than tox:

```bash
poetry run pytest tests/ -m "not slow"      # ~1s: skip NOTEARS training
poetry run pytest tests/                    # full suite incl. slow tests
poetry run ruff check causal_canvas/ tests/
poetry run mypy causal_canvas/
```

### The `slow` marker

Tests that exercise full training loops (`NOTEARS`, `NotearsMLP`, Bayesian-network fit) are tagged `@pytest.mark.slow`. They run in CI but are easy to skip locally with `-m "not slow"`.

## Making changes

1. **Branch from `develop`.** Use a short, descriptive branch name prefixed by type: `fix/…`, `docs/…`, `chore/…`, `refactor/…`, `feat/…`.
2. **Keep PRs focused.** One logical change per PR. Mixed-concern PRs are hard to review and hard to revert.
3. **Write a conventional-commits subject line** (`type(scope): summary`), with an imperative body that explains *why*, not just *what*. Recent merged PRs are the best style reference.
4. **Document notable changes in `CHANGELOG.md`** under `## [Unreleased]` in the relevant `Added` / `Changed` / `Fixed` / `Removed` / `Security` subsection.

## Pull request checklist

- [ ] `tox -e test` green locally.
- [ ] `tox -e lint` green locally.
- [ ] `CHANGELOG.md` updated under `[Unreleased]` if the change is user-visible.
- [ ] Added or updated tests for new behaviour.
- [ ] Docstrings on new public APIs use the NumPy docstring style (matches the rest of the codebase).
- [ ] No `dummy_example_results/` artefacts committed — outputs stay local.

## Commit message style

```
type(scope): imperative summary in under 70 characters

One or more paragraphs explaining motivation and any non-obvious
design decisions. Wrap at 72 columns.

Refs or closes issue / PR numbers where appropriate.
```

Common types: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `ci`, `security`, `deps`, `packaging`, `api`, `typing`.

## Versioning and releases

`pyproject.toml` carries the released version. Record user-visible changes in `CHANGELOG.md` under `## [Unreleased]`; they move under a version heading at release time.

## Reporting issues

Use the [issue templates](.github/ISSUE_TEMPLATE) — bug report for defects, feature request for enhancements. For security vulnerabilities, please contact the maintainers privately rather than opening a public issue.
