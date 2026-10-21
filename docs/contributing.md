# Contributing

The full contributor guide lives at [`CONTRIBUTING.md`](https://github.com/PlaytikaOSS/causal-canvas/blob/main/CONTRIBUTING.md) in the repo root. This page is a thin wrapper so the documentation site has the entry point in its nav.

Highlights:

- Dev setup is `poetry install` + `poetry run pre-commit install`.
- CI runs `tox -e test` (pytest + coverage), `tox -e lint` (pre-commit: ruff + mypy + pytest), and `tox -e audit` (pip-audit, non-blocking).
- Branch from `develop`, conventional-commits subject lines, one logical change per PR.

For everything else, see the canonical guide.
