# Security audit

`tox -e audit` runs [pip-audit](https://github.com/pypa/pip-audit) against the installed dependency tree and reports known vulnerabilities from the [PyPI Advisory Database](https://github.com/pypa/advisory-database) and OSV.

**Why `pip-audit` isn't in `pyproject.toml`**: it's installed at the tox-env level (via `deps = pip-audit` in `tox.ini`) rather than as a Poetry dev-dep. Its transitive tree pulls in `requests` → `certifi` (MPL-2.0), which conflicts with Playtika's open-source license policy. Keeping it out of `poetry.lock` keeps the project's distribution surface clean.

## Current posture: non-blocking

The `[testenv:audit]` section in `tox.ini` is configured to report findings but **not fail the build**. This is deliberate while there's a backlog of CVEs in transitive deps we can't upgrade in a single step (see below). New findings show up in the CI log; someone has to actually look at them for the signal to be useful.

Once the backlog is cleared, flip to strict mode by removing the `-` prefix from `tox.ini`'s `pip-audit` command.

## Current known issues (as of 0.1.0)

27 advisories across 14 packages on the dev+runtime tree. Grouped by whether they're addressable standalone or blocked on a dependency chain:

### Resolved by the `causalnex` migration (0.2.0)

`causalnex` (0.12.1) was archived upstream and pinned old versions of `torch` and `pyarrow`. 0.2.0 removed it (ADR-0001: pgmpy for the Bayesian network, NOTEARS vendored), so `torch>=2.13` and `pyarrow>=19` are now allowed. The entries below are kept for history and should disappear from `tox -e audit` once the lockfile is refreshed:

- **`torch` 2.2.1** — 5 CVEs (PYSEC-2025-41, PYSEC-2024-259, CVE-2025-2953, CVE-2025-3730)
- **`pyarrow` 15.0.0** — PYSEC-2024-161 (fix is in 17.0.0, the old constraint capped it at `<16.0.0` because of causalnex compat)

Tracking in the dependency-migration project.

### Addressable in separate PRs (dev tools / transitive tool chain)

These aren't load-bearing — upgrade whenever:

| Package | Version | CVEs | Fix |
|---|---|---|---|
| `pytest` | 8.0.2 | CVE-2025-71176 | 9.0.3 (major bump — check compat) |
| `setuptools` | 69.1.1 | PYSEC-2025-49, CVE-2024-6345 | 78.1.1+ |
| `pip` | 26.0.1 | CVE-2026-3219 | pending |
| `virtualenv` | 20.25.1 | PYSEC-2024-187, CVE-2026-22702 | 20.36.1 |
| `wheel` | 0.42.0 | CVE-2026-24049 | 0.46.2 |
| `jinja2` | 3.1.3 | 4 CVEs | 3.1.6 |
| `pillow` | 10.2.0 | CVE-2024-28219 | 10.3.0 |
| `fonttools` | 4.49.0 | CVE-2025-66034 | 4.60.2 |
| `pygments` | 2.17.2 | CVE-2026-4539 | 2.20.0 |
| `filelock` | 3.13.1 | CVE-2025-68146, CVE-2026-22701 | 3.20.3 |
| `tqdm` | 4.66.2 | CVE-2024-34062 | 4.66.3 |
| `scikit-learn` | 1.4.1.post1 | PYSEC-2024-110 | 1.5.0 |

Most of these are transitive through our dev tools (pytest, the jupyter ecosystem). Bumping `poetry.lock` with `poetry lock` periodically will resolve many of them automatically.

## Triaging a new finding

When `tox -e audit` reports a new CVE:

1. **Is the package in our `[project].dependencies`?** If yes, bump the constraint and run `poetry lock`.
2. **Is it transitive through a pinned dep?** Trace with `poetry show --tree <package>`. Fix the leaf package's constraint if we can.
3. **Is it blocked on `causalnex`?** Add to the "blocked" list above with the rationale.
4. **Is it a dev-only false positive?** Use `--ignore-vuln <ID>` on the `tox.ini` command line with a one-line comment explaining why.

## Tightening to strict mode

When the backlog is empty enough to be manageable (~0-3 `--ignore-vuln` entries), remove the `-` prefix from `tox.ini`'s `pip-audit` command. New CVEs then fail CI until triaged.
