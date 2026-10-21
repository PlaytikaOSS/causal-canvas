# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.1] — 2026-10-21

Packaging release for the public repository and PyPI; no library code changes.

### Added

- `tools/sanitize_for_oss.py` — exports a tag as a sanitised tree for the public repository (internal CI files, headers and PR references removed; PyPI metadata added). Not part of the exported tree.

### Changed

- Public package metadata: MIT licence, classifiers, keywords and project URLs; README links made absolute so they resolve on PyPI.

## [0.2.0] — 2026-10-21

Breaking release — see [ADR-0001](docs/adr/0001-causalnex-migration.md).

### Changed

- **BREAKING:** `causalnex` removed. The Bayesian network and inference now use `pgmpy` 1.1 (`causal_canvas.bayesian_network.BayesianNetwork`, `causal_canvas.inference_engine.InferenceEngine`). `InferenceEngine.query(parallel=...)` is gone; use `query(nodes=[...])` to restrict work to the nodes you need.
- **BREAKING:** previously saved model artefacts (`bayesian_network.pkl`) cannot be loaded; `safe_pickle_load` says to retrain.
- **BREAKING:** dependency floor raised with pgmpy 1.1 / DoWhy 0.14: `numpy>=2`, `pandas>=2.2.2`, `plotnine>=0.14`, Python `>=3.10,<3.13`.
- `torch` and `pyarrow` are no longer capped by the causalnex pin chain (`torch>=2.13`, `pyarrow>=19`), closing the transitive CVEs listed in `docs/SECURITY_AUDIT.md`.
- `evaluate_experiments.py` no longer crashes on `Path + str` output paths; `plot_conditionals` reads the CPD csv index column positionally; `BayesianNetworkEstimator.structural_model` accepts any `networkx.DiGraph` (a `StructureModel` or one read back from `.dot`); the example configs use the `cutoffs_target` key the schema actually reads.
- `predict` / `predict_probability`: with all parents present they read the CPT (other columns ignored, as before); with a parent missing they use exact posteriors over all observed columns rather than joint MAP.

### Added

- `causal_canvas.vendored` — the causalnex NOTEARS, distribution-type, discretiser and plotting code we depend on, owned in-tree (Apache-2.0, licence kept).
- `causal_canvas.effect_estimation.estimate_effect` — optional DoWhy identification / estimation / refutation (`pip install causal-canvas[dowhy]`).
- Optional extra `causal-canvas[mdlp]` for the MDLP discretiser (was a causalnex extra).
- Numerical-parity tests for marginal, conditional and interventional queries (`tests/test_bayesian_network_pgmpy.py`).
- ADR-0001 (`docs/adr/`).
- `tests/test_torch_compat.py` (`slow`): guards the vendored NOTEARS-MLP and every distribution type against torch releases.
- `benchmarks/`: `bench_inference.py` (JSON report with machine spec) and `export_legacy_model.py` to benchmark a pre-0.2.0 model from its exported CPTs without retraining.

## [0.1.0] — 2026-04-25

First tagged release. Consolidates a large cleanup cycle  that hardened typing, packaging, CI, and security, brought the README back in sync with the code, ported the GitHub wiki content in-tree, and added an experimental uv-on-CI probe so Phase B (causalnex migration) can land on a maintained package manager. Local validation: 54 tests pass in 28s on Apple M1 Pro / 32 GB / Python 3.10.19.

### Added

- Console scripts for the five entry points: `infer-bn`, `infer-dowhy`, `evaluate-experiments`, `repeat-experiments`, `iterate-structure-learning`.
- `--debug` flag on entry points to log the full config on startup; the default now logs a short summary only.
- Curated public API surface in `causal_canvas.__init__`: `StructureLearner`, `BayesianNetworkEstimator`, `Preprocessor`, `InferenceMethod`.
- `py.typed` marker so downstream consumers get types via `mypy` / editors.
- Smoke-test suite for core modules.
- Pickle-load path validation and artefact SHA-256 hashing in `BayesianNetworkEstimator` / `infer_do_why`.
- `tox -e lint` environment (pre-commit: ruff + mypy + pytest); `tox -e audit` (pip-audit, non-blocking).
- Ruff rule sets: `F` (pyflakes), `B` (bugbear), `UP` (pyupgrade), `E` (pycodestyle errors).
- Module-boundary docstrings on `custom_notears_mlp.py` and `nonlinear_notears.py`.
- `docs/SECURITY_AUDIT.md` cataloguing known CVEs in the dep surface with a triage flow.
- Concept docs ported in-tree from the GitHub wiki: `docs/concepts/{introduction,bayesian-networks-estimation,linear-causal-discovery,nonlinear-causal-discovery-deep-dive,interventions-counterfactuals-deep-dive}.md`. Configuration reference consolidated into `docs/reference/config.md`. External wiki links removed from the README.
- Experimental `[testenv:test-uv]` running pytest under uv. Currently inert in default CI.

### Changed

- `pyproject.toml` migrated to PEP 621 `[project]` layout, with a hybrid `[tool.poetry]` block retained for Poetry 1.x in the Playtika CI image.
- Migrated to `pydantic` v2 throughout; lock file regenerated; `conint`/`confloat` replaced with `Annotated + Field`.
- `condenced_thresholds` → `condensed_thresholds` (pydantic field) and `condenced_splits.json` → `condensed_splits.json` (artefact filename). Straight rename, no deprecation shim.
- `repeat_experiments.py` hardcoded `/mnt/VAST_SHARED/...` paths parameterised as Fire CLI args with Playtika-default values.
- Config logging: entry points now emit a compact summary by default (`event=... data=... output=...`); pass `--debug` for the full dump.
- README fully rewritten file tree, install section, and CLI invocations against the current code; typos fixed; stale conda/pip path dropped.

### Fixed

- `infer_do_why` no longer blocks on interactive matplotlib backends.
- `tox -e test` now actually runs the test suite; coverage target corrected; `--doctest-modules` removed.
- Mypy-clean baseline across the package: `custom_notears_mlp`, `nonlinear_notears`, `structure_learner`, `bayesian_network_estimator`, `evaluation_utils`, `DistTypeCategorical`, and the non-linear-NOTEARS body-level errors.
- Grammar slip in `StructureLearner` docstring: "rules care used top in order to avoid" → "rules are used to avoid".

### Removed

- `requirements.txt` (superseded by Poetry; console scripts are the install interface).
- `subprocess(..., shell=True)` in experiment runners, replaced with argv-list form.
- `dummy_example_results/` — 11 MB, 102 files of frozen pre-rename artefacts that no test referenced.

### Security

- Validate pickle-load paths and log artefact hashes before unpickling saved models; mitigates arbitrary-code-execution risk from untrusted checkpoint files.
- Drop `subprocess(shell=True)` in experiment runners to eliminate a shell-injection vector.
- `tox -e audit` runs `pip-audit` against the installed environment; findings are triaged in `docs/SECURITY_AUDIT.md`.

[Unreleased]: https://github.com/PlaytikaOSS/causal-canvas/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/PlaytikaOSS/causal-canvas/releases/tag/v0.1.0
