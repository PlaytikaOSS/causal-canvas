# Security

Causal Canvas handles two asset types with inherent risk: pickled model files (Bayesian networks, discretisers) and subprocess invocations (when `repeat-experiments` or `iterate-structure-learning` spawn further training jobs). This page documents the guarantees and the limits.

## Loading saved models safely

Bayesian-network and discretiser checkpoints are persisted with `pickle`. Pickle deserialisation **executes arbitrary Python** — the format is not safe for untrusted input.

Causal Canvas routes every load through `causal_canvas.utils.safe_pickle_load`, which:

- Resolves the file path and refuses to load anything outside a *trusted root* (`Path.cwd()` by default). Paths like `../../etc/bad.pkl` are rejected with a `ValueError`.
- Logs a SHA-256 prefix of the loaded bytes before deserialising, so tampering with a file produced in a prior run leaves a trail in the logs.

This limits the blast radius of a misconfigured config that points `model_path` at a surprise location. It does **not** make pickle safe — only load checkpoints produced by trusted runs. If you receive a `.pkl` from a third party, don't feed it to `infer-dowhy` without inspecting its contents first.

## No shell injection in experiment runners

`repeat-experiments` and `iterate-structure-learning` spawn child processes to run individual trainings. Both use `subprocess.check_call([sys.executable, script, ...])` with argv lists — never `shell=True`. Config values passed as command-line arguments are not interpolated into a shell string, so a malicious config cannot smuggle shell commands through.

## Supply-chain auditing

Known CVEs in the dependency tree are scanned with [pip-audit](https://github.com/pypa/pip-audit):

```bash
tox -e audit
```

Findings are **non-blocking** today: the `causalnex`-pinned `torch`/`pyarrow` chain was removed in 0.2.0; remaining advisories are in dev tooling. See [`SECURITY_AUDIT.md`](../SECURITY_AUDIT.md) for the current triage, the upgrade blockers, and the process for new findings.

## Reporting a vulnerability

Please contact the maintainers privately rather than opening a public issue. See [Contributing](../contributing.md).
