# Getting started

Causal Canvas runs from a single YAML config. The fastest way to see it work is the bundled UCI student-grades example.

## 1. Install

```bash
git clone https://github.com/PlaytikaOSS/causal-canvas.git
cd causal_canvas
poetry config virtualenvs.in-project true
poetry install
```

Python 3.10, 3.11 or 3.12 is required.

## 2. Run the example

```bash
poetry run infer-bn --config_file examples/students/config.yaml
```

This learns a causal DAG over UCI Portuguese-grades features (with `G3`, the final grade, as the target), discretises numeric features, fits a Bayesian network, and writes results to the path declared in the config's `output_path` field.

## 3. Pick the next step

- :material-clipboard-text: **Product manager** — read [What Causal Canvas does for product decisions](for-pms/overview.md) (10-minute read).
- :material-chart-line: **Analyst** — follow the [analyst quickstart](for-analysts/quickstart.md) and the [interpretation guide](for-analysts/interpreting-results.md).
- :material-function: **Statistician** — go to [assumptions and identifiability](for-statisticians/assumptions.md).

For the full configuration schema, the CLI flags, and the API reference, see [Reference](reference/index.md).
