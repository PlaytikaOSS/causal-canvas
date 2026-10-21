# CLI reference

`poetry install` exposes five console scripts that cover the primary workflows.

| Command | Purpose |
|---|---|
| `infer-bn` | Run causal discovery + Bayesian-network fit against a config. |
| `infer-dowhy` | Do-calculus inference (interventions, counterfactuals, ATE) on a fitted Bayesian network. |
| `evaluate-experiments` | Error analysis across a set of trained models. |
| `repeat-experiments` | Loop multiple causal-discovery + BN trainings across config variations. |
| `iterate-structure-learning` | Sweep structure-learning hyperparameters across a grid. |

## Common flags

Entry points are wired through [Fire](https://github.com/google/python-fire), so every function argument is also a CLI flag.

| Flag | Default | Description |
|---|---|---|
| `--config_file PATH` | *required* | YAML config for the run. Schemas differ per command — see [Configuration](config.md). |
| `--debug` | `False` | Log the full parsed config on startup. Defaults to a compact summary (`event=… data=… output=…`). Available on `infer-bn` and `infer-dowhy`. |

## Examples

Run causal discovery + BN fit on the bundled student-grades example:

```bash
poetry run infer-bn --config_file examples/students/config.yaml
```

Do-calculus inference against a fitted model:

```bash
poetry run infer-dowhy --config_file examples/students/config_inference.yaml
```

Error analysis across multiple trained models:

```bash
poetry run evaluate-experiments --config_file examples/students/config_evaluation.yaml
```

Override the Playtika-default dataset paths in `repeat-experiments`:

```bash
poetry run repeat-experiments \
  --segments_path /other/path/segments_ \
  --structures_path /other/path/structures/
```

The entry-point scripts are also runnable as modules (`poetry run python infer_bayesian_network.py --config_file …`). The console-script form is shorter and doesn't depend on the current working directory.

## Use as a Python library

The top-level package exposes a curated API:

```python
from causal_canvas import (
    BayesianNetworkEstimator,
    InferenceMethod,
    Preprocessor,
    StructureLearner,
)
```

The wheel ships with a `py.typed` marker, so `mypy` and editors pick up full type information.

A minimal end-to-end sketch:

```python
import pandas as pd
from causal_canvas import Preprocessor, StructureLearner

df = pd.read_parquet("data/my_features.parquet")

preprocessor = Preprocessor(
    df=df,
    event_col="churn",
    date_col="fs_calc_date",
    drop_columns=[],
    categorical_columns=["country", "platform"],
)
preprocessed = preprocessor.preprocess()

learner = StructureLearner(
    X=preprocessed,
    event_col="churn",
    connections_type="linear",
    max_iter=10_000,
    h_tol=1e-4,
    w_threshold=0.8,
    tabu_edges=[],
)
learner.discover_dag(path="./out/causal_discovery")
```

See [`infer_bayesian_network.py`](https://github.com/PlaytikaOSS/causal-canvas/blob/main/infer_bayesian_network.py) for the full reference driver and the [API reference](api/index.md) for class-level details.
