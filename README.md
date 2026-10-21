# Causal Canvas

Causal Canvas is a tool for causal discovery with structural learning and probabilistic modelling of
the learnt structure.
It uses a NOTEARS implementation (vendored from `causalnex`, see `causal_canvas/vendored/`) in which a fully connected graph is initially assumed
but then the structure is learned through the NOTEARS algorithm by [Zheng et al. (2018)](https://proceedings.neurips.cc/paper_files/paper/2018/file/e347c51419ffb23ca3fd5050202f9c3d-Paper.pdf).

Non-linear structure learning with NOTEARS-MLP is supported alongside the original linear variant. The tool also performs do-calculus inference (Pearl).

The pipeline reads a YAML config and:

- loads and preprocesses tabular data,
- fits a structure learner (linear or non-linear NOTEARS) under user-supplied tabu constraints,
- prunes the resulting weighted graph to a DAG,
- discretises numeric features and fits a Bayesian-network probabilistic model,
- extracts conditional probabilities for chosen nodes,
- writes interactive HTML graphs and CSV/JSON artefacts to the output directory.

For the conceptual background — causal discovery, Bayesian networks, do-calculus interventions and counterfactuals — see the in-tree documentation:

- [Introduction](https://github.com/PlaytikaOSS/causal-canvas/blob/main/docs/concepts/introduction.md) — methodology overview.
- [Linear causal discovery](https://github.com/PlaytikaOSS/causal-canvas/blob/main/docs/concepts/linear-causal-discovery.md) — NOTEARS and SEMs.
- [Non-linear causal discovery (deep dive)](https://github.com/PlaytikaOSS/causal-canvas/blob/main/docs/for-statisticians/nonlinear-causal-discovery-deep-dive.md) — NOTEARS-MLP and distribution types.
- [Bayesian networks](https://github.com/PlaytikaOSS/causal-canvas/blob/main/docs/concepts/bayesian-networks.md) and [estimation deep dive](https://github.com/PlaytikaOSS/causal-canvas/blob/main/docs/concepts/bayesian-networks-estimation.md).
- [Interventions and counterfactuals](https://github.com/PlaytikaOSS/causal-canvas/blob/main/docs/concepts/interventions-counterfactuals.md) and [deep dive](https://github.com/PlaytikaOSS/causal-canvas/blob/main/docs/concepts/interventions-counterfactuals-deep-dive.md).

The vendored NOTEARS code originates from [`causalnex`](https://causalnex.readthedocs.io/en/latest/index.html#) (Apache-2.0, archived upstream). Bayesian-network fitting and inference are provided by [`pgmpy`](https://pgmpy.org); optional causal-effect estimation by [`DoWhy`](https://www.pywhy.org/dowhy/) (`pip install causal-canvas[dowhy]`). See [ADR-0001](https://github.com/PlaytikaOSS/causal-canvas/blob/main/docs/adr/0001-causalnex-migration.md).

## Code structure

Source code

```text
├── causal_canvas                       <- Source package.
│   ├── __init__.py                     <- Curated public API.
│   ├── bayesian_network.py             <- pgmpy-backed discrete BN (CPT fit, predict).
│   ├── bayesian_network_estimator.py   <- Bayesian Network fitting class.
│   ├── custom_distributions/           <- Distribution types for non-linear NOTEARS.
│   ├── custom_notears_mlp.py           <- Low-level NOTEARS-MLP solver.
│   ├── effect_estimation.py            <- Optional DoWhy effect estimation.
│   ├── evaluation_utils.py             <- Error-analysis helpers.
│   ├── inference_engine.py             <- Exact marginals + do() interventions (pgmpy).
│   ├── inference_utils.py              <- Do-why inference helpers.
│   ├── nonlinear_notears.py            <- High-level non-linear-NOTEARS API.
│   ├── preprocessor.py                 <- Data preprocessing.
│   ├── py.typed                        <- PEP 561 type marker.
│   ├── script_config.py                <- Pydantic config schema.
│   ├── structure_learner.py            <- Causal discovery (linear + non-linear NOTEARS).
│   ├── utils.py                        <- Shared utilities.
│   └── vendored/                       <- NOTEARS / discretisers / plots owned in-tree (from causalnex, Apache-2.0).
├── data/                               <- Example input data.
├── docs/                               <- Documentation site source.
├── examples/                            <- Example scenarios (configs + per-scenario READMEs).
├── images/                             <- Diagrams and figures.
├── notebooks/                          <- Jupyter notebooks.
├── tests/                              <- Pytest test suite.
├── evaluate_experiments.py             <- Entry point → console script `evaluate-experiments`.
├── infer_bayesian_network.py           <- Entry point → console script `infer-bn`.
├── infer_do_why.py                     <- Entry point → console script `infer-dowhy`.
├── iterate_structure_learning.py       <- Entry point → console script `iterate-structure-learning`.
├── repeat_experiments.py               <- Entry point → console script `repeat-experiments`.
├── pyproject.toml                      <- Package metadata + tooling config.
├── poetry.lock                         <- Locked dependency versions.
├── tox.ini                             <- Tox environments (test, lint).
├── .pre-commit-config.yaml             <- Pre-commit hooks.
└── README.md                           <- This file.
```

## Install

Causal Canvas uses [Poetry](https://python-poetry.org/) for dependency management. Python 3.10, 3.11 or 3.12 is required.

```bash
poetry config virtualenvs.in-project true
poetry install
```

## Command-line reference

`poetry install` exposes five console scripts that cover the primary workflows:

| Command | Purpose |
|---|---|
| `infer-bn` | Run causal discovery + Bayesian-network fit against a config. |
| `infer-dowhy` | Do-calculus inference (interventions, counterfactuals, ATE) on a fitted Bayesian network. |
| `evaluate-experiments` | Error analysis across a set of trained models. |
| `repeat-experiments` | Loop multiple causal-discovery + BN trainings across config variations. |
| `iterate-structure-learning` | Sweep structure-learning hyperparameters across a grid. |

### Common flags

Entry points are wired through [Fire](https://github.com/google/python-fire), so every function argument is also a CLI flag.

| Flag | Default | Description |
|---|---|---|
| `--config_file PATH` | *required* | YAML config for the run. Schemas differ per command — see the Configuration sections below. |
| `--debug` | `False` | Log the full parsed config on startup. Defaults to a compact summary (`event=… data=… output=…`). Available on `infer-bn` and `infer-dowhy`. |

### Examples

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

The wheel ships with a `py.typed` marker, so `mypy` and editors pick up full type information from `causal_canvas` without configuration.

A minimal end-to-end sketch, mirroring what `infer-bn` does internally:

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

See `infer_bayesian_network.py` for the full reference driver (Bayesian-network fit, output plotting, conditional-dependency extraction).

## Security

Causal Canvas handles two asset types with inherent risk: pickled model files (Bayesian networks, discretisers) and subprocess invocations (when `repeat-experiments` or `iterate-structure-learning` spawn further training jobs). This section documents the guarantees and the limits.

### Loading saved models safely

Bayesian-network and discretiser checkpoints are persisted with `pickle`. Pickle deserialisation **executes arbitrary Python** — the format is not safe for untrusted input.

Causal Canvas routes every load through `causal_canvas.utils.safe_pickle_load`, which:

- Resolves the file path and refuses to load anything outside a *trusted root* (`Path.cwd()` by default). Paths like `../../etc/bad.pkl` are rejected with a `ValueError`.
- Logs a SHA-256 prefix of the loaded bytes before deserialising so tampering with a file that was produced in a prior run leaves a trail in the logs.

This limits the blast radius of a misconfigured config file that points `model_path` at a surprise location. It does **not** make pickle itself safe — only load checkpoints produced by trusted runs. If you receive a `.pkl` from a third party, don't feed it to `infer-dowhy` without inspecting its contents first.

### No shell injection in experiment runners

`repeat-experiments` and `iterate-structure-learning` spawn child processes to run individual trainings. Both use `subprocess.check_call([sys.executable, script, ...])` with argv lists — never `shell=True`. Config values passed as command-line arguments are not interpolated into a shell string, so a malicious config file can't smuggle shell commands through.

### Supply-chain auditing

Known CVEs in the dependency tree are scanned with [pip-audit](https://github.com/pypa/pip-audit):

```bash
tox -e audit
```

Findings are **non-blocking** today: the `causalnex` pin chain (`torch==2.2.1`, `pyarrow==15.0.0`) was removed in 0.2.0, so remaining advisories are in dev tooling. See [`docs/SECURITY_AUDIT.md`](https://github.com/PlaytikaOSS/causal-canvas/blob/main/docs/SECURITY_AUDIT.md) for the current triage, the upgrade blockers, and the process for new findings.

### Reporting a vulnerability

Please contact the maintainers privately rather than opening a public issue. See [`CONTRIBUTING.md`](https://github.com/PlaytikaOSS/causal-canvas/blob/main/CONTRIBUTING.md) for reporting guidance.

## Causal Discovery and Bayesian Network Fitting

### Configuration

The full configuration schema for `infer-bn` (data and I/O, causal-discovery hyperparameters, non-linear NOTEARS arguments, discretisation, Bayesian-network inference, visualisation) is documented in [docs/reference/config.md](https://github.com/PlaytikaOSS/causal-canvas/blob/main/docs/reference/config.md), grouped by pipeline stage with one table per group.

A bundled example lives at [`examples/students/config.yaml`](https://github.com/PlaytikaOSS/causal-canvas/blob/main/examples/students/config.yaml). Each example folder under `examples/` ships a `README.md` describing the scenario and the commands to reproduce its outputs.

### Experiment results

```text
└── experiment                                                <- Directory for the experiment results.
    ├── causal_discovery                                      <- Directory for the causal discovery.
    │   ├── 01_fully_connected_graph.html                     <- HTML file with interactive fully connected graph with or without lasso.
    │   ├── 02_largest_subgraph.html                          <- HTML file with interactive largest subgraph.
    │   ├── 03_threshold_graph.html                           <- HTML file with pruned graph.
    │   ├── 04_threshold_DAG.html                             <- HTML file with resulting DAG after pruning.
    │   ├── 05_DAG_subgraph.html                              <- HTML file with largest subgraph of the pruned DAG.
    │   ├── label_encoder.pkl                                 <- Label encoder for categorical variables.
    │   ├── DAG_for_inference.dot                             <- Structured DAG networkx class to use for inference.
    │   ├── scaler.pkl                                        <- Scaler for numerical variables.
    │   ├── all_weights_mlp.csv                               <- If MLP is chosen, resulting weight matrix.
    │   ├── threshold_weights_mlp.csv                         <- If MLP is chosen, resulting weight matrix after hard threshold.
    │   ├── all_std_error_weights_mlp.csv                     <- If MLP is chosen, resulting s.e. of weight matrix
    │   ├── threshold_std_error_weights_mlp.csv               <- If MLP is chosen, resulting s.e. of weight matrix after hard threshold.
    │   ├── all_weights_heatmap.png                           <- If MLP is chosen, heatmap weight matrix and target weights after hard threshold.
    │   ├── all_weights_w_threshold_heatmap.png               <- If MLP is chosen, heatmap of weight matrix and target weights after hard threshold.
    │   ├── score_edges_full_structure.yaml                   <- Weights of structural causal relationships for the full structure with or without lasso.
    │   └── score_edges_sparser_structure.yaml                <- Weights of structural causal relationships for the sparser structure with or without lasso.
    ├── bayesian_network                                      <- Directory for the bayesian_network probabilistic modelling.
    │   ├── dependency_estimates                              <- Directory with node specific CPD wrt to its parents
    │   │   ├── node_name.csv                                 <- CSV with node specific CPD wrt to its parents.
    │   │        ...
    │   ├── model_fit                                     <- Directory bayesian_network model class and discretiser.
    │   │   ├── bayesian_network.pkl                      <- Bayesian Network model class
    │   │   ├── quantile_splits.json                      <- Dictionary with numerical splits by feature as decided by the discretiser
    │   │   ├── discretiser.pkl                           <- Discretiser model class if `tree` or `mdlp` is used
    │   │   └── condensed_splits.json                     <- Dictionary with numerical splits by feature as decided by post-processing with condensing categories
    └── config.yaml                                       <- Configuration file.

```

## Error analysis

In this phase, we perform an evaluation of various model combinations following Bayesian network inference.
Initially, we generate bootstrap intervals for F1-score, precision, and recall using a default threshold
of 50% for the optimisation dataset. Subsequently, we determine an optimized threshold based on the optimisation
dataset's results. Then, we recompute these performance scores for the test dataset.
Overall, this phase facilitates a clear organization of error analysis results
making it easy to access and interpret the performance of different BN models and inference methods at each step of the analysis process.
It allows for a comprehensive evaluation of model performance and the impact of threshold optimisation on the test set predictions.

This phase outputs four different folders:

1. *cpd_plots Directory*: This directory contains plots depicting the variation the Conditional Probability Tables (CPTs)
for different models. It provides insight into how variable the CPTs are in each segment.

2. *optim_set_predictions Directory*: Within this directory, you have subdirectories for different inference methods.
Each subdirectory contains predictions made for the optimisation set before threshold optimisation.
The predictions are stored in CSV files. Additionally, there's a CSV file and
a plot with bootstrap estimated scores for all models in this directory, providing an overview of model performance.

3. *test_set_predictions Directory*: Similar to the `optim_set_predictions` directory,
this one also contains subdirectories for different inference methods.
Inside each subdirectory, you'll find predictions made for the test set using
the optimized threshold obtained from the optimization set.
These predictions are also stored in CSV files. Like before, there's a CSV file and a plot with bootstrap estimated
scores for all models, allowing you to assess model performance on the test set.

The full schema for `evaluate-experiments` is documented in [docs/reference/config.md](https://github.com/PlaytikaOSS/causal-canvas/blob/main/docs/reference/config.md#evaluate-experiments-configuration). A bundled example lives at [`examples/students/config_evaluation.yaml`](https://github.com/PlaytikaOSS/causal-canvas/blob/main/examples/students/config_evaluation.yaml).

### Error Analysis Usage

Run the `evaluate-experiments` console script:

```bash
poetry run evaluate-experiments --config_file examples/students/config_evaluation.yaml
```

Error analysis results

```text
└── error_analysis                                            <- Directory for the error analysis results.
    ├── cpd_plots                                             <- Directory with plots comprising how variable the CPTs are in each segment.
    │   ├── cpds_mle_tree_2.png                               <- Plot of model specific CPTs.
    │   ├── cpds_mle_tree_4.png
    │   ├──  ...
    │   └── cpds_mle_tree_4.png
    ├── optim_set_predictions                                    <- Directory with all predictions that were conducted for the optim set before threshold optimisation with threshold = 50%.
    │   ├── inference_method_1                                   <- Directory with all predictions for a specific inference method that was used.
    │   │   ├── predictions_with_discretiser_tree_2.csv          <- CSV with predictions for the model trained with inference_method_1, discretiser and argument.
    │   │   ├── predictions_with_discretiser_mdlp_5.csv
    │   │        ...
    │   ├── inference_method_2
    │   ...
    │   ├── bootstrap_scores.csv                                <- CSV with bootstrap estimated scores for all models.
    │   ├── bootstrap_scores.png                                <- Plot with bootstrap estimated scores for all models.
    │   └── prediction_times.png                                <- Plot with time it took for each prediction.
    ├── test_set_predictions                                    <- Directory with all predictions that were conducted for the test set with using the optimised threshold taken from optim_set.
    │   ├── inference_method_1                                  <- Directory with all predictions for a specific inference method that was used.
    │   │   ├── predictions_with_discretiser_argument_1.csv     <- CSV with predictions for the model trained with inference_method_1, discretiser and argument.
    │   │   ├── predictions_with_discretiser_argument_2.csv
    │   │        ...
    │   ├── inference_method_2
    │   ...
    │   ├── bootstrap_scores.csv                                <- CSV with bootstrap estimated scores for all models.
    │   ├── bootstrap_scores.png                                <- Plot with bootstrap estimated scores for all models.
    │   └── prediction_times.png                                <- Plot with time it took for each prediction.
    └── config.yaml                                             <- Configuration file used for the error analysis.

```

## Do-why inference tool

The `infer-dowhy` console script performs do-calculus Bayesian-network inference against a fitted model: marginals, interventions, counterfactuals, and ATE. The full schema for `config_inference.yaml` is documented in [docs/reference/config.md](https://github.com/PlaytikaOSS/causal-canvas/blob/main/docs/reference/config.md#infer-dowhy-configuration). A bundled example lives at [`examples/students/config_inference.yaml`](https://github.com/PlaytikaOSS/causal-canvas/blob/main/examples/students/config_inference.yaml).

For the conceptual difference between observational, interventional, and counterfactual queries, see [docs/concepts/interventions-counterfactuals.md](https://github.com/PlaytikaOSS/causal-canvas/blob/main/docs/concepts/interventions-counterfactuals.md).

```bash
poetry run infer-dowhy --config_file examples/students/config_inference.yaml
```

> :warning: **Security**: `config_inference.yaml`'s `model_path` is loaded via `safe_pickle_load`, which restricts the load path to a trusted root and logs a SHA-256 prefix of the bytes before deserialising. See the [Security section](#security) above.

## Author

Written by Sofia Maria Karadimitriou (<sofia.m.karadimitriou@gmail.com>)

## Citations

@software{Beaumont_CausalNex_2021,
author = {Beaumont, Paul and Horsburgh, Ben and Pilgerstorfer, Philip and Droth, Angel and Oentaryo, Richard and Ler, Steven and Nguyen, Hiep and Ferreira, Gabriel Azevedo and Patel, Zain and Leong, Wesley},
month = oct,
title = {{CausalNex}},
url = {<https://github.com/quantumblacklabs/causalnex}>,
year = {2021}
}
