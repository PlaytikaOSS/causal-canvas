# Students — UCI Portuguese Grades

End-to-end example using the UCI [Student Performance](https://archive.ics.uci.edu/dataset/320/student+performance)
dataset (Portuguese-language course). Features cover demographics, family
background, lifestyle, and prior-period grades; the target `G3` is the
final grade for the term.

The three configs in this folder cover the three core workflows: causal
discovery + Bayesian-network fit, error analysis across model variants,
and do-calculus inference on a fitted network.

## Configs

| File | Purpose | Console script |
|---|---|---|
| `config.yaml` | Causal discovery + Bayesian-network fit. | `infer-bn` |
| `config_evaluation.yaml` | Error analysis across multiple inference / discretiser combinations. | `evaluate-experiments` |
| `config_inference.yaml` | Do-calculus inference (marginals, interventions, counterfactuals) on the fitted BN. | `infer-dowhy` |

## Run

```bash
poetry run infer-bn --config_file examples/students/config.yaml
poetry run evaluate-experiments --config_file examples/students/config_evaluation.yaml
poetry run infer-dowhy --config_file examples/students/config_inference.yaml
```

## Outputs

Each config writes to its own `output_path`:

- `config.yaml` → `dummy_example_results/students_portuguese_grades/non_linear_causal_discovery/k2_prior/tree_4/`
- `config_evaluation.yaml` → `dummy_example_results/students_portuguese_grades/non_linear_causal_discovery/evaluation/`
- `config_inference.yaml` → `dummy_example_results/students_portuguese_grades/non_linear_causal_discovery/interventions/k2_prior_tree_2/`

The repo-tracked reference outputs under `dummy_example_results/` were
removed to keep the working tree small. Re-running the pipeline
regenerates them locally; the artefacts are not version-controlled.
