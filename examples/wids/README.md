# WIDS — Synthetic Churn

End-to-end example built for the WIDS workshop. The dataset is synthetic
churn data: a binary `churn` outcome driven by a simulated mix of
engagement, tech, and payment features (logins, game activity, special-
event notifications, crashes, days between login / payment).

## Configs

| File | Purpose | Console script |
|---|---|---|
| `config_simulation.yaml` | Causal discovery + Bayesian-network fit on the synthetic churn dataset. | `infer-bn` |
| `config_inference.yaml` | Do-calculus inference (marginals, interventions, counterfactuals) on the fitted BN. | `infer-dowhy` |

## Run

```bash
poetry run infer-bn --config_file examples/wids/config_simulation.yaml
poetry run infer-dowhy --config_file examples/wids/config_inference.yaml
```

Outputs land at the `output_path` set in each config (`out/` by default).

## Synthetic data generation

The synthetic-data generator that produces `data/synthetic_churn_data6.parquet`
lives at `notebooks/data_simulation.ipynb`. It is being rewritten in PR-05.
