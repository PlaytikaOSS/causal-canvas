# API reference

Auto-generated reference for the curated public surface of the `causal_canvas` package. Each page below is rendered by [mkdocstrings](https://mkdocstrings.github.io/) directly from the module's NumPy-style docstrings, so this stays in sync with the code.

## Public classes

- [StructureLearner](structure_learner.md) — causal-discovery driver (linear and non-linear NOTEARS).
- [BayesianNetworkEstimator](bayesian_network_estimator.md) — discretisation + Bayesian-network fit + persistence.
- [Preprocessor](preprocessor.md) — validation split, categorical encoding, sampling.

## Supporting modules

- [Non-linear NOTEARS](nonlinear_notears.md) — high-level wrapper around the MLP solver.
- [Inference utils](inference_utils.md) — do-calculus helpers consumed by `infer-dowhy`.
- [Evaluation utils](evaluation_utils.md) — bootstrap scoring across model combinations.
- [Config schema](script_config.md) — pydantic v2 `BaseModel` for the `infer-bn` config.

## Importing

The four classes most consumers need are re-exported from the top level:

```python
from causal_canvas import (
    BayesianNetworkEstimator,
    InferenceMethod,
    Preprocessor,
    StructureLearner,
)
```
