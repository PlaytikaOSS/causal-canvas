from causal_canvas.bayesian_network_estimator import (
    BayesianNetworkEstimator,
    InferenceMethod,
)
from causal_canvas.vendored.structure_model import StructureModel


def _small_structural_model() -> StructureModel:
    sm = StructureModel()
    sm.add_weighted_edges_from(
        [("A", "D", 1.0), ("B", "D", 1.0), ("C", "D", 1.0), ("A", "event", 0.5)]
    )
    return sm


def test_bayesian_network_estimator_instantiates(synthetic_data):
    """Smoke: construct BayesianNetworkEstimator with a minimal config."""
    sm = _small_structural_model()

    bn = BayesianNetworkEstimator(
        structural_model=sm,
        train_set=synthetic_data[["A", "B", "C", "D", "event"]],
        numerical_features=["A", "B", "C", "D"],
        event_col="event",
        discretiser_method="simple",
        discretiser_argument=3,
        discretisation_cutoffs_target=[],
        inference_method=InferenceMethod(method="MaximumLikelihoodEstimator", bayes_prior="K2"),
        proportion_threshold=None,
        max_categories=None,
    )

    assert bn.event_col == "event"
    assert bn.discretiser_method == "simple"
    assert bn.model is None  # not yet fitted


def test_inference_method_validates_literals():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        InferenceMethod(method="NotARealMethod", bayes_prior="K2")

    with pytest.raises(ValidationError):
        InferenceMethod(method="MaximumLikelihoodEstimator", bayes_prior="NotAPrior")
