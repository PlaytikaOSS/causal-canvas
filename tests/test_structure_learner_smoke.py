import pytest

from causal_canvas.structure_learner import StructureLearner


def test_structure_learner_instantiates(synthetic_data):
    """Smoke: StructureLearner can be instantiated with a preprocessed-shape DataFrame."""
    X = synthetic_data[["A", "B", "C", "D", "event"]]

    sl = StructureLearner(
        X=X,
        event_col="event",
        connections_type="linear",
        non_linear_args={},
        max_iter=10,
        h_tol=1e-4,
        w_threshold=0.1,
        lasso_multiplier=0.1,
    )

    assert sl.event_col == "event"
    assert sl.connections_type == "linear"


@pytest.mark.slow
def test_structure_learner_linear_notears_runs(synthetic_data):
    """End-to-end linear NOTEARS on synthetic data. Slow because it trains."""
    X = synthetic_data[["A", "B", "C", "D", "event"]]

    sl = StructureLearner(
        X=X,
        event_col="event",
        connections_type="linear",
        non_linear_args={},
        max_iter=20,
        h_tol=1e-3,
        w_threshold=0.1,
        lasso_multiplier=0.1,
    )

    sl.learn_structure()

    assert sl.structural_model is not None
    assert len(list(sl.structural_model.nodes)) > 0
