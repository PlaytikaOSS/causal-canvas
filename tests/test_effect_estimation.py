import networkx as nx
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("dowhy")

from causal_canvas.effect_estimation import estimate_effect  # noqa: E402


def test_backdoor_effect_recovers_true_coefficient():
    rng = np.random.default_rng(0)
    n = 2000
    z = rng.normal(size=n)  # confounder
    t = 0.8 * z + rng.normal(size=n)
    y = 2.0 * t + 1.5 * z + rng.normal(size=n)
    data = pd.DataFrame({"z": z, "t": t, "y": y})
    graph = nx.DiGraph([("z", "t"), ("z", "y"), ("t", "y")])

    result = estimate_effect(data, graph, treatment="t", outcome="y")

    assert result.value == pytest.approx(2.0, abs=0.1)
    # The naive regression of y on t is confounded; adjustment must matter.
    assert abs(np.polyfit(t, y, 1)[0] - 2.0) > 0.3


def test_missing_columns_raise():
    graph = nx.DiGraph([("a", "b")])
    with pytest.raises(ValueError, match="missing"):
        estimate_effect(pd.DataFrame({"a": [1.0, 2.0]}), graph, "a", "b")
