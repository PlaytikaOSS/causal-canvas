"""Torch-compatibility guard for the NOTEARS-MLP we own (ADR-0001, risk 2).

causalnex is dormant, so a torch release that breaks ``LocallyConnected``, the
dual-ascent optimiser loop or one of our distribution types is ours to catch.
Run this against the latest torch before every bump (``pytest -m slow``).
"""

import networkx as nx
import numpy as np
import pandas as pd
import pytest
import torch

from causal_canvas.custom_distributions import dist_type_aliases
from causal_canvas.nonlinear_notears import from_pandas

pytestmark = pytest.mark.slow


def _chain(n=400, seed=0) -> pd.DataFrame:
    """x -> y -> z with non-linear links; positive-valued so every dist type is valid."""
    rng = np.random.default_rng(seed)
    x = rng.gamma(4.0, 1.0, n)
    y = np.sin(x) + 3.0 + 0.1 * rng.normal(size=n)
    z = y**2 / 4.0 + 0.1 * rng.normal(size=n)
    return pd.DataFrame({"x": x, "y": y, "z": z})


def _fit(df, schema=None, **kw):
    torch.manual_seed(0)
    return from_pandas(
        df,
        dist_type_schema=schema,
        hidden_layer_units=[4],
        lasso_beta=0.01,
        ridge_beta=0.01,
        use_bias=True,
        max_iter=15,
        h_tol=1e-6,
        w_threshold=0.0,
        use_gpu=False,
        verbose=False,
        **kw,
    )


def test_fit_returns_acyclic_graph_with_finite_weights():
    sm, model = _fit(_chain())
    # NOTEARS only drives h(W) below h_tol, so tiny back-edges survive an unthresholded fit;
    # the pipeline prunes them with threshold_till_dag().
    pruned = sm.copy()
    pruned.threshold_till_dag()
    assert nx.is_directed_acyclic_graph(pruned)
    assert np.isfinite(np.asarray(model.adj_mean_effect)).all()
    assert set(sm.nodes) == {"x", "y", "z"}


def test_chain_structure_is_connected():
    sm, _ = _fit(_chain())
    sm.threshold_till_dag()
    undirected = {frozenset(e) for e in sm.edges}
    # x-y carries the strongest signal; a torch regression usually shows as an empty graph.
    assert frozenset({"x", "y"}) in undirected or frozenset({"y", "z"}) in undirected


def test_fit_is_deterministic_for_a_fixed_seed():
    a, _ = _fit(_chain())
    b, _ = _fit(_chain())
    assert sorted(a.edges(data="weight")) == sorted(b.edges(data="weight"))


@pytest.mark.parametrize("alias", sorted(set(dist_type_aliases) - {"cat"}))
def test_every_distribution_type_trains(alias):
    df = _chain()
    if alias in ("bin",):
        df["z"] = (df["z"] > df["z"].median()).astype(float)
    elif alias in ("ord", "poiss"):
        df["z"] = np.round(df["z"]).astype(float)
    sm, model = _fit(df, schema={"x": "cont", "y": "cont", "z": alias})
    assert np.isfinite(np.asarray(model.adj_mean_effect)).all(), alias
    assert set(sm.nodes) >= {"x", "y"}
