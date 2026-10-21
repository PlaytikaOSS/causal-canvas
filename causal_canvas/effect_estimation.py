"""
Causal-effect identification, estimation and refutation with DoWhy.

Complements the Bayesian-network ``do()`` queries in
:mod:`causal_canvas.inference_engine` (ADR-0001): where the BN answers "how does
the *distribution* of the target move under an intervention" on discretised
data, this module answers "what is the average effect of the treatment" on the
original data, with back-door / front-door / IV identification from the learned
DAG and optional refutation tests.

DoWhy is an optional dependency (``pip install causal-canvas[dowhy]``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx
import pandas as pd


@dataclass
class EffectEstimate:
    """Result of :func:`estimate_effect`."""

    treatment: str
    outcome: str
    value: float
    estimand: str
    method: str
    refutations: dict[str, str] = field(default_factory=dict)


def estimate_effect(
    data: pd.DataFrame,
    graph: nx.DiGraph,
    treatment: str,
    outcome: str,
    method_name: str = "backdoor.linear_regression",
    refuters: list[str] | None = None,
) -> EffectEstimate:
    """
    Estimate the average causal effect of ``treatment`` on ``outcome``.

    Parameters
    ----------
    data: pd.DataFrame
        Original (non-discretised) data with one column per node of ``graph``.
    graph: nx.DiGraph
        The causal DAG, e.g. ``StructureLearner.structural_model``.
    treatment, outcome: str
        Node names.
    method_name: str
        DoWhy estimation method (default: back-door adjustment by linear regression).
    refuters: list of str, optional
        DoWhy refuter names, e.g. ``["random_common_cause", "placebo_treatment_refuter"]``.

    Raises
    ------
    ImportError
        If DoWhy is not installed.
    ValueError
        If a node of the graph is missing from ``data`` or the effect is not identifiable.
    """
    try:
        from dowhy import CausalModel
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "estimate_effect needs DoWhy: pip install 'causal-canvas[dowhy]'"
        ) from exc

    missing = [n for n in (treatment, outcome, *graph.nodes) if n not in data.columns]
    if missing:
        raise ValueError(f"Columns missing from data: {sorted(set(missing))}")

    dag = nx.DiGraph(graph.edges)
    dag.add_nodes_from(graph.nodes)
    model = CausalModel(data=data, treatment=treatment, outcome=outcome, graph=dag)
    estimand = model.identify_effect(proceed_when_unidentifiable=False)
    estimate = model.estimate_effect(estimand, method_name=method_name)

    refutations = {
        name: str(model.refute_estimate(estimand, estimate, method_name=name))
        for name in (refuters or [])
    }
    return EffectEstimate(
        treatment=treatment,
        outcome=outcome,
        value=float(estimate.value),
        estimand=str(estimand.estimands.get("backdoor", estimand.estimands)),
        method=method_name,
        refutations=refutations,
    )
