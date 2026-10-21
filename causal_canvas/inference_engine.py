"""
Exact marginal queries and (soft) ``do()`` interventions on a fitted
:class:`causal_canvas.bayesian_network.BayesianNetwork`.

Replaces ``causalnex.inference.InferenceEngine`` (ADR-0001) with pgmpy's
``VariableElimination``. Behaviour kept from the old engine:

* ``query()`` returns ``{node: {state: probability}}`` for every node;
* ``do_intervention(node, state)`` accepts a state (point mass) or a
  ``{state: probability}`` map and persists until ``reset_do(node)``;
* an intervened node loses its incoming edges (graph surgery).

Unlike the old engine, ``query`` can be restricted to the nodes you need
(``nodes=[...]``); variable elimination then only touches the relevant part of
the network, which is where the large speed-ups come from.
"""

from __future__ import annotations

import math
from collections.abc import Hashable
from copy import deepcopy
from typing import Any, overload

from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

from causal_canvas.bayesian_network import BayesianNetwork

Marginals = dict[str, dict[Hashable, float]]


class InferenceEngine:
    """Exact inference over a fitted :class:`BayesianNetwork`."""

    def __init__(self, bn: BayesianNetwork):
        if not bn.model.get_cpds():
            raise ValueError(
                "Bayesian Network does not contain any CPDs. You should fit CPDs "
                "before doing inference (see `BayesianNetwork.fit_cpds`)."
            )
        self._bn = bn
        self._states = {n: list(s) for n, s in bn.node_states.items()}
        self._interventions: dict[str, dict[Hashable, float]] = {}
        self._engine: VariableElimination | None = None

    # ----------------------------------------------------------------- query
    @overload
    def query(
        self, observations: dict[str, Any] | None = ..., nodes: list[str] | None = ...
    ) -> Marginals:
        ...

    @overload
    def query(
        self, observations: list[dict[str, Any]], nodes: list[str] | None = ...
    ) -> list[Marginals]:
        ...

    def query(
        self,
        observations: dict[str, Any] | list[dict[str, Any]] | None = None,
        nodes: list[str] | None = None,
    ) -> Marginals | list[Marginals]:
        """
        Marginal distributions given one or more observations.

        Parameters
        ----------
        observations: dict or list of dict
            Observed states, e.g. ``{"node_a": 1}``. A list yields one result per entry.
        nodes: list of str
            Restrict the result to these nodes (default: every node).

        Returns
        -------
        ``{node: {state: probability}}``, or a list of those for a list of observations.
        Observed nodes are returned as a point mass on the observed state.
        """
        if observations is not None and not isinstance(observations, (dict, list)):
            raise TypeError("Expecting observations to be a dict, list or None")
        if isinstance(observations, list):
            return [self._single_query(obs, nodes) for obs in observations]
        return self._single_query(observations, nodes)

    def _single_query(self, observations: dict[str, Any] | None, nodes: list[str] | None):
        observations = dict(observations or {})
        unknown = [n for n in {*observations, *(nodes or [])} if n not in self._states]
        if unknown:
            raise ValueError(f"Unknown nodes: {sorted(unknown)}")
        for node, state in observations.items():
            if state not in self._states[node]:
                raise ValueError(f"State {state!r} is not a state of {node!r}")

        targets = list(nodes) if nodes is not None else list(self._states)
        engine = self._get_engine()
        results: Marginals = {}
        for node in targets:
            if node in observations:
                results[node] = {s: float(s == observations[node]) for s in self._states[node]}
                continue
            factor = engine.query([node], evidence=dict(observations), show_progress=False)
            results[node] = {s: float(factor.get_value(**{node: s})) for s in self._states[node]}
        return results

    # ----------------------------------------------------------- interventions
    def do_intervention(
        self,
        node: str,
        state: Hashable | dict[Hashable, float] | None = None,
    ):
        """
        Intervene on ``node``.

        ``state`` is either a single state (point mass) or a ``{state: probability}``
        map over all states of the node. The node's incoming edges are cut.
        """
        if node not in self._states:
            raise ValueError(f"Unknown node: {node!r}")
        if not isinstance(state, dict):
            if state not in self._states[node]:
                raise ValueError(f"State {state!r} is not a state of {node!r}")
            state = {s: float(s == state) for s in self._states[node]}

        if not math.isclose(sum(state.values()), 1.0):
            raise ValueError("The cpd for the provided observation must sum to 1")
        if max(state.values()) > 1.0 or min(state.values()) < 0:
            raise ValueError("The cpd for the provided observation must be between 0 and 1")
        if set(state) != set(self._states[node]):
            raise ValueError(
                f"The cpd states do not match expected states: "
                f"expected {set(self._states[node])}, found {set(state)}"
            )
        self._interventions[node] = dict(state)
        self._engine = None

    def reset_do(self, observation: str):
        """Remove any intervention applied to ``observation``."""
        self._interventions.pop(observation, None)
        self._engine = None

    # ---------------------------------------------------------------- internals
    def _get_engine(self) -> VariableElimination:
        if self._engine is None:
            self._engine = VariableElimination(self._mutilated_model())
        return self._engine

    def _mutilated_model(self):
        """Copy of the fitted model with intervened nodes cut off from their parents."""
        model = self._bn.model
        if not self._interventions:
            return model
        model = deepcopy(model)
        for node, distribution in self._interventions.items():
            for parent in list(model.get_parents(node)):
                model.remove_edge(parent, node)
            model.remove_cpds(node)
            states = self._states[node]
            model.add_cpds(
                TabularCPD(
                    node,
                    len(states),
                    [[distribution[s]] for s in states],
                    state_names={node: states},
                )
            )
        model.check_model()
        return model
