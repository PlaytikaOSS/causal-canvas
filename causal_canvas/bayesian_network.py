"""
Discrete Bayesian network backed by ``pgmpy``.

Replaces ``causalnex.network.BayesianNetwork`` (ADR-0001). The public surface
that causal-canvas consumes is kept: :meth:`fit_node_states_and_cpds`,
:attr:`cpds` (one ``DataFrame`` per node, rows = node states, columns = a
``MultiIndex`` over parent states), :meth:`predict` and
:meth:`predict_probability`. The fitted ``pgmpy.models.DiscreteBayesianNetwork``
is exposed as :attr:`model` for anything richer (exact inference, ``do()``).
"""

from __future__ import annotations

from collections.abc import Hashable

import networkx as nx
import pandas as pd
from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.parameter_estimator import DiscreteBayesianEstimator, DiscreteMLE
from pgmpy.parameter_estimator.base import DiscreteParameterEstimator

VALID_METHODS = ("MaximumLikelihoodEstimator", "BayesianEstimator")
VALID_BAYES_PRIORS = ("BDeu", "K2")


class BayesianNetwork:
    """A discrete Bayesian network over a fixed DAG, fitted with pgmpy."""

    def __init__(self, structure: nx.DiGraph):
        """
        Parameters
        ----------
        structure: nx.DiGraph
            The DAG (a ``StructureModel`` or any ``networkx.DiGraph``). Edge and
            node attributes are ignored; only the topology is used.

        Raises
        ------
        ValueError
            If ``structure`` contains a cycle.
        """
        if not nx.is_directed_acyclic_graph(structure):
            raise ValueError("The structure of a Bayesian network must be a DAG")
        self._structure = structure
        self._model = DiscreteBayesianNetwork()
        self._model.add_nodes_from(structure.nodes)
        self._model.add_edges_from(structure.edges)
        self._node_states: dict[str, list[Hashable]] = {}

    # ------------------------------------------------------------------ views
    @property
    def structure(self) -> nx.DiGraph:
        return self._structure

    @property
    def model(self) -> DiscreteBayesianNetwork:
        """The underlying pgmpy model (CPDs present once fitted)."""
        return self._model

    @property
    def nodes(self) -> list[str]:
        return sorted(self._model.nodes())

    @property
    def edges(self) -> list[tuple[str, str]]:
        return sorted(self._model.edges())

    @property
    def node_states(self) -> dict[str, list[Hashable]]:
        """Sorted states of every node, learned by :meth:`fit_node_states`."""
        return self._node_states

    @property
    def cpds(self) -> dict[str, pd.DataFrame]:
        """
        Conditional probability table of each node.

        Rows are the node's states; columns are a ``MultiIndex`` over every
        combination of parent states (a single ``""`` column for root nodes).
        """
        return {cpd.variable: self._cpd_to_frame(cpd) for cpd in self._model.get_cpds()}

    def _cpd_to_frame(self, cpd) -> pd.DataFrame:
        node, parents = cpd.variable, list(cpd.variables[1:])
        node_states = list(cpd.state_names[node])
        if parents:
            columns: pd.Index = pd.MultiIndex.from_product(
                [list(cpd.state_names[p]) for p in parents], names=parents
            )
        else:
            columns = pd.Index([""])
        frame = pd.DataFrame(
            cpd.get_values().reshape(len(node_states), len(columns)),
            index=pd.Index(node_states, name=node),
            columns=columns,
        )
        return frame

    # ---------------------------------------------------------------- fitting
    def fit_node_states(self, df: pd.DataFrame) -> BayesianNetwork:
        """Learn the states of every node from ``df`` (must contain every state)."""
        missing = [n for n in self._model.nodes() if n not in df.columns]
        if missing:
            raise ValueError(f"Data is missing columns for nodes: {missing}")
        if df[list(self._model.nodes())].isna().any().any():
            raise ValueError("Data contains missing values; a Bayesian network needs complete data")
        # .tolist() yields native Python scalars: with numpy 2 an np.int64 state would
        # leak its repr ("np.int64(0)") into tuple labels and CSV indexes.
        self._node_states = {n: sorted(df[n].unique().tolist()) for n in self._model.nodes()}
        return self

    def fit_cpds(
        self,
        data: pd.DataFrame,
        method: str = "MaximumLikelihoodEstimator",
        bayes_prior: str | None = None,
        equivalent_sample_size: int | None = None,
    ) -> BayesianNetwork:
        """
        Learn all conditional probability distributions.

        Parameters
        ----------
        method: str
            ``MaximumLikelihoodEstimator`` or ``BayesianEstimator``.
        bayes_prior: str
            ``K2`` (all pseudo counts 1) or ``BDeu`` (uniform pseudo counts summing
            to ``equivalent_sample_size``); only for ``BayesianEstimator``.
        equivalent_sample_size: int
            Prior strength for ``BDeu``.
        """
        if method not in VALID_METHODS:
            raise ValueError(f"unrecognised method, please use one of {list(VALID_METHODS)}")
        if method == "BayesianEstimator" and bayes_prior not in VALID_BAYES_PRIORS:
            raise ValueError(
                f"unrecognised bayes_prior, please use one of {list(VALID_BAYES_PRIORS)}"
            )
        if not self._node_states:
            raise ValueError("Node states are not fitted; call fit_node_states first")

        # Drop stale CPDs so refitting never mixes old and new tables.
        stale = self._model.get_cpds()
        if stale:
            self._model.remove_cpds(*stale)
        frame = data[list(self._model.nodes())]
        if method == "MaximumLikelihoodEstimator":
            estimator: DiscreteParameterEstimator = DiscreteMLE(state_names=self._node_states)
        else:
            estimator = DiscreteBayesianEstimator(
                state_names=self._node_states,
                prior_type=str(bayes_prior),
                # pgmpy's default (5) applies when the caller leaves it unset.
                **(
                    {"equivalent_sample_size": equivalent_sample_size}
                    if equivalent_sample_size is not None
                    else {}
                ),
            )
        self._model.fit(frame, estimator=estimator)
        return self

    def fit_node_states_and_cpds(
        self,
        data: pd.DataFrame,
        method: str = "MaximumLikelihoodEstimator",
        bayes_prior: str | None = None,
        equivalent_sample_size: int | None = None,
    ) -> BayesianNetwork:
        """Call :meth:`fit_node_states` then :meth:`fit_cpds`."""
        return self.fit_node_states(data).fit_cpds(
            data, method, bayes_prior, equivalent_sample_size
        )

    # ------------------------------------------------------------- prediction
    def predict(self, data: pd.DataFrame, node: str) -> pd.DataFrame:
        """
        Most probable state of ``node`` given the columns of ``data``.

        Conditioning follows :meth:`predict_probability`.

        Returns a frame with a single ``{node}_prediction`` column.
        """
        probabilities = self._posterior(data, node)
        prediction = probabilities.idxmax(axis=1)
        return prediction.to_frame(f"{node}_prediction")

    def predict_probability(self, data: pd.DataFrame, node: str) -> pd.DataFrame:
        """
        Probability of every state of ``node`` given the columns of ``data``.

        What is conditioned on depends on which columns are present (same
        semantics as causalnex):

        * **all parents of ``node`` present** -- ``P(node | parents)``, read
          straight from the CPT. Any other columns (children, siblings, ...)
          are ignored on purpose: this is fast and is what error analysis
          (``evaluate_experiments``) relies on.
        * **some parent missing** -- exact ``P(node | every observed column)``
          by variable elimination, which does use non-parent evidence.

        So adding a missing parent to ``data`` can change the answer; it is not
        a refinement of the same posterior. For full-evidence posteriors always,
        use :class:`causal_canvas.inference_engine.InferenceEngine`.

        Returns one ``{node}_{state}`` column per state.
        """
        probabilities = self._posterior(data, node)
        probabilities.columns = [f"{node}_{state}" for state in probabilities.columns]
        return probabilities

    def _posterior(self, data: pd.DataFrame, node: str) -> pd.DataFrame:
        """P(node | observed columns of data), one row per row of ``data``, one column per state."""
        if node not in self._model.nodes():
            raise ValueError(f"{node!r} is not a node of the network")
        states = self._node_states[node]
        parents = list(self._model.get_cpds(node).variables[1:])
        if all(p in data.columns for p in parents):
            return self._posterior_from_parents(data, node, parents, states)
        return self._posterior_by_inference(data, node, states)

    def _posterior_from_parents(
        self, data: pd.DataFrame, node: str, parents: list[str], states: list[Hashable]
    ) -> pd.DataFrame:
        """Fast path: every parent is observed, so read the CPT directly."""
        table = self.cpds[node]
        if not parents:
            rows = pd.DataFrame({state: table.at[state, ""] for state in states}, index=data.index)
            return rows
        lookup = table.T  # index: parent-state combinations; columns: node states
        keys = pd.MultiIndex.from_frame(data[parents].reset_index(drop=True))
        probabilities = lookup.reindex(keys)
        if probabilities.isna().any().any():
            bad = data[parents].reset_index(drop=True)[probabilities.isna().any(axis=1).values]
            raise ValueError(
                f"Parent states of {node!r} not seen when fitting: "
                f"{bad.drop_duplicates().to_dict('records')[:5]}"
            )
        probabilities.index = data.index
        return probabilities[states]

    def _posterior_by_inference(
        self, data: pd.DataFrame, node: str, states: list[Hashable]
    ) -> pd.DataFrame:
        """Slow path: some parents are missing, so marginalise with exact inference."""
        evidence_cols = [c for c in data.columns if c in self._model.nodes() and c != node]
        engine = VariableElimination(self._model)
        cache: dict[tuple, list[float]] = {}
        out = []
        for key in data[evidence_cols].itertuples(index=False, name=None):
            if key not in cache:
                factor = engine.query(
                    [node],
                    evidence=dict(zip(evidence_cols, key)),
                    show_progress=False,
                )
                cache[key] = [float(factor.get_value(**{node: state})) for state in states]
            out.append(cache[key])
        return pd.DataFrame(out, index=data.index, columns=states)
