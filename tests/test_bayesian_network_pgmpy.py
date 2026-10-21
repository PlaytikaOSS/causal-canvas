"""Numerical-parity gate for the pgmpy-backed BN layer (ADR-0001 §6).

The reference network and the three reference queries are specified in the ADR.
Exact discrete inference is deterministic, so every query must agree with an
independent brute-force enumeration to 1e-6.
"""

import itertools

import networkx as nx
import pandas as pd
import pytest
from pgmpy.factors.discrete import TabularCPD

from causal_canvas.bayesian_network import BayesianNetwork
from causal_canvas.inference_engine import InferenceEngine

EDGES = [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D"), ("D", "E")]
P_A1 = 0.4
P_B1 = {0: 0.2, 1: 0.7}
P_C1 = {0: 0.3, 1: 0.8}
P_D1 = {(0, 0): 0.05, (0, 1): 0.4, (1, 0): 0.5, (1, 1): 0.95}
P_E1 = {0: 0.1, 1: 0.85}


def _bern(p1, v):
    return p1 if v == 1 else 1 - p1


def _joint(a, b, c, d, e):
    return (
        _bern(P_A1, a)
        * _bern(P_B1[a], b)
        * _bern(P_C1[a], c)
        * _bern(P_D1[(b, c)], d)
        * _bern(P_E1[d], e)
    )


def _enumerate(pred, joint=_joint):
    return sum(joint(*s) for s in itertools.product((0, 1), repeat=5) if pred(*s))


@pytest.fixture
def toy_bn() -> BayesianNetwork:
    bn = BayesianNetwork(nx.DiGraph(EDGES))
    bn._node_states = {n: [0, 1] for n in "ABCDE"}
    states = {n: [0, 1] for n in "ABCDE"}

    def cpd(node, p_row, evidence=()):
        # pgmpy layout: rows = node states, columns = parent-state combinations
        # (last evidence variable varies fastest).
        return TabularCPD(
            node,
            2,
            [[1 - p for p in p_row], list(p_row)],
            evidence=list(evidence),
            evidence_card=[2] * len(evidence),
            state_names={n: states[n] for n in (node, *evidence)},
        )

    bn.model.add_cpds(
        cpd("A", [P_A1]),
        cpd("B", [P_B1[0], P_B1[1]], ["A"]),
        cpd("C", [P_C1[0], P_C1[1]], ["A"]),
        cpd("D", [P_D1[(0, 0)], P_D1[(0, 1)], P_D1[(1, 0)], P_D1[(1, 1)]], ["B", "C"]),
        cpd("E", [P_E1[0], P_E1[1]], ["D"]),
    )
    bn.model.check_model()
    return bn


def test_marginal(toy_bn):
    p = InferenceEngine(toy_bn).query(nodes=["D"])["D"][1]
    assert p == pytest.approx(_enumerate(lambda a, b, c, d, e: d == 1), abs=1e-6)


def test_conditional(toy_bn):
    p = InferenceEngine(toy_bn).query({"D": 1}, nodes=["B"])["B"][1]
    expected = _enumerate(lambda a, b, c, d, e: b == 1 and d == 1) / _enumerate(
        lambda a, b, c, d, e: d == 1
    )
    assert p == pytest.approx(expected, abs=1e-6)


def test_interventional(toy_bn):
    engine = InferenceEngine(toy_bn)
    engine.do_intervention("B", 1)
    p = engine.query(nodes=["D"])["D"][1]

    def truncated(a, b, c, d, e):  # truncated factorisation: drop P(B | A)
        return _bern(P_A1, a) * _bern(P_C1[a], c) * _bern(P_D1[(b, c)], d) * _bern(P_E1[d], e)

    expected = _enumerate(lambda a, b, c, d, e: b == 1 and d == 1, joint=truncated)
    expected /= _enumerate(lambda a, b, c, d, e: b == 1, joint=truncated)
    assert p == pytest.approx(expected, abs=1e-6)
    # Intervention differs from conditioning: B's descendants move, its parent does not.
    assert engine.query(nodes=["A"])["A"][1] == pytest.approx(P_A1, abs=1e-6)

    engine.reset_do("B")
    assert engine.query(nodes=["D"])["D"][1] == pytest.approx(
        _enumerate(lambda a, b, c, d, e: d == 1), abs=1e-6
    )


def test_soft_intervention_and_validation(toy_bn):
    engine = InferenceEngine(toy_bn)
    engine.do_intervention("B", {0: 0.5, 1: 0.5})
    assert engine.query(nodes=["B"])["B"] == {0: 0.5, 1: 0.5}
    with pytest.raises(ValueError):
        engine.do_intervention("B", {0: 0.5, 1: 0.6})
    with pytest.raises(ValueError):
        engine.do_intervention("B", {0: 0.5, 7: 0.5})


def test_cpds_layout_and_predict(toy_bn):
    cpd_d = toy_bn.cpds["D"]
    assert list(cpd_d.index) == [0, 1]
    assert cpd_d.columns.names == ["B", "C"]
    assert cpd_d[(1, 1)].loc[1] == pytest.approx(0.95)
    assert toy_bn.cpds["A"].at[1, ""] == pytest.approx(0.4)

    data = pd.DataFrame({"B": [0, 1, 1], "C": [0, 0, 1]})
    proba = toy_bn.predict_probability(data, "D")
    assert proba["D_1"].tolist() == pytest.approx([0.05, 0.5, 0.95])
    assert toy_bn.predict(data, "D")["D_prediction"].tolist() == [0, 0, 1]

    # Parents missing -> exact inference over the rest of the network.
    partial = pd.DataFrame({"A": [0, 1]})
    p_inc = toy_bn.predict_probability(partial, "E")
    for a, p in zip((0, 1), p_inc["E_1"]):
        expected = _enumerate(lambda aa, b, c, d, e, a=a: aa == a and e == 1) / _enumerate(
            lambda aa, b, c, d, e, a=a: aa == a
        )
        assert p == pytest.approx(expected, abs=1e-6)


def test_fit_recovers_cpds():
    data = pd.DataFrame({"x": [0, 0, 1, 1] * 25, "y": [0, 1, 1, 1] * 25})
    bn = BayesianNetwork(nx.DiGraph([("x", "y")]))
    bn.fit_node_states_and_cpds(data, method="MaximumLikelihoodEstimator")
    assert bn.cpds["y"][(0,)].loc[1] == pytest.approx(0.5)
    assert bn.cpds["y"][(1,)].loc[1] == pytest.approx(1.0)
    bn.fit_node_states_and_cpds(data, method="BayesianEstimator", bayes_prior="K2")
    assert bn.cpds["y"][(1,)].loc[1] < 1.0
    with pytest.raises(ValueError):
        bn.fit_cpds(data, method="BayesianEstimator", bayes_prior="nope")


def test_predict_with_all_parents_ignores_non_parent_columns(toy_bn):
    """Documented semantics: complete parents -> P(node | parents), extra evidence ignored."""
    parents_only = pd.DataFrame({"B": [1], "C": [0]})
    with_child = parents_only.assign(E=[1])  # a descendant of D carries information about D
    a = toy_bn.predict_probability(parents_only, "D")["D_1"].iloc[0]
    b = toy_bn.predict_probability(with_child, "D")["D_1"].iloc[0]
    assert a == pytest.approx(b) == pytest.approx(0.5)
    # Whereas the full-evidence posterior does use E.
    full = InferenceEngine(toy_bn).query({"B": 1, "C": 0, "E": 1}, nodes=["D"])["D"][1]
    assert full != pytest.approx(0.5, abs=1e-3)


def test_node_states_are_native_python_scalars():
    """numpy-2 scalars would leak "np.int64(0)" into labels such as ('G2', 0)."""
    data = pd.DataFrame({"x": [0, 1, 1, 0], "y": [1, 1, 0, 0]})
    bn = BayesianNetwork(nx.DiGraph([("x", "y")])).fit_node_states_and_cpds(data)
    assert all(type(s) is int for states in bn.node_states.values() for s in states)
    marginals = InferenceEngine(bn).query()
    assert all(type(s) is int for dist in marginals.values() for s in dist)
