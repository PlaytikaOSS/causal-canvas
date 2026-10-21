"""Inference benchmark for the pgmpy-backed BN (ADR-0001 §9 Q2 / Q7).

Target from the ADR: queries that took ~7 min on causalnex should take <= 30 s
(stretch <= 5 s). Two modes:

* ``--model path/to/bayesian_network.pkl`` -- time a real fitted
  ``BayesianNetworkEstimator`` (the case that matters). The path must live under
  the current working directory (``safe_pickle_load`` trust model).
* ``--tables model_tables.json`` -- CPTs exported from a pre-0.2.0 pickle by
  ``benchmarks/export_legacy_model.py`` (no training data needed).
* default -- a synthetic random discrete BN (``--nodes``, ``--card``,
  ``--max-parents``, ``--rows``) so the harness runs anywhere.

Timed: full ``query()``, a targeted ``query(nodes=[target])``, a conditional
query, and ``do_intervention`` + targeted query. The report is JSON and records
the machine spec (CPU model, cores, RAM, Python and library versions) -- it
deliberately does NOT record hostnames or FQDNs.

    python benchmarks/bench_inference.py --nodes 40 --card 4 --max-parents 3
    python benchmarks/bench_inference.py --model out/bayesian_network/model_fit/bayesian_network.pkl
"""

import json
import os
import platform
import statistics
import time
from pathlib import Path
from typing import Optional

import fire
import networkx as nx
import numpy as np
import pandas as pd

from causal_canvas.bayesian_network import BayesianNetwork
from causal_canvas.inference_engine import InferenceEngine


def machine_spec() -> dict:
    import pgmpy

    spec = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "pgmpy": pgmpy.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    try:
        import torch

        spec["torch"] = torch.__version__
        spec["cuda"] = torch.cuda.is_available()
    except ImportError:
        pass
    try:  # RAM / CPU model are best-effort and Unix-only.
        spec["ram_gb"] = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1)
    except (ValueError, OSError, AttributeError):
        pass
    spec["cpu_model"] = platform.processor() or platform.machine()
    return spec


def random_network(nodes: int, card: int, max_parents: int, rows: int, seed: int):
    """Random DAG over ``nodes`` discrete variables, with data sampled from random CPTs."""
    rng = np.random.default_rng(seed)
    names = [f"v{i}" for i in range(nodes)]
    graph = nx.DiGraph()
    graph.add_nodes_from(names)
    for i in range(1, nodes):
        k = int(rng.integers(0, min(max_parents, i) + 1))
        for parent in rng.choice(i, size=k, replace=False):
            graph.add_edge(names[int(parent)], names[i])

    # Forward-sample: each node's state is a noisy function of its parents' states.
    data = {}
    for name in names:
        parents = list(graph.predecessors(name))
        base = rng.integers(0, card, size=rows)
        if parents:
            mix = sum(data[p] for p in parents) % card
            flip = rng.random(rows) < 0.3
            data[name] = np.where(flip, base, mix)
        else:
            data[name] = base
    return graph, pd.DataFrame(data)


def _timeit(fn, repeats: int) -> dict:
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return {
        "median_s": round(statistics.median(times), 4),
        "min_s": round(min(times), 4),
        "max_s": round(max(times), 4),
        "repeats": repeats,
    }


def network_from_tables(path: str) -> tuple[BayesianNetwork, str]:
    """Rebuild a fitted BayesianNetwork from ``export_legacy_model.py`` JSON."""
    from pgmpy.factors.discrete import TabularCPD

    spec = json.loads(Path(path).read_text())
    graph = nx.DiGraph()
    graph.add_nodes_from(spec["nodes"])
    graph.add_edges_from(map(tuple, spec["edges"]))
    bn = BayesianNetwork(graph)
    bn._node_states = {n: spec["cpds"][n]["states"] for n in spec["nodes"]}
    for node in spec["nodes"]:
        c = spec["cpds"][node]
        names = {node: c["states"], **dict(zip(c["parents"], c["parent_states"], strict=True))}
        bn.model.add_cpds(
            TabularCPD(
                node,
                len(c["states"]),
                c["values"],
                evidence=c["parents"] or None,
                evidence_card=[len(s) for s in c["parent_states"]] or None,
                state_names=names,
            )
        )
    bn.model.check_model()
    return bn, spec.get("target") or spec["nodes"][-1]


def run(
    model: Optional[str] = None,
    tables: Optional[str] = None,
    nodes: int = 30,
    card: int = 4,
    max_parents: int = 3,
    rows: int = 20000,
    repeats: int = 3,
    seed: int = 0,
    output: Optional[str] = None,
):
    report: dict = {"machine": machine_spec(), "results": {}}

    if tables:
        bn, target = network_from_tables(tables)
        report["source"] = {"tables": Path(tables).name}
    elif model:
        from causal_canvas.utils import safe_pickle_load

        estimator = safe_pickle_load(model)
        bn = estimator.model
        target = estimator.event_col
        report["source"] = {"model": Path(model).name}
    else:
        graph, data = random_network(nodes, card, max_parents, rows, seed)
        bn = BayesianNetwork(graph)
        start = time.perf_counter()
        bn.fit_node_states_and_cpds(data, method="BayesianEstimator", bayes_prior="K2")
        report["results"]["fit"] = {"median_s": round(time.perf_counter() - start, 4)}
        target = f"v{nodes - 1}"
        report["source"] = {
            "synthetic": {
                "nodes": nodes,
                "card": card,
                "max_parents": max_parents,
                "rows": rows,
                "edges": graph.number_of_edges(),
            }
        }

    engine = InferenceEngine(bn)
    states = bn.node_states
    cond_node = next(n for n in bn.nodes if n != target)
    cond = {cond_node: states[cond_node][0]}
    lever = cond_node
    uniform = {s: 1 / len(states[lever]) for s in states[lever]}

    res = report["results"]
    res["query_all_nodes"] = _timeit(lambda: engine.query(), repeats)
    res["query_target"] = _timeit(lambda: engine.query(nodes=[target]), repeats)
    res["query_target_given_evidence"] = _timeit(
        lambda: engine.query(cond, nodes=[target]), repeats
    )

    def intervene():
        engine.do_intervention(lever, uniform)
        engine.query(nodes=[target])
        engine.reset_do(lever)

    res["do_then_query_target"] = _timeit(intervene, repeats)

    slowest = max(v["median_s"] for k, v in res.items() if k != "fit")
    report["adr_target"] = {
        "target_s": 30,
        "stretch_s": 5,
        "slowest_median_s": slowest,
        "meets_target": slowest <= 30,
        "meets_stretch": slowest <= 5,
    }
    text = json.dumps(report, indent=2)
    if output:
        Path(output).write_text(text)
    print(text)


if __name__ == "__main__":
    fire.Fire(run)
