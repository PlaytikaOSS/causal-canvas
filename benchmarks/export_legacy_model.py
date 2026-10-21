"""Export a pre-0.2.0 fitted model (causalnex pickle) to a portable JSON of CPTs.

Run this inside the OLD environment (causal-canvas < 0.2.0 + causalnex), where the
pickle still loads. The JSON holds only structure and probability tables -- no
training data -- and can be benchmarked in the new environment with::

    python benchmarks/bench_inference.py --tables model_tables.json

Usage (old environment):

    python export_legacy_model.py --model path/to/bayesian_network.pkl --output model_tables.json

Security: pickle executes code on load; only export models you produced yourself.
"""

import json
import pickle
from pathlib import Path

import fire


def _native(x):
    return x.item() if hasattr(x, "item") else x


def export(model: str, output: str = "model_tables.json"):
    with Path(model).open("rb") as fp:
        estimator = pickle.load(fp)  # noqa: S301 - trusted, user-produced artefact
    bn = estimator.model
    cpds = {}
    for node, table in bn.cpds.items():
        parents = [n for n in (table.columns.names or []) if n]
        cpds[node] = {
            "states": [_native(s) for s in table.index],
            "parents": parents,
            "parent_states": [
                [_native(s) for s in table.columns.get_level_values(p).unique()] for p in parents
            ],
            # rows = node states; columns = parent-state combinations in table order
            "values": table.values.tolist(),
        }
    payload = {
        "target": getattr(estimator, "event_col", None),
        "nodes": sorted(bn.nodes),
        "edges": [list(e) for e in bn.edges],
        "cpds": cpds,
    }
    Path(output).write_text(json.dumps(payload))
    print(f"wrote {output}: {len(payload['nodes'])} nodes, {len(payload['edges'])} edges")


if __name__ == "__main__":
    fire.Fire(export)
