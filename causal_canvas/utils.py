"""
Shared utilities used across the package.

Small cross-cutting helpers: :func:`safe_pickle_load` (path-validated
unpickler with SHA-256 logging — see the Security section in the
project README for the trust model), :func:`restructure_nx_object`
(extracts the target weakly-connected component for inference), and
:func:`save_structure` (DAG persistence as a Graphviz ``.dot`` file).

**Layer**: internal. These helpers have no external entry-point users;
they are imported by other first-party modules only.

See also
--------
:mod:`causal_canvas.bayesian_network_estimator` — calls
    :func:`safe_pickle_load` to reload fitted networks.
``infer_do_why.py`` — top-level script that loads pickled artefacts.
README — the Security section explains the pickle trust model.
"""

import hashlib
import pickle
from pathlib import Path
from typing import Any, Optional, Union

import networkx as nx
from loguru import logger

from causal_canvas.structure_learner import StructureModel


def safe_pickle_load(path: Union[str, Path], trusted_root: Optional[Path] = None) -> Any:
    """
    Load a pickle file after verifying it lives inside a trusted root.

    Pickle executes arbitrary Python during deserialization, so the file must
    be trusted. We enforce that the resolved path lives inside
    ``trusted_root`` (defaults to ``Path.cwd()``). Paths like
    ``../../etc/passwd`` are rejected because their resolved form escapes
    the project root.

    A SHA-256 prefix is logged on every load so tampering with a file that
    was produced in a prior run leaves a trail.

    Parameters
    ----------
    path:
        Path to the pickle file.
    trusted_root:
        Directory the path must resolve inside. Defaults to current working
        directory.

    Raises
    ------
    ValueError:
        If ``path`` resolves outside ``trusted_root``.
    RuntimeError:
        If the pickle was written by causal-canvas < 0.2.0 (it references the
        removed ``causalnex`` package); such artefacts must be retrained.
    """
    resolved = Path(path).resolve()
    root = (trusted_root or Path.cwd()).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(
            f"Refusing to load pickle outside trusted root: {resolved} not under {root}"
        )

    data = resolved.read_bytes()
    digest = hashlib.sha256(data).hexdigest()[:16]
    logger.info(f"Loading pickle {resolved} (sha256={digest})")
    try:
        return pickle.loads(data)
    except ModuleNotFoundError as exc:
        if (exc.name or "").split(".")[0] == "causalnex":
            raise RuntimeError(
                f"{resolved} was written by causal-canvas < 0.2.0 and references the removed "
                "`causalnex` package. 0.2.0 moved the Bayesian network to pgmpy and does not "
                "read the old format: re-run `infer-bn` to retrain the model (ADR-0001)."
            ) from exc
        raise


def restructure_nx_object(nx_structure):
    """
    Restructures a NetworkX directed graph by extracting the target connected component.

    Parameters
    ----------
        nx_structure (nx.DiGraph): The original directed graph represented as a NetworkX DiGraph.

    Returns
    ----------
        nx.DiGraph: The connected component of the input graph.
    """

    component_graphs = []
    for component in nx.weakly_connected_components(nx_structure):
        G = nx.DiGraph()
        for node in component:
            for edge in nx_structure.out_edges(node):
                G.add_edge(edge[0], edge[1])
        component_graphs.append(G)
    return component_graphs[0]


def save_structure(sm_object: StructureModel, path: Path):
    """
    Save a structure in .dot files:

    Parameters
    ----------
    sm_object: StructureModel
    path:  str
        folder path to save file
    """
    nx.drawing.nx_pydot.write_dot(sm_object, path)
