from pathlib import Path

import networkx as nx

from causal_canvas.structure_learner import StructureModel


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
