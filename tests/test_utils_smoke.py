import networkx as nx

from causal_canvas.utils import restructure_nx_object, save_structure


def test_restructure_nx_object_returns_connected_component():
    graph = nx.DiGraph()
    graph.add_edges_from([("A", "B"), ("B", "C")])
    graph.add_node("isolated")

    result = restructure_nx_object(graph)

    assert isinstance(result, nx.DiGraph)
    assert set(result.edges) == {("A", "B"), ("B", "C")}


def test_save_structure_writes_dot(tmp_path):
    from causal_canvas.vendored.structure_model import StructureModel

    sm = StructureModel()
    sm.add_weighted_edges_from([("A", "B", 0.5)])
    out = tmp_path / "graph.dot"

    save_structure(sm, out)

    assert out.exists()
    content = out.read_text()
    assert "A" in content and "B" in content
