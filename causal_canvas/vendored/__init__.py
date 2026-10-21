"""Code vendored from ``causalnex`` 0.12.1 (Apache-2.0, QuantumBlack Visual Analytics).

``causalnex`` is unmaintained and pins a CVE-affected dependency chain
(``torch==2.2.1``, ``pyarrow==15.0.0``), so the pieces causal-canvas relies on
are owned here instead of installed (ADR-0001). The Bayesian-network and
inference layers are *not* vendored: they moved to ``pgmpy``.

Modifications are limited to rewriting intra-package imports from
``causalnex.*`` to ``causal_canvas.vendored.*`` and the compatibility fixes
listed in ``CHANGELOG.md``. The original licence is kept in
``LICENSE.causalnex.md``; every file retains its upstream header.

Contents
--------
structure_model      ``StructureModel`` (networkx ``DiGraph`` subclass)
notears_linear       linear NOTEARS (``from_pandas`` / ``from_pandas_lasso``)
locally_connected    ``LocallyConnected`` torch layer used by NotearsMLP
dist_type            ``DistTypeBase`` and the upstream distribution types
discretiser          quantile / fixed / decision-tree / MDLP discretisers
decision_tree_tools  helper used by the tree discretisers
plots                ``plot_structure`` (graphviz-style rendering)
"""
