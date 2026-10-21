import random

import networkx as nx
import numpy as np
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def _seed_everything():
    """Seed every RNG the library might touch so tests are deterministic."""
    random.seed(42)
    np.random.seed(42)
    try:
        import torch

        torch.manual_seed(42)
    except ImportError:
        pass


@pytest.fixture
def synthetic_dag() -> nx.DiGraph:
    """A deterministic 4-node DAG: A, B, C are parents of D; A is a parent of B and C."""
    graph = nx.DiGraph()
    graph.add_edges_from([("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")])
    return graph


@pytest.fixture
def synthetic_data() -> pd.DataFrame:
    """
    Small Gaussian sample from `synthetic_dag`:
    A ~ N(0, 1); B = 2*A + eps; C = -A + eps; D = B + C + eps.
    Adds a timestamp column (required by Preprocessor) and a binary event column.
    """
    n = 200
    a = np.random.randn(n)
    b = 2 * a + np.random.randn(n) * 0.1
    c = -a + np.random.randn(n) * 0.1
    d = b + c + np.random.randn(n) * 0.1
    event = (d > d.mean()).astype(int)
    return pd.DataFrame(
        {
            "A": a,
            "B": b,
            "C": c,
            "D": d,
            "event": event,
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="D").astype(str),
        }
    )
