from collections.abc import Hashable
from typing import Any

from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import DiscreteBayesianNetwork

class VariableElimination:
    def __init__(self, model: DiscreteBayesianNetwork) -> None: ...
    def query(
        self,
        variables: list[Hashable],
        evidence: dict[Any, Any] | None = ...,
        virtual_evidence: list[Any] | None = ...,
        elimination_order: Any = ...,
        joint: bool = ...,
        show_progress: bool = ...,
    ) -> TabularCPD: ...
