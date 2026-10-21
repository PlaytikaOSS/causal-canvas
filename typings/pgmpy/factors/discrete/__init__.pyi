from collections.abc import Hashable, Sequence
from typing import Any

import numpy as np

class TabularCPD:
    variable: Any
    variables: list[Any]
    state_names: dict[Any, list[Any]]
    def __init__(
        self,
        variable: Hashable,
        variable_card: int,
        values: Sequence[Sequence[float]] | np.ndarray,
        evidence: Sequence[Hashable] | None = ...,
        evidence_card: Sequence[int] | None = ...,
        state_names: dict[Hashable, Sequence[Hashable]] = ...,
    ) -> None: ...
    def get_values(self) -> np.ndarray: ...
    def get_value(self, **kwargs: Hashable) -> float: ...
