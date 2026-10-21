from .base import DiscreteParameterEstimator as DiscreteParameterEstimator

class DiscreteMLE(DiscreteParameterEstimator):
    def __init__(self, state_names: dict | None = ..., n_jobs: int = ...) -> None: ...

class DiscreteBayesianEstimator(DiscreteParameterEstimator):
    def __init__(
        self,
        state_names: dict | None = ...,
        prior_type: str = ...,
        equivalent_sample_size: float | dict = ...,
        pseudo_counts: object = ...,
        n_jobs: int = ...,
    ) -> None: ...
