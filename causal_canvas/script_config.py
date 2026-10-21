"""
Pydantic schema for the ``infer-bn`` script configuration.

Defines the pydantic v2 ``BaseModel`` describing the shape of the
``config.yaml`` consumed by the ``infer-bn`` console entry point —
field-level validation, default values, and YAML loading via
``ScriptConfig.load_yaml``. Importing this module has no side effects
beyond schema construction.

**Layer**: internal — schema validation only. Not part of the public
runtime API.

See also
--------
``infer_bayesian_network.py`` — the script that loads ``config.yaml``
    through :meth:`ScriptConfig.load_yaml`.
"""

from abc import ABC
from datetime import date
from pathlib import Path
from typing import Annotated, Optional, Union

import yaml
from pydantic import (
    BaseModel,
    Field,
    FilePath,
    field_validator,
    model_validator,
)

from causal_canvas.bayesian_network_estimator import (
    DISCRETISER_METHODS,
    InferenceMethod,
)


class YamlConfig(BaseModel, ABC):
    @classmethod
    def load_yaml(cls, file: Union[str, Path]):
        """Create an instance of the class from a yaml file"""
        path = Path(file)

        if not path.exists():
            raise FileNotFoundError(f"File {file} not found")

        config = yaml.safe_load(path.read_text())
        return cls(**config)


class DateInterval(BaseModel):
    start: date
    end: date

    @model_validator(mode="after")
    def start_le_end(self) -> "DateInterval":
        assert self.start <= self.end
        return self


class Discretiser(BaseModel):
    method: DISCRETISER_METHODS = "simple"
    argument: Annotated[int, Field(gt=0)] = 5
    cutoffs_target: list = Field(default_factory=list)
    proportion_threshold: Optional[Annotated[float, Field(gt=0, le=1)]] = None
    max_categories: Optional[int] = None


class ScriptConfig(YamlConfig):
    data_input_path: FilePath
    structure_path: Optional[FilePath] = None
    train_dates: Optional[DateInterval] = None
    id_column: Optional[str] = None
    date_column: str
    event_column: str
    drop_columns: list[str]
    features_select: list[str]
    categorical_features: list[str]
    drop_nans: Optional[bool] = True
    connections_type: Optional[str] = None
    lasso_multiplier: Optional[float] = None
    non_linear_args: dict = Field(default_factory=dict)
    max_iter: Annotated[int, Field(ge=1)]
    h_tol: Annotated[float, Field(ge=1e-10)]
    w_threshold: Optional[Annotated[float, Field(ge=0)]] = None
    tabu_edges: Optional[list] = None
    tabu_edge_features: Optional[list[str]] = None
    event_graph_label: Optional[str] = None
    event_color: Optional[str] = None
    higher_contribution_feature_color: Optional[str] = None
    invert_signs: Optional[bool] = None
    discretiser: Discretiser
    inference_method: InferenceMethod
    conditional_dependency_estimates: Optional[list] = None
    sample_frac: Optional[Annotated[float, Field(gt=0, le=1)]] = None
    # create_new_folder: Optional[bool] = True
    add_datetime_to_folder: Optional[bool] = False
    output_path: Path

    @field_validator("max_iter", mode="before")
    @classmethod
    def set_max_iter_default(cls, v):
        return 100 if v is None else v

    @field_validator("h_tol", mode="before")
    @classmethod
    def set_h_tol_default(cls, v):
        return 1e-8 if v is None else v

    @field_validator("categorical_features", mode="before")
    @classmethod
    def categorical_features_replace_none_with_list(cls, v) -> list[str]:
        return v if v is not None else []


class ScriptConfigEvaluation(YamlConfig):
    data_train_input_path: list[FilePath]
    data_test_input_path: Optional[list[FilePath]] = None
    event_column: dict
    date_column: Optional[str] = None
    optim_test_dates: Optional[list[str]] = None
    test_dates: Optional[list[str]] = None
    all_models_path: str
    inference_methods: list[str]
    models_combinations: dict
    features: list[str]
    use_multiprocessing: bool
    score_to_optimise: str
    boot_iterations: Annotated[int, Field(ge=1)]
    alpha: Annotated[float, Field(ge=0, le=1)]
    output_path: Path


class Conditional(BaseModel):
    feature: str
    conditional_features: list[list[str]]


class ProbaShift(BaseModel):
    p0: float
    p1: float

    @classmethod
    def init_low(cls) -> "ProbaShift":
        return cls(p0=0.9, p1=0.1)

    @classmethod
    def init_high(cls) -> "ProbaShift":
        return cls(p0=0.1, p1=0.9)

    def to_state_map(self) -> dict:
        """Return {0: p0, 1: p1} — the shape InferenceEngine.do_intervention wants."""
        return {0: self.p0, 1: self.p1}


class Intervention(BaseModel):
    feature: str
    low: ProbaShift = Field(default_factory=ProbaShift.init_low)
    high: ProbaShift = Field(default_factory=ProbaShift.init_high)


class ScriptConfigInference(YamlConfig):
    event_column: str
    model_path: FilePath
    output_path: Path
    target_class: int
    conditionals: list[Conditional]
    interventions: list[Intervention]
    counterfactuals: list[str]
