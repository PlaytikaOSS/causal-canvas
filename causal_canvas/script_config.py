from abc import ABC
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Union

import yaml
from pydantic import (
    BaseModel,
    Field,
    FilePath,
    confloat,
    conint,
    root_validator,
    validator,
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

    @root_validator(pre=False)
    def start_le_end(cls, values):
        assert values["start"] <= values["end"]
        return values


class Discretiser(BaseModel):
    method: DISCRETISER_METHODS = "simple"
    argument: conint(gt=0) = 5
    cutoffs_target: list = Field(default_factory=list)
    proportion_threshold: Optional[confloat(gt=0, le=1)]
    max_categories: Optional[int]


class ScriptConfig(YamlConfig):
    data_input_path: FilePath
    structure_path: Optional[FilePath]
    train_dates: Optional[DateInterval]
    id_column: Optional[str]
    date_column: str
    event_column: str
    drop_columns: List[str]
    features_select: List[str]
    categorical_features: List[str]
    drop_nans: Optional[bool] = True
    connections_type: Optional[str]
    lasso_multiplier: Optional[float]
    non_linear_args: Dict = Field(default_factory=dict)
    max_iter: conint(ge=1)
    h_tol: confloat(ge=1e-10)
    w_threshold: Optional[confloat(ge=0)]
    tabu_edges: Optional[List] = None
    tabu_edge_features: Optional[List[str]] = None
    event_graph_label: Optional[str]
    event_color: Optional[str]
    higher_contribution_feature_color: Optional[str]
    invert_signs: Optional[bool]
    discretiser: Discretiser
    inference_method: InferenceMethod
    conditional_dependency_estimates: Optional[List]
    sample_frac: Optional[confloat(gt=0, le=1)]
    # create_new_folder: Optional[bool] = True
    add_datetime_to_folder: Optional[bool] = False
    output_path: Path

    @validator("max_iter", pre=True)
    def set_max_iter_default(cls, v):
        return 100 if v is None else v

    @validator("h_tol", pre=True)
    def set_h_tol_default(cls, v):
        return 1e-8 if v is None else v

    @validator("categorical_features")
    def categorical_features_replace_none_with_list(cls, v) -> list[str]:
        return v if v is not None else []


class ScriptConfigEvaluation(YamlConfig):
    data_train_input_path: List[FilePath]
    data_test_input_path: Optional[List[FilePath]]
    event_column: Dict
    date_column: Optional[str] = None
    optim_test_dates: Optional[List[str]]
    test_dates: Optional[List[str]]
    all_models_path: str
    inference_methods: List[str]
    models_combinations: Dict
    features: List[str]
    use_multiprocessing: bool
    score_to_optimise: str
    boot_iterations: conint(ge=1)
    alpha: confloat(ge=0, le=1)
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

    def dict(self):
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
