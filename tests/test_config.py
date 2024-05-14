from datetime import date
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from causal_canvas.script_config import DateInterval, ScriptConfig, YamlConfig


@pytest.fixture
def tmp_file(tmp_path) -> Path:
    return tmp_path / "conf.yaml"


@pytest.fixture
def script_config_dict(tmp_path) -> dict:
    input_path = tmp_path / "data.csv"
    input_path.touch()

    structure_path = tmp_path / "structure.csv"
    structure_path.touch()

    return {
        "data_input_path": str(input_path),
        "structure_path": str(structure_path),
        "train_dates": {"start": "2019-01-01", "end": "2019-01-02"},
        "id_column": "id",
        "date_column": "date",
        "event_column": "event",
        "drop_columns": ["drop1", "drop2"],
        "features_select": ["feat1", "feat2"],
        "categorical_features": ["cat1", "cat2"],
        "drop_nans": True,
        "connections_type": "connections",
        "lasso_multiplier": 1.0,
        "non_linear_args": {"arg1": "value1"},
        "max_iter": 100,
        "h_tol": 1e-5,
        "w_threshold": 0.1,
        "tabu_edges": ["edge1", "edge2"],
        "tabu_edge_features": ["feat1", "feat2"],
        "event_graph_label": "label",
        "event_color": "color",
        "higher_contribution_feature_color": "color",
        "invert_signs": False,
        "discretiser": {"arg1": "value1"},
        "inference_method": {
            "method": "MaximumLikelihoodEstimator",
            "bayes_prior": "K2",
        },
        "conditional_dependency_estimates": ["est1", "est2"],
        "sample_frac": 0.5,
        "create_new_folder": True,
        "add_datetime_to_folder": False,
        "output_path": "output",
    }


def test_load_yaml(tmp_file):
    # Arrange
    tmp_file.write_text(
        yaml.dump({"data_input_path": "data.csv", "output_path": "output"})
    )

    # Act
    config = YamlConfig.load_yaml(tmp_file)

    # Assert
    assert isinstance(config, YamlConfig)


def test_load_yaml_str(tmp_file):
    # Arrange
    tmp_file.write_text(
        yaml.dump({"data_input_path": "data.csv", "output_path": "output"})
    )

    # Act
    config = YamlConfig.load_yaml(str(tmp_file))  # type: ignore

    # Assert
    assert isinstance(config, YamlConfig)


def test_load_script_config(tmp_file, script_config_dict):
    # Generate a configuration dict for ScriptConfig
    tmp_file.write_text(yaml.dump(script_config_dict))

    # Act
    config = ScriptConfig.load_yaml(tmp_file)

    # Assert
    assert isinstance(config, ScriptConfig)


@pytest.mark.parametrize(
    argnames=["key", "value"],
    argvalues=[
        ("max_iter", 0),
        ("max_iter", -1),
        ("h_tol", 0),
        ("h_tol", -1e-10),
        ("w_threshold", -1),
        ("sample_frac", 0),
        ("sample_frac", 1.1),
    ],
)
def test_load_script_config_constrictions(tmp_file, script_config_dict, key, value):
    # Arrange
    bad_dict = script_config_dict.copy()
    bad_dict[key] = value

    tmp_file.write_text(yaml.dump(bad_dict))

    # Act
    with pytest.raises(ValueError):
        ScriptConfig.load_yaml(tmp_file)


@pytest.mark.parametrize(
    argnames=["key", "default_value"], argvalues=[("max_iter", 100), ("h_tol", 1e-8)]
)
def test_script_config_default_values(tmp_file, script_config_dict, key, default_value):
    a_dict = script_config_dict.copy()
    a_dict[key] = None

    tmp_file.write_text(yaml.dump(a_dict))

    # Act
    sc = ScriptConfig.load_yaml(tmp_file).dict()

    assert sc[key] == default_value


def test_dateinterval():
    start, end = "2019-01-01", "2019-01-02"
    di = DateInterval(start=start, end=end)
    assert isinstance(di.start, date)

    di = DateInterval(start=start, end=start)
    assert di.start == di.end

    with pytest.raises(ValidationError):
        DateInterval(start=end, end=start)
