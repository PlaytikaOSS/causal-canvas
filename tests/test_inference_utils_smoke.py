import numpy as np
import pandas as pd

from causal_canvas.inference_utils import (
    change_keys,
    compute_uplift,
    convert_dict_for_json,
)


def test_change_keys_remaps_with_fallback():
    assert change_keys({"old": "new"}, {"old": 1, "other": 2}) == {"new": 1, "other": 2}


def test_convert_dict_for_json_strips_numpy_types():
    src = {
        np.int64(1): np.float64(0.5),
        "nested": {np.bool_(True): np.int64(42)},
        "plain": "string",
    }

    result = convert_dict_for_json(src)

    assert result[1.0] == 0.5
    assert result["nested"][1.0] == 42.0
    assert result["plain"] == "string"
    # Every leaf must be JSON-serialisable (no numpy types remain)
    import json

    json.dumps(result)


def test_compute_uplift_control_minus_treatment():
    cf = pd.DataFrame(
        {
            "control": [0.2, 0.8],
            "A_low": [0.5, 0.5],
            "A_high": [0.1, 0.9],
        },
        index=[0, 1],
    )

    uplifts = compute_uplift(cf, target_class=1)

    assert list(uplifts.columns) == ["bin", "uplift"]
    assert set(uplifts["bin"]) == {"A_low", "A_high"}
    # control[1]=0.8; uplift_low = 0.8 - 0.5 = 0.3; uplift_high = 0.8 - 0.9 = -0.1
    row_low = uplifts[uplifts["bin"] == "A_low"].iloc[0]
    row_high = uplifts[uplifts["bin"] == "A_high"].iloc[0]
    assert abs(row_low["uplift"] - 0.3) < 1e-9
    assert abs(row_high["uplift"] - (-0.1)) < 1e-9
