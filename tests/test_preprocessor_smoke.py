from causal_canvas.preprocessor import Preprocessor


def test_preprocess_returns_dataframe_with_event(synthetic_data, tmp_path):
    preprocessor = Preprocessor(
        df=synthetic_data,
        event_col="event",
        user_id_col=None,
        date_col="timestamp",
        drop_columns=[],
        categorical_columns=[],
        features_select=["A", "B", "C", "D"],
    )

    out = preprocessor.preprocess(artifacts_dir=tmp_path)

    assert "event" in out.columns
    # all requested numeric features must survive
    for col in ["A", "B", "C", "D"]:
        assert col in out.columns
    assert len(out) == len(synthetic_data)
    # timestamp and user_id (None) must be dropped
    assert "timestamp" not in out.columns


def test_preprocess_runs_on_data_with_nans(synthetic_data, tmp_path):
    """
    Preprocessor itself does NOT drop NaNs (callers like infer_bayesian_network.py
    do that before handing data in). This test pins the behaviour so a future
    refactor that silently starts dropping rows is caught.
    """
    df = synthetic_data.copy()
    df.loc[0, "A"] = float("nan")

    preprocessor = Preprocessor(
        df=df,
        event_col="event",
        user_id_col=None,
        date_col="timestamp",
        drop_columns=[],
        categorical_columns=[],
        features_select=["A", "B", "C", "D"],
    )

    out = preprocessor.preprocess(artifacts_dir=tmp_path)

    # Row count preserved — preprocessor does not silently drop NaN rows
    assert len(out) == len(df)
