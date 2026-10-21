import pickle

import pytest

from causal_canvas.utils import safe_pickle_load


def test_safe_pickle_load_round_trips(tmp_path):
    """A pickle written and read back from the same trusted dir should work."""
    target = tmp_path / "model.pkl"
    payload = {"weights": [1, 2, 3], "name": "fake_model"}

    with target.open("wb") as fp:
        pickle.dump(payload, fp)

    result = safe_pickle_load(target, trusted_root=tmp_path)

    assert result == payload


def test_safe_pickle_load_rejects_path_traversal(tmp_path):
    """A path that resolves outside the trusted root must be refused."""
    # create a dummy "outside" file
    outside = tmp_path.parent / "outside.pkl"
    with outside.open("wb") as fp:
        pickle.dump({"hostile": True}, fp)

    try:
        with pytest.raises(ValueError, match="outside trusted root"):
            safe_pickle_load(outside, trusted_root=tmp_path)
    finally:
        outside.unlink(missing_ok=True)


def test_safe_pickle_load_rejects_relative_traversal(tmp_path):
    """Even a `..` suffix must resolve inside the trusted root to be allowed."""
    target = tmp_path / "model.pkl"
    with target.open("wb") as fp:
        pickle.dump({"ok": 1}, fp)

    # path that escapes and re-enters — must NOT bypass the check
    escaped = tmp_path / ".." / tmp_path.name / "model.pkl"
    result = safe_pickle_load(escaped, trusted_root=tmp_path)
    assert result == {"ok": 1}

    # path that escapes and does NOT re-enter — must fail
    with pytest.raises(ValueError, match="outside trusted root"):
        safe_pickle_load(tmp_path / ".." / "nothing.pkl", trusted_root=tmp_path)


def test_safe_pickle_load_explains_legacy_causalnex_pickles(tmp_path):
    """Pre-0.2.0 artefacts reference `causalnex`; the error must say to retrain."""
    import sys
    import types

    legacy = types.ModuleType("causalnex_legacy_stub")
    legacy.__dict__["__name__"] = "causalnex.network"
    sys.modules["causalnex"] = types.ModuleType("causalnex")
    sys.modules["causalnex.network"] = legacy
    try:
        cls = type("BayesianNetwork", (), {"__module__": "causalnex.network"})
        legacy.BayesianNetwork = cls
        target = tmp_path / "old.pkl"
        target.write_bytes(pickle.dumps(cls()))
    finally:
        sys.modules.pop("causalnex.network")
        sys.modules.pop("causalnex")

    with pytest.raises(RuntimeError, match="retrain"):
        safe_pickle_load(target, trusted_root=tmp_path)
