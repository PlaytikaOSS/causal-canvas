from causal_canvas.evaluation_utils import round_interval


def test_round_interval_parses_parentheses_brackets():
    assert round_interval("(4.7654, 8.7671]") == "[4.765, 8.767)"


def test_round_interval_preserves_integers_as_three_dp():
    assert round_interval("[0, 1)") == "[0.000, 1.000)"


def test_round_interval_handles_negatives():
    assert round_interval("(-1.5, 0.5]") == "[-1.500, 0.500)"
