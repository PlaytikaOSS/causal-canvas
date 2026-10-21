"""
Causal Canvas — curated public API.

Re-exports the four classes most consumers need:
:class:`StructureLearner`, :class:`BayesianNetworkEstimator`,
:class:`Preprocessor`, and :class:`InferenceMethod`. Lower-level helpers
(free functions in ``inference_utils`` / ``evaluation_utils``,
distribution types, utilities) remain importable from their submodules.

**Layer**: high-level / public. The wheel ships a ``py.typed`` marker,
so importing from this top-level surface gives downstream code full
type information without further configuration.

See also
--------
:mod:`causal_canvas.structure_learner` — :class:`StructureLearner`.
:mod:`causal_canvas.bayesian_network_estimator` —
    :class:`BayesianNetworkEstimator`, :class:`InferenceMethod`.
:mod:`causal_canvas.preprocessor` — :class:`Preprocessor`.
"""

from causal_canvas.bayesian_network_estimator import (
    BayesianNetworkEstimator,
    InferenceMethod,
)
from causal_canvas.preprocessor import Preprocessor
from causal_canvas.structure_learner import StructureLearner

__all__ = [
    "BayesianNetworkEstimator",
    "InferenceMethod",
    "Preprocessor",
    "StructureLearner",
]
