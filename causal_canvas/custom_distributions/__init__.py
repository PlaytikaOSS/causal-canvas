"""
Distribution-type bundle for non-linear NOTEARS.

Distribution-type implementations registered with
``causal_canvas.vendored.dist_type`` so that
:class:`causal_canvas.custom_notears_mlp.NotearsMLP` can model
non-Gaussian features: gamma, tweedie, bimodal positive-continuous,
ordinal, categorical, etc. The ``dist_type_aliases`` mapping translates
the short string codes used in pipeline configs (``"gamma"``,
``"tweedie"``, ``"cat"``, …) into the corresponding ``DistTypeBase``
subclasses.

**Layer**: low-level / vendored.

**Relationship to causalnex**: re-exports the upstream
``DistTypeBinary``, ``DistTypeContinuous``, ``DistTypeOrdinal``, and
``DistTypePoisson`` classes alongside Playtika-specific subclasses
that follow the same ``DistTypeBase`` interface, so they can be
mixed freely when configuring an MLP fit.

See also
--------
:mod:`causal_canvas.custom_notears_mlp` — primary consumer.
:mod:`causal_canvas.custom_distributions.categorical` —
    :class:`DistTypeCategorical` implementation.
"""

from causal_canvas.vendored.dist_type.binary import DistTypeBinary
from causal_canvas.vendored.dist_type.continuous import DistTypeContinuous
from causal_canvas.vendored.dist_type.ordinal import DistTypeOrdinal
from causal_canvas.vendored.dist_type.poisson import DistTypePoisson

from .categorical import DistTypeCategorical
from .gamma import DistTypeGamma
from .non_negative_bimodal import DistTypeNonNegBimodal
from .non_negative_continuous import DistTypeNonNegativeContinuous
from .tweedie import DistTypeTweedie

dist_type_aliases = {
    "bin": DistTypeBinary,
    "cat": DistTypeCategorical,
    "cont": DistTypeContinuous,
    "pos_cont": DistTypeNonNegativeContinuous,
    "bimodal_pos_cont": DistTypeNonNegBimodal,
    "ord": DistTypeOrdinal,
    "poiss": DistTypePoisson,
    "gamma": DistTypeGamma,
    "tweedie": DistTypeTweedie,
}

__all__ = [
    "DistTypeBinary",
    "DistTypeCategorical",
    "DistTypeContinuous",
    "DistTypeNonNegativeContinuous",
    "DistTypeNonNegBimodal",
    "DistTypeOrdinal",
    "DistTypePoisson",
    "DistTypeGamma",
    "DistTypeTweedie",
]
