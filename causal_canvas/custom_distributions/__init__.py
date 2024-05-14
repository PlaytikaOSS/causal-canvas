from causalnex.structure.pytorch.dist_type.binary import DistTypeBinary
from causalnex.structure.pytorch.dist_type.continuous import DistTypeContinuous
from causalnex.structure.pytorch.dist_type.ordinal import DistTypeOrdinal
from causalnex.structure.pytorch.dist_type.poisson import DistTypePoisson

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
