"""Model components."""

from .adapters import (
    LatentConditionedFiLMSpectralAdapter,
    PaperSpectralGateAdapter,
    StateConditionedSpectralAdapter,
)
from .backbone import SpectralGatedBackbone
from .bfn import (
    DegradationEvidenceEncoder,
    DegradationLatentBFN,
    ResidualBayesianFlow,
    degradation_latent_targets,
    local_entropy_map,
)
from .coefficient_state import BasisBFNReadout, CoefficientStateBasisReadout
from .foundation import (
    ExternalSSRFoundation,
    SpectralUncertaintyAdapter,
    build_foundation,
)
from .mean import ConditionalMeanCorrector
from .model import BasisBFN, SGBFormer, build_model
from .receiver import (
    CategoryRoutedReadout,
    PrivateBottleneckRoutedResidualHead,
    ResidualReceiver,
    RoutedResidualHead,
)
from .signed_basis import SignedSpatialBasisReadout
from .spectral import SpectralGate, spectral_scale_delta

__all__ = [
    "ExternalSSRFoundation",
    "ConditionalMeanCorrector",
    "BasisBFNReadout",
    "CoefficientStateBasisReadout",
    "DegradationEvidenceEncoder",
    "DegradationLatentBFN",
    "LatentConditionedFiLMSpectralAdapter",
    "PaperSpectralGateAdapter",
    "CategoryRoutedReadout",
    "PrivateBottleneckRoutedResidualHead",
    "ResidualBayesianFlow",
    "ResidualReceiver",
    "RoutedResidualHead",
    "BasisBFN",
    "SGBFormer",
    "SignedSpatialBasisReadout",
    "SpectralUncertaintyAdapter",
    "StateConditionedSpectralAdapter",
    "SpectralGate",
    "SpectralGatedBackbone",
    "build_foundation",
    "build_model",
    "degradation_latent_targets",
    "local_entropy_map",
    "spectral_scale_delta",
]
