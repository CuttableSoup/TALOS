"""TALOS: Targeted Ablation Laser Orbital Sweeper — engagement simulation."""

from .debris import DebrisPopulation
from .elements import KeplerElements, elements_to_rv, rv_to_elements
from .engagement import AblationParams, BurstResult, integrate_burst
from .platform import (
    ATPConfig,
    LaserConfig,
    Platform,
    ResourceConfig,
    baseline_constellation,
    baseline_platform,
)

__all__ = [
    "ATPConfig",
    "AblationParams",
    "BurstResult",
    "DebrisPopulation",
    "KeplerElements",
    "LaserConfig",
    "Platform",
    "ResourceConfig",
    "baseline_constellation",
    "baseline_platform",
    "elements_to_rv",
    "integrate_burst",
    "rv_to_elements",
]
