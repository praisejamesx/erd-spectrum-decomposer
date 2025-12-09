"""
Elegant Recursive Discovery (ERD) Engine
An autonomous AI framework for explanatory decomposition of scientific data.
"""

from .core import (
    ElegantComponentNode,
    ElegantPolynomialSearcher,
    ElegantExponentialSearcher,
    GaussianPeakSearcher,
    ElegantLorentzianSearcher
)
from .engine import SpectralPatternEngine, ElegantRecursiveDiscovery
from .utils import plot_results, load_ruff_data, add_noise

__version__ = "1.0.0"
__all__ = [
    'ElegantComponentNode',
    'ElegantPolynomialSearcher',
    'ElegantExponentialSearcher',
    'GaussianPeakSearcher',
    'ElegantLorentzianSearcher',
    'SpectralPatternEngine',
    'ElegantRecursiveDiscovery',
    'plot_results',
    'load_ruff_data',
    'add_noise'
]