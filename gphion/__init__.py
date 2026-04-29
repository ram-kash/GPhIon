"""
GPHIon - Glassy Phase Ionic Conductor Analysis Package
=====================================================

A comprehensive package for analyzing ion diffusion in glassy phase ionic conductors
using molecular dynamics simulations.

The package implements the Time-Averaged Occupancy Grid (TAOG) method along with
various other analysis techniques including Van Hove correlation functions,
collective jump analysis, and advanced visualization capabilities.
"""

__version__ = "1.0.0"
__author__ = "Dr. Ram Sewak"
__email__ = "ram.sewak@iitgn.ac.in"
__license__ = "MIT"

from .core.sim_data import SimData
from .core.coordinate_processor import CoordinateProcessor
from .analyzers.site_finder import AmorphousSiteFinder
from .analyzers.jump_analyzer import JumpAnalyzer
from .analyzers.diffusivity_analyzer import JumpDiffusivityAnalyzer
from .analyzers.collective_jump_analyzer import CollectiveJumpAnalyzer
from .analyzers.vibration_analyzer import VibrationAnalyzer
from .analyzers.rdf_analyzer import RDFAnalyzer, RDFData
from .analyzers.tracer_analyzer import TracerPropertyAnalyzer
from .analyzers.vanhove_analyzer import VanHoveAnalyzer
from .analyzers.ngp_analyzer import NonGaussianCalculator
from .analyzers.activation_energy_analyzer import ActivationEnergyAnalyzer
from .visualization.plotting import plot_taog_density
from .visualization.vanhove_plotter import VanHovePlotter
from .visualization.displacement_plotter import DisplacementPlotter
from .visualization.collective_plotter import CollectivePlotter
from .workflow.main import load_simulation_data

__all__ = [
    # Core classes
    'SimData',
    'CoordinateProcessor',
    
    # Analysis classes
    'AmorphousSiteFinder',
    'JumpAnalyzer',
    'JumpDiffusivityAnalyzer',
    'CollectiveJumpAnalyzer',
    'VibrationAnalyzer',
    'RDFAnalyzer',
    'RDFData',
    'TracerPropertyAnalyzer',
    'VanHoveAnalyzer',
    'ActivationEnergyAnalyzer',
    'NonGaussianCalculator',
    
    # Visualization classes
    'plot_taog_density',
    'VanHovePlotter',
    'DisplacementPlotter',
    'CollectivePlotter',
    # Utility functions
    'load_simulation_data',
]

# Package metadata
__title__ = "GPHIon"
__description__ = "Glassy Phase Ionic Conductor Analysis Package"
__url__ = "https://github.com/yourusername/gphion"

