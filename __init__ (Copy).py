"""
TAOG (Time-Averaged Occupancy Grid) Analysis Package

A comprehensive package for analyzing ion diffusion in amorphous materials
using molecular dynamics simulations.
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from .core.sim_data import SimData
from .core.coordinate_processor import CoordinateProcessor
from .analyzers.site_finder import AmorphousSiteFinder
from .analyzers.jump_analyzer import SiteAnalyzerMDA
from .analyzers.diffusivity_analyzer import JumpDiffusivityAnalyzer
from .analyzers.collective_jump_analyzer import CollectiveJumpAnalyzer
from .analyzers.vibration_analyzer import VibrationAnalyzer
from .analyzers.rdf_analyzer import RDFAnalyzer, RDFData
from .analyzers.tracer_analyzer import TracerPropertyAnalyzer
from .analyzers.vanhove_analyzer import VanHoveAnalyzer
from .visualization.plotting import plot_taog_density
from .visualization.vanhove_plotter import VanHovePlotter
from .visualization.displacement_plotter import DisplacementPlotter
from .visualization.collective_plotter import CollectivePlotter
from .visualization.connectivity_plotter import ConnectivityPlotter
from .workflow.main import load_simulation_data

__all__ = [
    'SimData',
    'CoordinateProcessor', 
    'AmorphousSiteFinder',
    'SiteAnalyzerMDA',
    'JumpDiffusivityAnalyzer',
    'CollectiveJumpAnalyzer',
    'VibrationAnalyzer',
    'RDFAnalyzer',
    'RDFData',
    'TracerPropertyAnalyzer',
    'VanHoveAnalyzer',
    'plot_taog_density',
    'VanHovePlotter',
    'DisplacementPlotter',
    'CollectivePlotter',
    'ConnectivityPlotter',
    'load_simulation_data'
]

