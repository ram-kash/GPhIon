# GPHIon - Glassy Phase Ionic Conductor Analysis Package

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-active-brightgreen.svg)]()

GPHIon is a comprehensive Python package for analyzing ion diffusion in glassy phase ionic conductors using molecular dynamics simulations. The package implements the Time-Averaged Occupancy Grid (TAOG) method along with various other analysis techniques to provide deep insights into ionic conduction mechanisms.

## Features

### Core Analysis Capabilities
- **🔬 TAOG Analysis**: Time-Averaged Occupancy Grid for site discovery in amorphous materials
- **⚡ Jump Analysis**: Detection and analysis of ion hopping events between sites
- **🤝 Collective Jump Analysis**: Identification of correlated and cooperative jump events
- **📊 Diffusivity Calculations**: Both tracer and jump diffusivity measurements
- **🌊 Van Hove Functions**: Self and distinct Van Hove correlation function analysis
- **📈 Vibrational Analysis**: Ion attempt frequency and vibration amplitude calculations
- **📐 Radial Distribution Functions**: Site-specific and total RDF analysis
- **🔗 Connectivity Analysis**: Site connectivity networks and diffusion pathway visualization

### Visualization Capabilities
- **3D Density Plots**: TAOG density visualization with isosurfaces
- **Site Connectivity Networks**: 3D and 2D connectivity plots showing hopping pathways
- **Jump Statistics**: Histograms and correlation matrices for collective behavior
- **Displacement Analysis**: Individual and averaged displacement trajectories
- **Van Hove Plots**: Customizable correlation function visualization
- **Publication-Ready Figures**: High-quality plots with full customization control

## Installation

### Prerequisites
- Python 3.8 or higher
- MDAnalysis
- NumPy
- SciPy
- Matplotlib
- NetworkX
- tqdm
- scikit-image (optional, for advanced isosurface generation)
- seaborn (optional, for enhanced heatmaps)

### Install from Source
git clone https://github.com/yourusername/gphion.git
cd gphion
pip install -e .

text

### Install Dependencies
pip install -r requirements.txt

text

## Quick Start

### Basic Analysis Workflow

import os
from gphion import (
load_simulation_data,
CoordinateProcessor,
AmorphousSiteFinder,
SiteAnalyzerMDA,
JumpDiffusivityAnalyzer,
VanHovePlotter,
ConnectivityPlotter
)

1. Load simulation data
sim_data = load_simulation_data(
dump_file="trajectory.lammpstrj",
type_to_element={1: 'S', 2: 'F', 3: 'O', 4: 'N', 5: 'C', 6: 'Li'},
diff_elem='Li',
dt_ps=2.0
)

2. Process coordinates and calculate displacements
coord_processor = CoordinateProcessor()
sim_data = coord_processor.frac_and_disp(sim_data)

3. Discover hopping sites using TAOG
site_finder = AmorphousSiteFinder(sim_data, grid_resolution=100)
sites = site_finder.find_sites_from_density()

4. Analyze jumps between sites
jump_analyzer = SiteAnalyzerMDA(
sim_data.universe.select_atoms("element Li"), sites
)
jump_analyzer.run()

5. Calculate diffusivity
diffusivity_analyzer = JumpDiffusivityAnalyzer(
sim_data, jump_analyzer.results, sites
)
jump_diffusivity = diffusivity_analyzer.calculate_jump_properties()

6. Visualize connectivity
conn_plotter = ConnectivityPlotter()
conn_plotter.plot_3d_site_connectivity(
sites, jump_analyzer.results.all_trans
)

text

### Van Hove Analysis

from gphion import VanHoveAnalyzer, VanHovePlotter

Analyze Van Hove correlation functions
vh_analyzer = VanHoveAnalyzer(sim_data)
vh_results = vh_analyzer.run_complete_analysis()

Create custom plots
vh_plotter = VanHovePlotter()
vh_plotter.plot_van_hove_functions(
r_values=vh_results['r_values'],
self_results=vh_results['self_normalized'],
distinct_results=vh_results['distinct_normalized'],
time_lags=vh_results['time_lags'],
timestep=sim_data.time_step,
sigma=2.0, # Gaussian smoothing
xlim=(0, 10),
yscale='log'
)

text

## Package Structure

gphion/
├── init.py
├── core/
│ ├── init.py
│ ├── sim_data.py # Main data structure
│ ├── coordinate_processor.py # Coordinate processing
│ └── utils.py # Utility functions
├── analyzers/
│ ├── init.py
│ ├── site_finder.py # TAOG site discovery
│ ├── jump_analyzer.py # Jump detection
│ ├── diffusivity_analyzer.py # Diffusivity calculations
│ ├── collective_jump_analyzer.py # Collective behavior
│ ├── vibration_analyzer.py # Vibrational analysis
│ ├── rdf_analyzer.py # Radial distribution functions
│ ├── tracer_analyzer.py # Tracer properties
│ └── vanhove_analyzer.py # Van Hove functions
├── visualization/
│ ├── init.py
│ ├── plotting.py # 3D density plots
│ ├── vanhove_plotter.py # Van Hove visualization
│ ├── displacement_plotter.py # Displacement analysis
│ ├── collective_plotter.py # Collective jump plots
│ └── connectivity_plotter.py # Site connectivity
└── workflow/
├── init.py
└── main.py # Main workflow functions

text

## Documentation

### Key Classes

- **`SimData`**: Main data container for simulation information
- **`AmorphousSiteFinder`**: TAOG-based site discovery
- **`SiteAnalyzerMDA`**: Jump detection and analysis
- **`VanHoveAnalyzer`**: Van Hove correlation function calculation
- **`ConnectivityPlotter`**: Site connectivity visualization
- **`CollectiveJumpAnalyzer`**: Cooperative behavior analysis

### Analysis Types

1. **Site Discovery**: Uses 3D occupancy grids to find ion hopping sites
2. **Jump Analysis**: Detects and categorizes hopping events between sites
3. **Diffusivity**: Calculates both tracer (MSD-based) and jump diffusivity
4. **Collective Behavior**: Identifies spatially and temporally correlated jumps
5. **Connectivity**: Maps diffusion pathways as network graphs
6. **Van Hove Functions**: Analyzes particle displacement correlation functions

## Examples

See the `examples/` directory for complete analysis workflows:

- `basic_analysis.py`: Standard TAOG analysis workflow
- `van_hove_analysis.py`: Van Hove correlation function analysis
- `connectivity_analysis.py`: Site connectivity and pathway analysis
- `collective_jumps.py`: Collective jump behavior analysis

## Input File Formats

GPHIon supports various MD simulation output formats through MDAnalysis:
- LAMMPS dump files (`.lammpstrj`)
- XYZ files
- PDB files
- DCD trajectories
- And many others supported by MDAnalysis

## Output

The package generates:
- **Data files**: `.npy` arrays for site coordinates, density grids, Van Hove data
- **Analysis results**: Jump statistics, diffusivity values, connectivity matrices
- **Visualizations**: Publication-ready plots in PNG/PDF format
- **Summary reports**: Comprehensive analysis summaries

## Performance

GPHIon is optimized for:
- Large trajectory files (>GB)
- High-resolution density grids
- Long simulation times
- Multiple ion types

Memory usage scales with:
- Number of atoms × trajectory length
- Grid resolution³
- Number of discovered sites

## Citation

If you use GPHIon in your research, please cite:

@software{gphion2025,
title={GPHIon: Glassy Phase Ionic Conductor Analysis Package},
author={Your Name},
year={2025},
url={https://github.com/yourusername/gphion}
}

text

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup
git clone https://github.com/yourusername/gphion.git
cd gphion
pip install -e ".[dev]"

text

### Running Tests
pytest tests/

text

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- MDAnalysis team for the excellent molecular analysis framework
- Scientific Python community for the foundational packages
- Original TAOG method developers
- Contributors and users of the package

## Support

- **Documentation**: [https://gphion.readthedocs.io](https://gphion.readthedocs.io)
- **Issues**: [GitHub Issues](https://github.com/yourusername/gphion/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/gphion/discussions)
- **Email**: your.email@institution.edu

## Changelog

### Version 1.0.0 (2025-07-31)
- Initial release
- Complete TAOG analysis implementation
- Van Hove correlation function analysis
- Site connectivity visualization
- Collective jump analysis
- Comprehensive visualization suite

