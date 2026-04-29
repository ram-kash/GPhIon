# GPhion Analysis Script - Output Files Documentation

## Complete List of Generated Outputs

### Data Files Location
`tests/analysis_output/`

---

## 1. Vibration Analysis
**Directory**: `tests/analysis_output/`

### Data Files:
- `vibrational_amplitudes.dat` - All vibrational amplitudes from all atoms
- `per_atom_frequencies.dat` - Mean frequency for each atom
- `frequency_spectrum.dat` - Frequency axis and spectrum values
- `vibration_summary.dat` - Summary statistics (attempt_freq, std, vibration_amp)
- `amplitude_histogram_data.dat` - Histogram bins and counts
- `amplitude_gaussian_fit.dat` - Gaussian fit parameters

### Plot Files:
- **`vibrational_amplitudes.png`** - Histogram of vibrational amplitudes with Gaussian fit
- **`frequency_spectrum.png`** - Frequency spectrum showing attempt frequency

---

## 2. Site Discovery (TAOG)
**Directory**: `tests/analysis_output/`

### Data Files:
- `discovered_sites_cart.npy` - Cartesian coordinates of all discovered sites
- `site_radii.npy` - Radii of each site
- `occupancy_grid_edges.npy` - Grid edge coordinates

### Plot Files:
- **`taog_density_plot.png`** - 3D visualization of TAOG density

---

## 3. Jump Analysis
**Directory**: `tests/analysis_output/jump_output/`

### Data Files:
- `site_occupancies.txt` - Average occupancy per site
- `occupancy_factors.txt` - Occupancy factor per site
- `jump_events.txt` - All jump events (atom, from_site, to_site, time)

**Directory**: `tests/analysis_output/`
- `jump_distances.dat` - All jump distances
- `detailed_jump_data.dat` - Detailed jump information
- `jump_distance_histogram.dat` - Histogram of jump distances
- `jump_distance_statistics.dat` - Statistical summary
- `histogram_bin_edges.dat` - Bin edges for histogram
- `jump_distances_vs_time.dat` - Temporal jump distance data

---

## 4. Activation Energy
**Directory**: `tests/analysis_output/`

### Data Files:
- `activation_energy_report.txt` - Detailed text report of activation energies

### Plot Files:
- **`activation_energies_comparison.png`** - Comparison plot of site-specific activation energies

---

## 5. Collective Jump Analysis
**Directory**: `tests/analysis_output/`

### Data Files:
- `jump_histogram_vs_time_data.dat` - Jump counts vs time
- `collective_jump_matrix_data.dat` - Jump type correlation matrix
- `jump_types_list.txt` - List of jump type labels
- `multi_correlation_statistics.dat` - Statistics on multi-correlated jumps
- `collective_summary_statistics.dat` - Statistical summary

### Plot Files:
- **`jump_histogram_vs_time.png`** - Histogram of jumps over time
- **`collective_jump_matrix.png`** - Jump type correlation matrix
- **`collective_jump_heatmap.png`** - Heatmap version of correlation
- **`collective_summary.png`** - Summary statistics visualization

---

## 6. Displacement Analysis
**Directory**: `tests/analysis_output/individual_displacements/`
- `displacements_matrix.txt` - Time series of displacement for each atom (48 atoms)

**Directory**: `tests/analysis_output/displacement_histogram/`
- `histogram_data.txt` - Histogram of final displacements (bin edges and counts)

### Plot Files:
**Directory**: `tests/analysis_output/`
- **`individual_displacements.png`** - Displacement trajectories (up to 50 atoms)
- **`displacement_histogram.png`** - Histogram of final displacements

---

## 7. Van Hove Analysis
**Directory**: `tests/analysis_output/`

### Data Files:
- `g_self_rt.dat` - Self-part of Van Hove correlation function
- `g_distinct_rt.dat` - Distinct-part of Van Hove correlation function

### Plot Files:
- **`van_hove_self.png`** - Self-part Van Hove function plots
- **`van_hove_distinct.png`** - Distinct-part Van Hove function plots

---

## Total Output Summary

### Plot Files (12 plots):
1. vibrational_amplitudes.png
2. frequency_spectrum.png
3. taog_density_plot.png
4. activation_energies_comparison.png
5. jump_histogram_vs_time.png
6. collective_jump_matrix.png
7. collective_jump_heatmap.png
8. collective_summary.png
9. individual_displacements.png
10. displacement_histogram.png
11. van_hove_self.png
12. van_hove_distinct.png

### Data Files:
- **Vibration**: 6 data files
- **Sites**: 3 data files (.npy)
- **Jumps**: 9 data files
- **Activation Energy**: 1 report file
- **Collective**: 5 data files
- **Displacement**: 2 data files
- **Van Hove**: 2 data files

**Total**: 28+ data files + 12 publication-quality plots

---

## Running the Script

```bash
cd /home/ram/Music/GPhIon
python3 tests/GPhion_analysis.py
```

All outputs will be saved to: `tests/analysis_output/`
