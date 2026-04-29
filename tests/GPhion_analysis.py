#!/usr/bin/env python
# coding: utf-8
"""
Comprehensive GPhIon Analysis Script
===================================
Optimized to eliminate duplicate plots and use appropriate plotting methods.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Determine script directory for robust path handling
script_dir = os.path.dirname(os.path.abspath(__file__))

# Add package path
sys.path.append(os.path.abspath(os.path.join(script_dir, '../')))

from gphion import (
    load_simulation_data,
    CoordinateProcessor,
    TracerPropertyAnalyzer,
    VibrationAnalyzer,
    AmorphousSiteFinder,
    JumpAnalyzer,
    JumpDiffusivityAnalyzer,
    CollectiveJumpAnalyzer,
    plot_taog_density,
    ActivationEnergyAnalyzer,
    CollectivePlotter,
    DisplacementPlotter,
    VanHoveAnalyzer,
    VanHovePlotter,
    RDFAnalyzer
)

# =============================================================================
# 1. PARAMETERS
# =============================================================================
print("=" * 80)
print("TAOG (Time-Averaged Occupancy Grid) Analysis")
print("=" * 80)

project_folder = script_dir
dump_filename = 'pure_400.lammpstrj'
dump_filepath = os.path.join(project_folder, dump_filename)
output_data_path = os.path.join(project_folder, "analysis_output")

os.makedirs(output_data_path, exist_ok=True)
if not os.path.exists(dump_filepath):
    raise FileNotFoundError(f"Trajectory file not found at '{dump_filepath}'")

temperature = 450.0
type_to_element = {1: 'S', 2: 'F', 3: 'O', 4: 'N', 5: 'C', 6: 'Li', 7: 'Co', 8: 'H'}
diffusing_element = 'Li'
timestep_ps = 2.0
density_grid_resolution = 100
diffusion_dimensionality = 3
ion_charge_z = 1.0
experimental_coll_dist = 10.0
experimental_coll_steps = None

# =============================================================================
# 2. LOAD SIMULATION DATA
# =============================================================================
print("\n🔄 Loading simulation data...")
sim_data_obj = load_simulation_data(
    dump_file=dump_filepath,
    type_to_element=type_to_element,
    diff_elem=diffusing_element,
    dt_ps=timestep_ps,
    temperature=temperature
)
print(f"✅ Loaded {sim_data_obj.nr_steps} frames; "
      f"{sim_data_obj.nr_diffusing}/{sim_data_obj.nr_atoms} diffusing atoms.")

# =============================================================================
# 3. COORDINATE & DISPLACEMENT
# =============================================================================
print("\n🔄 Processing coordinates...")
coord_processor = CoordinateProcessor()
sim_data_obj = coord_processor.frac_and_disp(sim_data_obj)
print("✅ Coordinate processing complete")

# =============================================================================
# 4. TRACER PROPERTIES
# =============================================================================
print("\n🔄 Calculating tracer properties...")
tracer_analyzer = TracerPropertyAnalyzer(sim_data_obj)
tracer_results = tracer_analyzer.calculate_tracer_properties(
    diffusion_dim=diffusion_dimensionality, 
    z_ion=ion_charge_z
)
print("✅ Tracer property analysis complete")

# =============================================================================
# 5. VIBRATION ANALYSIS
# =============================================================================
print("\n🔄 Analyzing vibrational properties...")
vib_analyzer = VibrationAnalyzer()
attempt_freq, vibration_amp, std_freq = vib_analyzer.vibration_properties(
    sim_data_obj, show_pics=True, save_data=True, output_dir=output_data_path
)

sim_data_obj.attempt_freq = attempt_freq
sim_data_obj.vibration_amp = vibration_amp
sim_data_obj.std_attempt_freq = std_freq
print("✅ Vibrational analysis complete")

# =============================================================================
# 6. SITE DISCOVERY
# =============================================================================
print("\n🔄 Discovering sites...")
site_finder = AmorphousSiteFinder(
    sim_data=sim_data_obj, 
    grid_resolution=density_grid_resolution, 
    save_path=output_data_path
)
density_grid, edges = site_finder._build_occupancy_grid()
labeled_peaks, num_features, peak_indices = site_finder._find_grid_peaks(
    density_grid, return_labels=True
)
discovered_sites_cart = site_finder._convert_indices_to_coords(peak_indices, edges)

if discovered_sites_cart.size == 0:
    raise RuntimeError("❌ No sites discovered.")

site_radii = site_finder.get_site_radii(
    labeled_peaks, num_features, edges, discovered_sites_cart, pct=95
)
np.save(os.path.join(output_data_path, 'discovered_sites_cart.npy'), discovered_sites_cart)
np.save(os.path.join(output_data_path, 'site_radii.npy'), site_radii)
print(f"✅ {len(discovered_sites_cart)} sites found; mean radius = {site_radii.mean():.3f} Å")

# TAOG Visualization
try:
    plot_taog_density(
        density_grid, edges, threshold_ratio=0.1,
        output_file=os.path.join(output_data_path, 'taog_density_plot.png'),
        save_payload_dir=output_data_path
    )
    print("✅ TAOG visualization saved")
except Exception as e:
    print(f"⚠️ TAOG visualization failed: {e}")

# =============================================================================
# 7. JUMP ANALYSIS
# =============================================================================
print("\n🔄 Running jump analysis...")
diffusing_atoms_group = sim_data_obj.universe.select_atoms(f"element {diffusing_element}")
jump_analysis = JumpAnalyzer(diffusing_atoms_group, discovered_sites_cart)
jump_analysis.site_radius = 2 * site_radii.max()
jump_analysis.run(verbose=True)
num_jumps_found = len(jump_analysis.results.all_trans)
print(f"✅ {num_jumps_found} jumps detected")

# =============================================================================
# 8. JUMP DIFFUSIVITY
# =============================================================================
print("\n🔄 Calculating jump diffusivity...")
jump_diffusivity_val = 0.0
if num_jumps_found > 0:
    diffusivity_analyzer = JumpDiffusivityAnalyzer(
        sim_data_obj, jump_analysis.results, discovered_sites_cart
    )
    # Note: show_plot=False to avoid duplicate plots
    jump_diffusivity_val = diffusivity_analyzer.calculate_jump_properties(
        show_plot=False, save_data=True, output_dir=output_data_path
    )
    diffusivity_analyzer.save_jump_summary(jump_diffusivity_val, output_data_path)
    print("✅ Jump diffusivity calculation complete")
else:
    print("⚠️ No jumps found - skipping")

# =============================================================================
# 9. COLLECTIVE JUMPS ANALYSIS
# =============================================================================
print("\n🔄 Analyzing collective jump behavior...")
collective_results = None
if num_jumps_found > 0:
    collective_analyzer = CollectiveJumpAnalyzer(
        sim_data=sim_data_obj,
        jump_analysis_results=jump_analysis.results,
        site_coordinates_cart=discovered_sites_cart,
        coll_dist=experimental_coll_dist,
        coll_steps_manual=experimental_coll_steps
    )
    collective_results = collective_analyzer.analyze()
    print("✅ Collective jump analysis complete")
else:
    print("⚠️ No jumps found - skipping collective jump analysis")

# =============================================================================
# 10. ACTIVATION ENERGY ANALYSIS
# =============================================================================
print("\n⚡ Calculating activation energies...")
site_energies = {}
if num_jumps_found > 0:
    sim_data_obj.temperature = temperature
    activation_analyzer = ActivationEnergyAnalyzer(sim_data_obj)
    total_time = sim_data_obj.total_time
    eff_freq = num_jumps_found / (sim_data_obj.nr_diffusing * total_time)
    print(f" Effective freq: {eff_freq:.2e} Hz | Attempt freq: {sim_data_obj.attempt_freq:.2e} Hz")
    
    if eff_freq / sim_data_obj.attempt_freq < 1.0:
        try:
            overall_activation = activation_analyzer.calculate_activation_energy_from_frequencies(
                effective_jump_freq=eff_freq, 
                attempt_freq=sim_data_obj.attempt_freq, 
                temperature=temperature
            )
            print(f"✅ Overall Activation Energy: {overall_activation['activation_energy_eV']:.3f} eV")
        except Exception as e:
            print(f"⚠️ Overall activation energy failed: {e}")

    try:
        site_energies = activation_analyzer.calculate_activation_energies_with_occupancy(
            jump_analysis, discovered_sites_cart, temperature
        )
        if site_energies:
            activation_analyzer.compare_activation_energies(
                site_energies,
                output_file=f"{output_data_path}/activation_energies_comparison.png"
            )
            activation_analyzer.generate_activation_energy_report(
                site_energies,
                output_file=f"{output_data_path}/activation_energy_report.txt"
            )
            print(f"✅ Site-specific activation energies calculated")
    except Exception as e:
        print(f"⚠️ Site-specific activation energy failed: {e}")
else:
    print("⚠️ No jumps found - skipping")

# Compute average activation energy
activation_values = [v["activation_energy_eV"] for v in site_energies.values() 
                     if "activation_energy_eV" in v]
if activation_values:
    avg_activation = sum(activation_values) / len(activation_values)
    print(f"🔹 Average Activation Energy: {avg_activation:.3f} eV")
else:
    print("⚠️ No valid activation energy values found for averaging.")

# =============================================================================
# 11. DATA SAVING
# =============================================================================
print("\n💾 Saving jump and occupancy data...")
jump_output_path = os.path.join(output_data_path, "jump_output")
os.makedirs(jump_output_path, exist_ok=True)

if num_jumps_found > 0:
    np.savetxt(
        f'{jump_output_path}/site_occupancies.txt',
        np.column_stack([
            np.arange(len(jump_analysis.results.site_occupancies)),
            jump_analysis.results.site_occupancies
        ]),
        header='Site_ID Average_Occupancy(atoms)', fmt='%d %.6f'
    )
    np.savetxt(
        f'{jump_output_path}/occupancy_factors.txt',
        np.column_stack([
            np.arange(len(jump_analysis.results.average_occupancy_factors)),
            jump_analysis.results.average_occupancy_factors
        ]),
        header='Site_ID Occupancy_Factor', fmt='%d %.6f'
    )
    np.savetxt(
        f'{jump_output_path}/jump_events.txt',
        jump_analysis.results.all_trans,
        header='AtomIndex FromSite ToSite Time(ps)', fmt='%d %d %d %.6f'
    )
    print("✅ Jump data saved")

# =============================================================================
# 12. COLLECTIVE PLOTTING (Single comprehensive call)
# =============================================================================
if collective_results:
    print("\n📊 Generating all collective jump analysis plots...")
    coll_plotter = CollectivePlotter()
    coll_plotter.plot_all_collective_analyses(
        collective_results,
        jump_analysis.results.all_trans,
        timestep=sim_data_obj.time_step,
        output_dir=output_data_path
    )
    print(f"✅ Collective plots saved to: {output_data_path}")
else:
    print("⚠️ No collective results - skipping collective plotting")

# =============================================================================
# 13. DISPLACEMENT ANALYSIS (Save data files)
# =============================================================================
print("\n🔄 Displacement analysis...")
if num_jumps_found > 0:
    disp_plotter = DisplacementPlotter()
    disp_plotter.plot_individual_displacements(
        sim_data_obj,
        timestep=sim_data_obj.time_step,
        max_atoms=50,
        output_file=f"{output_data_path}/individual_displacements.png",
        data_output_dir=output_data_path  # Enable data saving
    )
    disp_plotter.plot_displacement_histogram(
        sim_data_obj,
        bins=60,
        output_file=f"{output_data_path}/displacement_histogram.png",
        data_output_dir=output_data_path  # Enable data saving
    )
    print("✅ Displacement plots and data saved")

# =============================================================================
# 14. VAN HOVE ANALYSIS
# =============================================================================
print("\n📊 Van Hove analysis...")
# Adjust time lags based on trajectory length
if sim_data_obj.nr_steps > 1000:
    time_lags = [10, 100, 300, 500, 700, 900]
else:
    time_lags = [10, 50, 100, 200]

vanhove_analyzer = VanHoveAnalyzer(sim_data_obj, diffusing_element=diffusing_element)
vanhove_results = vanhove_analyzer.run_complete_analysis(
    time_lags=time_lags,
    max_r=None,
    n_bins=2000,
    output_dir=output_data_path
)

vh_plotter = VanHovePlotter()
vh_plotter.plot_van_hove_functions(
    r_values=vanhove_results['r_values'],
    self_results=vanhove_results['self_normalized'],
    distinct_results=vanhove_results['distinct_normalized'],
    time_lags=vanhove_results['time_lags'],
    timestep=sim_data_obj.time_step,
    diffusing_element=diffusing_element,
    sigma=2.0,
    xlim=(0, 10),
    yscale='log',
    output_dir=output_data_path
)
print("✅ Van Hove analysis complete")

# =============================================================================
# 15. FINAL SUMMARY
# =============================================================================
print("\n" + "="*80)
print("📊 FINAL COMPREHENSIVE ANALYSIS RESULTS")
print("="*80)

print(f"\n📈 BASIC STATISTICS:")
print(f"   Sites Found: {len(discovered_sites_cart)}")
print(f"   Jumps Detected: {num_jumps_found}")
print(f"   Simulation Time: {sim_data_obj.total_time:.2e} s")

print(f"\n🔬 DIFFUSION PROPERTIES:")
print(f"   Tracer Diffusivity (MSD): {tracer_results['tracer_diffusivity']:.4e} m²/s")
print(f"   Jump Diffusivity: {jump_diffusivity_val:.4e} m²/s")
print(f"   Tracer Conductivity: {tracer_results['tracer_conductivity']:.4e} S/m")

if collective_results:
    num_coll_pairs = len(collective_results.get('collective_pairs', []))
    print(f"\n🤝 COLLECTIVE BEHAVIOR:")
    print(f"   Collective Jump Pairs: {num_coll_pairs}")

print(f"\n📁 All output files saved to: {output_data_path}")
print("\n✅ COMPLETE COMPREHENSIVE ANALYSIS FINISHED!")
print("="*80)
