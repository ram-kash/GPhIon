"""Main workflow and data loading utilities."""

import numpy as np
import MDAnalysis as mda
from ..core.sim_data import SimData

def load_simulation_data(dump_file, type_to_element, diff_elem, dt_ps, temperature=None):
    """Read the trajectory and prepare the initial SimData object."""
    print("--- Stage 1: Reading LAMMPS Trajectory ---")
    
    sim_data = SimData()
    sim_data.time_step = dt_ps * 1e-12
    sim_data.diff_elem = diff_elem
    sim_data.temperature = temperature if temperature is not None else 300.0  # Default to 300K if not provided

    u = mda.Universe(dump_file, format="LAMMPSDUMP", dt=dt_ps)
    atom_types = u.atoms.types.astype(int)
    elements = np.array([type_to_element.get(t, 'X') for t in atom_types])
    u.add_TopologyAttr('element', elements)

    sim_data.universe = u
    sim_data.nr_atoms = len(u.atoms)
    sim_data.lattice = u.trajectory.ts.triclinic_dimensions

    diff_atoms = u.select_atoms(f"element {sim_data.diff_elem}")
    sim_data.nr_diffusing = len(diff_atoms)
    sim_data.diffusing_atom = np.zeros(sim_data.nr_atoms, dtype=bool)
    sim_data.diffusing_atom[diff_atoms.indices] = True

    sim_data.nr_steps = len(u.trajectory)
    sim_data.cart_pos = np.zeros((3, sim_data.nr_atoms, sim_data.nr_steps))

    for i, ts in enumerate(u.trajectory):
        sim_data.cart_pos[:, :, i] = ts.positions.T

    sim_data.total_time = sim_data.nr_steps * sim_data.time_step

    print(f"DEBUG: Trajectory loaded with {sim_data.nr_steps} frames.")
    return sim_data


def main_analysis_workflow():
    """Complete analysis workflow example."""
    import os
    import traceback
    from ..core.coordinate_processor import CoordinateProcessor
    from ..analyzers.tracer_analyzer import TracerPropertyAnalyzer
    from ..analyzers.vibration_analyzer import VibrationAnalyzer
    from ..analyzers.site_finder import AmorphousSiteFinder
    from ..analyzers.jump_analyzer import JumpAnalyzer
    from ..analyzers.diffusivity_analyzer import JumpDiffusivityAnalyzer
    from ..analyzers.collective_jump_analyzer import CollectiveJumpAnalyzer

    try:
        # Define Parameters
        project_folder = '/media/ram/ext_disk/proj_mof/dp_data/analysis/py/'
        dump_filename = '400.lammpstrj'
        dump_filepath = os.path.join(project_folder, dump_filename)
        
        type_to_element = {1: 'S', 2: 'F', 3: 'O', 4: 'N', 5: 'C', 6: 'Li', 7: 'Co', 8: 'H'}
        diffusing_element = 'Li'
        timestep_ps = 2.0
        density_grid_resolution = 100
        output_data_path = os.path.join(project_folder, "analysis_output")
        
        # Parameters for analysis
        diffusion_dimensionality = 3
        ion_charge_z = 1.0
        experimental_coll_dist = 6.0
        experimental_coll_steps = None

        if not os.path.exists(dump_filepath):
            raise FileNotFoundError(f"CRITICAL ERROR: Trajectory file not found at '{dump_filepath}'")

        # Load Trajectory Data
        sim_data_obj = load_simulation_data(
            dump_file=dump_filepath,
            type_to_element=type_to_element,
            diff_elem=diffusing_element,
            dt_ps=timestep_ps
        )

        # Calculate Displacement
        coord_processor = CoordinateProcessor()
        sim_data_obj = coord_processor.frac_and_disp(sim_data_obj)

        # Calculate Tracer Properties (from MSD)
        tracer_analyzer = TracerPropertyAnalyzer(sim_data_obj)
        tracer_results = tracer_analyzer.calculate_tracer_properties(
            diffusion_dim=diffusion_dimensionality,
            z_ion=ion_charge_z
        )

        # Calculate Jump Frequency
        vib_analyzer = VibrationAnalyzer()
        attempt_freq, vibration_amp, std_freq = vib_analyzer.vibration_properties(
            sim_data_obj, show_pics=True
        )

        sim_data_obj.attempt_freq = attempt_freq
        sim_data_obj.vibration_amp = vibration_amp
        sim_data_obj.std_attempt_freq = std_freq

        # Discover Sites from Density Grid
        site_finder = AmorphousSiteFinder(
            sim_data=sim_data_obj,
            grid_resolution=density_grid_resolution,
            save_path=output_data_path
        )

        discovered_sites_cartesian = site_finder.find_sites_from_density()

        if discovered_sites_cartesian.size == 0:
            raise RuntimeError("No sites were discovered.")

        # Analyze Jumps Between Discovered Sites
        jump_analysis = JumpAnalyzer(
            diffusing_atom_group=sim_data_obj.universe.select_atoms(f"element {diffusing_element}"),
            site_coordinates_cart=discovered_sites_cartesian
        )

        jump_analysis.run(verbose=True)

        # Calculate Jump Diffusivity
        jump_diffusivity_val = 0.0
        if len(jump_analysis.results.all_trans) > 0:
            diffusivity_analyzer = JumpDiffusivityAnalyzer(
                sim_data=sim_data_obj,
                jump_analysis_results=jump_analysis.results,
                site_coordinates_cart=discovered_sites_cartesian
            )

            jump_diffusivity_val = diffusivity_analyzer.calculate_jump_properties(show_plot=True)

        # Analyze Collective Jumps
        collective_results = None
        if len(jump_analysis.results.all_trans) > 0:
            collective_analyzer = CollectiveJumpAnalyzer(
                sim_data=sim_data_obj,
                jump_analysis_results=jump_analysis.results,
                site_coordinates_cart=discovered_sites_cartesian,
                coll_dist=experimental_coll_dist,
                coll_steps_manual=experimental_coll_steps
            )

            collective_results = collective_analyzer.analyze()

        # Final Summary
        num_sites_found = len(discovered_sites_cartesian)
        num_jumps_found = len(jump_analysis.results.all_trans)

        print("\n\n--- ANALYSIS COMPLETE ---")
        print(f"Number of Jump Sites Found: {num_sites_found}")
        print(f"Tracer Diffusivity (MSD) (m^2/s): {tracer_results['tracer_diffusivity']:.4e}")
        print(f"Jump Diffusivity (m^2/s): {jump_diffusivity_val:.4e}")
        print(f"Tracer Conductivity (S/m): {tracer_results['tracer_conductivity']:.4f}")
        print(f"Jump Frequency (Hz): {attempt_freq:.2e} (± {std_freq:.1e})")
        print(f"Vibration Amplitude (Å): {vibration_amp:.4f}")

        if collective_results:
            num_coll_pairs = len(collective_results.get('collective_pairs', []))
            num_multi_jumps = len(collective_results.get('multi_collective_jumps', []))
            print(f"Collective Jump Pairs Found: {num_coll_pairs}")
            print(f"Multi-Correlated Jumps: {num_multi_jumps}")
        else:
            print("Collective Jump Pairs Found: 0 (Analysis not run or no jumps)")
            print("Multi-Correlated Jumps: 0 (Analysis not run or no jumps)")

        if num_jumps_found > 0:
            print(f"\n - Total Jumps Detected: {num_jumps_found}")
            print(" - Example Jump Events (atom_idx_in_group, from_site_idx, to_site_idx, frame_#):")
            print(jump_analysis.results.all_trans[:5])

    except Exception as e:
        print("\n--- AN ERROR OCCURRED ---")
        traceback.print_exc()


if __name__ == '__main__':
    main_analysis_workflow()

