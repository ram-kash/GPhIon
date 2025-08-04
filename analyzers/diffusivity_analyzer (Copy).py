"""Jump diffusivity calculation and analysis."""

import numpy as np
import matplotlib.pyplot as plt
from ..core.utils import calc_dist_sqrd_frac

class JumpDiffusivityAnalyzer:
    """Calculates jump diffusivity and analyzes jump distances from raw jump event data."""

    def __init__(self, sim_data, jump_analysis_results, site_coordinates_cart):
        """
        Args:
            sim_data (SimData): The main simulation data object.
            jump_analysis_results: The results object from the completed SiteAnalyzerMDA run.
            site_coordinates_cart (np.ndarray): The Cartesian coordinates of the discovered sites.
        """
        self.sim_data = sim_data
        self.jumps = jump_analysis_results.all_trans
        self.sites_cart = site_coordinates_cart
        print("\n--- Stage 4: Post-Processing Jumps for Diffusivity Analysis ---")

    def calculate_jump_properties(self, diffusion_dim=3, resolution=0.1, show_plot=True):
        """
        Execute the jump diffusivity calculation and generate the jump distance histogram.
        
        Args:
            diffusion_dim (int): The number of dimensions for diffusion (typically 3).
            resolution (float): The bin size in Angstroms for the jump distance histogram.
            show_plot (bool): If True, displays the jump distance histogram plot.
            
        Returns:
            float: The calculated jump diffusivity in m^2/s.
        """
        if self.jumps.size == 0:
            print("WARNING: No jumps were detected, cannot calculate jump diffusivity.")
            return 0.0

        # Setup for calculation
        nr_diffusing = self.sim_data.nr_diffusing
        total_sim_time = self.sim_data.nr_steps * self.sim_data.time_step
        lattice_matrix = self.sim_data.lattice.T
        inv_lattice_matrix = np.linalg.inv(lattice_matrix)
        ang2m2 = 1E-20  # Conversion from Angstrom^2 to meter^2

        # Convert Cartesian site coordinates to fractional for distance calculation
        sites_frac = np.dot(inv_lattice_matrix, self.sites_cart.T).T

        # Histogram setup
        max_dist = 10.0  # Max expected jump distance in Angstrom
        bins = np.arange(0, max_dist + resolution, resolution)
        jump_dist_hist = np.zeros(len(bins) - 1)
        sum_dist_sq = 0.0

        # Process each jump event
        for jump_event in self.jumps:
            # Get site indices (and convert from 1-based to 0-based)
            from_site_idx = int(jump_event[1]) - 1
            to_site_idx = int(jump_event[2]) - 1

            # Get fractional coordinates for the sites involved in the jump
            pos1_frac = sites_frac[from_site_idx]
            pos2_frac = sites_frac[to_site_idx]

            # Calculate squared distance with periodic boundary conditions
            dist_sq = calc_dist_sqrd_frac(pos1_frac, pos2_frac, lattice_matrix)
            sum_dist_sq += dist_sq

            # For the histogram
            distance = np.sqrt(dist_sq)
            hist_bin_idx = np.searchsorted(bins, distance, side='right') - 1
            if 0 <= hist_bin_idx < len(jump_dist_hist):
                jump_dist_hist[hist_bin_idx] += 1

        # Final diffusivity calculation
        # Formula: D = (1 / (2*d*N*t)) * sum(r_ij^2)
        denominator = (2 * diffusion_dim * nr_diffusing * total_sim_time)
        jump_diffusivity = (sum_dist_sq * ang2m2) / denominator

        print(f"Jump diffusivity calculated assuming {diffusion_dim}D diffusion.")
        print(f" - Total simulation time: {total_sim_time:.2e} s")
        print(f" - Number of diffusing atoms: {nr_diffusing}")
        print(f" - Resulting Jump Diffusivity (m^2/s): {jump_diffusivity:.4e}")

        # Plotting
        if show_plot:
            bin_centers = (bins[:-1] + bins[1:]) / 2
            plt.figure(figsize=(10, 6))
            plt.bar(bin_centers, jump_dist_hist, width=resolution, align='center', edgecolor='black')
            plt.title('Jump Distance Distribution')
            plt.xlabel('Jump Distance (Å)')
            plt.ylabel('Number of Jumps')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.show()

        return jump_diffusivity

