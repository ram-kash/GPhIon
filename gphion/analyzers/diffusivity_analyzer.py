"""Jump diffusivity calculation and analysis."""

import numpy as np
import matplotlib.pyplot as plt
import os
from ..core.utils import calc_dist_sqrd_frac
from ..core import AnalysisBase

class JumpDiffusivityAnalyzer(AnalysisBase):
    """Calculates jump diffusivity and analyzes jump distances from raw jump event data."""

    def __init__(self, sim_data, jump_analysis_results, site_coordinates_cart, *, show_plot: bool = True, save_data: bool = False, output_dir: str | None = None, save_payload_dir: str | None = None):
        """
        Args:
            sim_data (SimData): The main simulation data object.
            jump_analysis_results: The results object from the completed SiteAnalyzerMDA run.
            site_coordinates_cart (np.ndarray): The Cartesian coordinates of the discovered sites.
        """
        super().__init__(show_plot=show_plot, save_data=save_data, output_dir=output_dir, save_payload_dir=save_payload_dir)
        self.sim_data = sim_data
        self.jumps = jump_analysis_results.all_trans
        self.sites_cart = site_coordinates_cart
        print("\n--- Stage 4: Post-Processing Jumps for Diffusivity Analysis ---")

    def calculate_jump_properties(self, diffusion_dim=3, resolution=0.1, show_plot=None, save_data=None, output_dir=None):
        """
        Execute the jump diffusivity calculation and generate the jump distance histogram.
        
        Args:
            diffusion_dim (int): The number of dimensions for diffusion (typically 3).
            resolution (float): The bin size in Angstroms for the jump distance histogram.
            show_plot (bool): If True, displays the jump distance histogram plot.
            save_data (bool): If True, save jump distance data to files.
            output_dir (str, optional): Directory to save data and plots.
            
        Returns:
            float: The calculated jump diffusivity in m^2/s.
        """
        if self.jumps.size == 0:
            print("WARNING: No jumps were detected, cannot calculate jump diffusivity.")
            return 0.0

        # Setup for calculation
        nr_diffusing = self.sim_data.nr_diffusing
        total_sim_time = self.sim_data.nr_steps * self.sim_data.time_step
        lattice_matrix = self.sim_data.lattice
        inv_lattice_matrix = np.linalg.inv(lattice_matrix)
        ang2m2 = 1E-20  # Conversion from Angstrom^2 to meter^2

        # Convert Cartesian site coordinates to fractional for distance calculation
        sites_frac = np.dot(inv_lattice_matrix, self.sites_cart.T).T

        # Histogram setup
        max_dist = 10.0  # Max expected jump distance in Angstrom
        bins = np.arange(0, max_dist + resolution, resolution)
        jump_dist_hist = np.zeros(len(bins) - 1)
        sum_dist_sq = 0.0

        # Store individual jump data for saving
        jump_distances = []
        jump_info = []  # Store detailed jump information

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

            # Calculate actual distance
            distance = np.sqrt(dist_sq)
            jump_distances.append(distance)
            
            # Store detailed jump information
            jump_info.append({
                'atom_id': int(jump_event[0]),
                'from_site': int(jump_event[1]),
                'to_site': int(jump_event[2]),
                'time': jump_event[3],
                'distance': distance,
                'distance_squared': dist_sq
            })

            # For the histogram
            hist_bin_idx = np.searchsorted(bins, distance, side='right') - 1
            if 0 <= hist_bin_idx < len(jump_dist_hist):
                jump_dist_hist[hist_bin_idx] += 1

        # Convert to numpy array for easier handling
        jump_distances = np.array(jump_distances)

        # Final diffusivity calculation
        # Formula: D = (1 / (2*d*N*t)) * sum(r_ij^2)
        denominator = (2 * diffusion_dim * nr_diffusing * total_sim_time)
        jump_diffusivity = (sum_dist_sq * ang2m2) / denominator

        print(f"Jump diffusivity calculated assuming {diffusion_dim}D diffusion.")
        print(f" - Total simulation time: {total_sim_time:.2e} s")
        print(f" - Number of diffusing atoms: {nr_diffusing}")
        print(f" - Resulting Jump Diffusivity (m^2/s): {jump_diffusivity:.4e}")

        # Resolve effective flags/dirs from overrides or instance defaults
        eff_show_plot = self.show_plot if show_plot is None else show_plot
        eff_save_data = self.save_data if save_data is None else save_data
        eff_output_dir = output_dir or self.output_dir
        self.output_dir = eff_output_dir  # Ensure persist_plot picks this up

        # Save data if requested and always persist plot data for reproducibility
        if eff_save_data:
            self._save_jump_distance_data(
                jump_distances, jump_info, bins, jump_dist_hist, resolution,
                jump_diffusivity, total_sim_time, nr_diffusing, eff_output_dir
            )
        # Always save plot payload to allow later re-plotting
        try:
            bin_centers = (bins[:-1] + bins[1:]) / 2
            self.persist_plot(
                name="jump_distance_distribution",
                payload={
                    "bin_centers": bin_centers,
                    "hist": jump_dist_hist,
                    "bins": bins,
                    "jump_distances": jump_distances,
                    "stats": {
                        "mean": float(np.mean(jump_distances)),
                        "std": float(np.std(jump_distances)),
                        "n": int(len(jump_distances)),
                    },
                    "params": {
                        "resolution": float(resolution),
                        "diffusion_dim": int(diffusion_dim),
                        "nr_diffusing": int(nr_diffusing),
                        "total_sim_time": float(total_sim_time),
                    },
                },
            )
        except Exception as e:
            print(f"WARNING: Failed to save plot data payload: {e})")

        # Plotting
        if eff_show_plot:
            bin_centers = (bins[:-1] + bins[1:]) / 2
            plt.figure(figsize=(10, 6))
            plt.bar(bin_centers, jump_dist_hist, width=resolution, align='center', 
                   edgecolor='black', alpha=0.7, color='skyblue')
            plt.title('Jump Distance Distribution')
            plt.xlabel('Jump Distance (Å)')
            plt.ylabel('Number of Jumps')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Add statistics to plot
            mean_dist = np.mean(jump_distances)
            std_dist = np.std(jump_distances)
            plt.axvline(mean_dist, color='red', linestyle='--', linewidth=2, 
                       label=f'Mean: {mean_dist:.2f} Å')
            plt.legend()
            
            # Add text box with statistics
            stats_text = f'Total jumps: {len(jump_distances)}\nMean: {mean_dist:.2f} Å\nStd: {std_dist:.2f} Å'
            plt.text(0.98, 0.98, stats_text, transform=plt.gca().transAxes, 
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            if eff_output_dir:
                plt.savefig(os.path.join(eff_output_dir, "jump_distance_distribution.png"), 
                           dpi=300, bbox_inches='tight')
            plt.show()

        return jump_diffusivity

    def _save_jump_distance_data(self, jump_distances, jump_info, bins, histogram, 
                               resolution, jump_diffusivity, total_sim_time, 
                               nr_diffusing, output_dir=None):
        """
        Save jump distance analysis data to files.
        
        Args:
            jump_distances (np.array): Array of all jump distances
            jump_info (list): Detailed information for each jump
            bins (np.array): Histogram bin edges
            histogram (np.array): Histogram counts
            resolution (float): Bin resolution
            jump_diffusivity (float): Calculated jump diffusivity
            total_sim_time (float): Total simulation time
            nr_diffusing (int): Number of diffusing atoms
            output_dir (str, optional): Output directory
        """
        if output_dir is None:
            output_dir = "."
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Save raw jump distance data
        distance_file = os.path.join(output_dir, "jump_distances.dat")
        np.savetxt(distance_file, jump_distances,
                  header="Jump distances (Angstrom) for all detected jumps",
                  fmt="%.6f")
        print(f"Saved jump distances: {distance_file}")
        
        # 2. Save detailed jump information
        detailed_file = os.path.join(output_dir, "detailed_jump_data.dat")
        with open(detailed_file, 'w') as f:
            f.write("# Detailed Jump Information\n")
            f.write("# Atom_ID  From_Site  To_Site  Time  Distance(Angstrom)  Distance_Squared(Angstrom^2)\n")
            for jump in jump_info:
                f.write(f"{jump['atom_id']:8d}  {jump['from_site']:9d}  {jump['to_site']:7d}  "
                       f"{jump['time']:8.2f}  {jump['distance']:17.6f}  {jump['distance_squared']:26.6e}\n")
        print(f"Saved detailed jump data: {detailed_file}")
        
        # 3. Save histogram data
        histogram_file = os.path.join(output_dir, "jump_distance_histogram.dat")
        bin_centers = (bins[:-1] + bins[1:]) / 2
        hist_data = np.column_stack([bin_centers, histogram])
        np.savetxt(histogram_file, hist_data,
                  header="Distance_Center(Angstrom)  Count",
                  fmt="%.6f  %d")
        print(f"Saved histogram data: {histogram_file}")
        
        # 4. Save distribution statistics
        stats_file = os.path.join(output_dir, "jump_distance_statistics.dat")
        mean_dist = np.mean(jump_distances)
        std_dist = np.std(jump_distances)
        median_dist = np.median(jump_distances)
        min_dist = np.min(jump_distances)
        max_dist = np.max(jump_distances)
        
        with open(stats_file, 'w') as f:
            f.write("# Jump Distance Distribution Statistics\n")
            f.write("# =====================================\n")
            f.write(f"# Analysis timestamp: {np.datetime64('now')}\n")
            f.write("#\n")
            f.write("# DISTANCE STATISTICS:\n")
            f.write(f"Mean_Distance(Angstrom): {mean_dist:.6f}\n")
            f.write(f"Std_Distance(Angstrom): {std_dist:.6f}\n")
            f.write(f"Median_Distance(Angstrom): {median_dist:.6f}\n")
            f.write(f"Min_Distance(Angstrom): {min_dist:.6f}\n")
            f.write(f"Max_Distance(Angstrom): {max_dist:.6f}\n")
            f.write("#\n")
            f.write("# DIFFUSIVITY RESULTS:\n")
            f.write(f"Jump_Diffusivity(m^2/s): {jump_diffusivity:.6e}\n")
            f.write(f"Total_Simulation_Time(s): {total_sim_time:.6e}\n")
            f.write(f"Number_Diffusing_Atoms: {nr_diffusing}\n")
            f.write("#\n")
            f.write("# HISTOGRAM PARAMETERS:\n")
            f.write(f"Bin_Resolution(Angstrom): {resolution:.6f}\n")
            f.write(f"Number_of_Bins: {len(histogram)}\n")
            f.write(f"Total_Jumps: {len(jump_distances)}\n")
            f.write("#\n")
            f.write("# PERCENTILES:\n")
            percentiles = [10, 25, 50, 75, 90, 95, 99]
            for p in percentiles:
                value = np.percentile(jump_distances, p)
                f.write(f"Percentile_{p:02d}(Angstrom): {value:.6f}\n")
        print(f"Saved statistics: {stats_file}")
        
        # 5. Save histogram bin edges for exact reconstruction
        bins_file = os.path.join(output_dir, "histogram_bin_edges.dat")
        np.savetxt(bins_file, bins,
                  header="Histogram bin edges (Angstrom)",
                  fmt="%.6f")
        print(f"Saved bin edges: {bins_file}")
        
        # 6. Save jump distance vs time data for temporal analysis
        temporal_file = os.path.join(output_dir, "jump_distances_vs_time.dat")
        temporal_data = np.column_stack([
            [jump['time'] for jump in jump_info],
            [jump['distance'] for jump in jump_info]
        ])
        np.savetxt(temporal_file, temporal_data,
                  header="Time  Distance(Angstrom)",
                  fmt="%.2f  %.6f")
        print(f"Saved temporal data: {temporal_file}")

    def save_jump_summary(self, jump_diffusivity, output_dir=None):
        """
        Save a comprehensive summary of jump analysis results.
        
        Args:
            jump_diffusivity (float): Calculated jump diffusivity
            output_dir (str, optional): Output directory
        """
        if output_dir is None:
            output_dir = "."
            
        os.makedirs(output_dir, exist_ok=True)
        
        summary_file = os.path.join(output_dir, "jump_analysis_summary.dat")
        
        with open(summary_file, 'w') as f:
            f.write("# Jump Diffusivity Analysis Summary\n")
            f.write("# ==================================\n")
            f.write(f"# Analysis timestamp: {np.datetime64('now')}\n")
            f.write("#\n")
            f.write("# SIMULATION PARAMETERS:\n")
            f.write(f"Total_Steps: {self.sim_data.nr_steps}\n")
            f.write(f"Time_Step(s): {self.sim_data.time_step:.6e}\n")
            f.write(f"Total_Time(s): {self.sim_data.nr_steps * self.sim_data.time_step:.6e}\n")
            f.write(f"Number_of_Atoms: {self.sim_data.nr_atoms}\n")
            f.write(f"Number_Diffusing_Atoms: {self.sim_data.nr_diffusing}\n")
            f.write(f"Diffusing_Element: {self.sim_data.diff_elem}\n")
            f.write("#\n")
            f.write("# ANALYSIS RESULTS:\n")
            f.write(f"Total_Jumps_Detected: {len(self.jumps)}\n")
            f.write(f"Number_of_Sites: {len(self.sites_cart)}\n")
            f.write(f"Jump_Diffusivity(m^2/s): {jump_diffusivity:.6e}\n")
            f.write("#\n")
            f.write("# RATES:\n")
            total_time = self.sim_data.nr_steps * self.sim_data.time_step
            jump_rate_per_atom = len(self.jumps) / (self.sim_data.nr_diffusing * total_time)
            f.write(f"Jump_Rate_per_Atom(jumps/s): {jump_rate_per_atom:.6e}\n")
            f.write(f"Total_Jump_Rate(jumps/s): {len(self.jumps) / total_time:.6e}\n")
            
        print(f"Saved analysis summary: {summary_file}")

