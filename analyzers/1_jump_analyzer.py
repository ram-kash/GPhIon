"""Jump analysis based on existing TAOG pipeline data structures."""

import numpy as np
import matplotlib.pyplot as plt
import os
from ..core.utils import calc_dist_sqrd_frac

class JumpAnalyzer:
    """Analyzes jump statistics and calculates occupancy factors from discovered sites and simulation data."""

    def __init__(self, sim_data, discovered_sites_cart, site_radii=None):
        """
        Initialize the JumpAnalyzer using existing TAOG pipeline data.
        
        Args:
            sim_data (SimData): The populated simulation data object from your pipeline
            discovered_sites_cart (np.ndarray): Site coordinates from AmorphousSiteFinder.find_sites_from_density()
            site_radii (np.ndarray, optional): Site radii from AmorphousSiteFinder.get_site_radii()
        """
        self.sim_data = sim_data
        self.sites_cart = discovered_sites_cart
        self.site_radii = site_radii if site_radii is not None else np.ones(len(discovered_sites_cart)) * 1.5
        self.nr_sites = len(discovered_sites_cart)
        self.nr_diffusing = sim_data.nr_diffusing
        
        # Convert Cartesian site coordinates to fractional
        lattice_inv = np.linalg.inv(sim_data.lattice)
        self.sites_frac = np.dot(lattice_inv, discovered_sites_cart.T).T
        
        # Initialize results storage
        self.results = type('Results', (), {})()
        self.results.site_occupancies = np.zeros(self.nr_sites)
        self.results.average_occupancy_factors = np.zeros(self.nr_sites)
        self.results.residence_time_stats = {}
        self.results.all_trans = []  # Jump events in same format as SiteAnalyzerMDA
        
        # Initialize site tracking arrays
        self.site_occupancy_frames = np.zeros((sim_data.nr_steps, self.nr_sites))
        self.site_residence_times = {i: [] for i in range(self.nr_sites)}
        self.atom_current_sites = [-1] * sim_data.nr_atoms  # Current site for each atom
        self.atom_site_entry_times = [0] * sim_data.nr_atoms  # When atom entered current site
        
        # Create transition matrix
        self.transitions = np.zeros((self.nr_sites, self.nr_sites))
        
        print("\n--- Stage 3: Jump Analysis from Discovered Sites ---")
        print(f"DEBUG: Number of sites: {self.nr_sites}")
        print(f"DEBUG: Number of diffusing atoms: {self.nr_diffusing}")
        print(f"DEBUG: Site radii range: {np.min(self.site_radii):.3f} - {np.max(self.site_radii):.3f} Å")

    def analyze_trajectory(self):
        """
        Analyze the full trajectory to identify site occupancies and transitions.
        This replaces the MDAnalysis.analysis.base functionality.
        """
        print("Analyzing trajectory for site occupancies and transitions...")
        
        # Process each frame
        for frame_idx in range(self.sim_data.nr_steps):
            self._process_frame(frame_idx)
        
        # Finalize residence times for atoms still in sites at trajectory end
        final_frame = self.sim_data.nr_steps - 1
        final_time = final_frame * self.sim_data.time_step
        
        for atom_idx in range(self.sim_data.nr_atoms):
            if self.sim_data.diffusing_atom[atom_idx] and self.atom_current_sites[atom_idx] != -1:
                site_idx = self.atom_current_sites[atom_idx]
                entry_time = self.atom_site_entry_times[atom_idx]
                residence_time = final_time - entry_time
                if residence_time > 0:
                    self.site_residence_times[site_idx].append(residence_time)
        
        # Calculate occupancy factors
        self._calculate_occupancy_factors()
        
        print(f"Analysis complete. Found {len(self.results.all_trans)} jump events.")
        print(f"Total transitions in matrix: {np.sum(self.transitions)}")

    def _process_frame(self, frame_idx):
        """Process a single frame to update site occupancies and detect transitions."""
        current_time = frame_idx * self.sim_data.time_step
        
        # Get fractional coordinates for this frame
        lattice_inv = np.linalg.inv(self.sim_data.lattice)
        
        for atom_idx in range(self.sim_data.nr_atoms):
            if not self.sim_data.diffusing_atom[atom_idx]:
                continue
            
            # Get atom position in fractional coordinates
            atom_pos_cart = self.sim_data.cart_pos[:, atom_idx, frame_idx]
            atom_pos_frac = np.dot(lattice_inv, atom_pos_cart)
            
            # Find which site (if any) this atom is in
            current_site = self._find_atom_site(atom_pos_frac, atom_pos_cart)
            previous_site = self.atom_current_sites[atom_idx]
            
            # Handle site changes
            if current_site != previous_site:
                # Atom left previous site
                if previous_site != -1:
                    entry_time = self.atom_site_entry_times[atom_idx]
                    residence_time = current_time - entry_time
                    if residence_time > 0:
                        self.site_residence_times[previous_site].append(residence_time)
                
                # Atom entered new site
                if current_site != -1:
                    self.atom_site_entry_times[atom_idx] = current_time
                    
                    # Record jump event if both sites are valid
                    if previous_site != -1 and current_site != -1:
                        # Format: [atom_id, from_site+1, to_site+1, time] (1-based site indexing)
                        jump_event = [atom_idx + 1, previous_site + 1, current_site + 1, current_time]
                        self.results.all_trans.append(jump_event)
                        self.transitions[previous_site, current_site] += 1
                
                self.atom_current_sites[atom_idx] = current_site
            
            # Update occupancy for current frame
            if current_site != -1:
                self.site_occupancy_frames[frame_idx, current_site] += 1

    def _find_atom_site(self, atom_pos_frac, atom_pos_cart):
        """
        Find which site an atom belongs to based on distance criteria.
        
        Args:
            atom_pos_frac (np.ndarray): Atom position in fractional coordinates
            atom_pos_cart (np.ndarray): Atom position in Cartesian coordinates
            
        Returns:
            int: Site index or -1 if not in any site
        """
        for site_idx in range(self.nr_sites):
            site_pos_frac = self.sites_frac[site_idx]
            
            # Calculate distance with periodic boundary conditions
            dist_sq = calc_dist_sqrd_frac(atom_pos_frac, site_pos_frac, self.sim_data.lattice)
            distance = np.sqrt(dist_sq)
            
            if distance <= self.site_radii[site_idx]:
                return site_idx
        
        return -1  # Not in any site

    def calculate_jump_statistics(self, diffusion_dim=3, resolution=0.1, show_pics=True, 
                                save_data=False, output_dir=None):
        """
        Calculate jump statistics and occupancy factors.
        
        Args:
            diffusion_dim (int): Number of dimensions for diffusion (default: 3)
            resolution (float): Bin size in Angstroms for histogram (default: 0.1)
            show_pics (bool): If True, display histogram plot
            save_data (bool): If True, save analysis data to files
            output_dir (str, optional): Directory to save data and plots
            
        Returns:
            tuple: (jump_data, occupancy_data)
        """
        if len(self.results.all_trans) == 0:
            print("WARNING: No jump events detected. Run analyze_trajectory() first.")
            return {}, {}
        
        # Convert results to numpy array for easier processing
        self.results.all_trans = np.array(self.results.all_trans)
        
        # Setup for histogram
        max_dist = 10.0
        max_bins = int(np.ceil(max_dist / resolution))
        jumps_vs_dist = np.zeros(max_bins)
        max_bin = 0
        
        # Store detailed jump information
        jump_distances = []
        jump_counts = []
        site_pairs = []
        jump_data_for_diffusion = []
        
        print(f"DEBUG: Processing {len(self.results.all_trans)} jump events...")
        
        # Process each unique site pair based on transitions matrix
        for i in range(self.nr_sites):
            for j in range(self.nr_sites):
                if i != j and self.transitions[i, j] > 0:
                    nr_jumps = int(self.transitions[i, j])
                    
                    # Calculate distance between sites
                    pos_i = self.sites_frac[i]
                    pos_j = self.sites_frac[j]
                    
                    dist_sq = calc_dist_sqrd_frac(pos_i, pos_j, self.sim_data.lattice)
                    dist = np.sqrt(dist_sq)
                    
                    # Store jump data for external diffusion calculation
                    occupancy_factor_i = self.results.average_occupancy_factors[i]
                    occupancy_factor_j = self.results.average_occupancy_factors[j]
                    
                    jump_data_for_diffusion.append({
                        'site_i': i,
                        'site_j': j,
                        'distance': dist,
                        'distance_sq': dist_sq,
                        'nr_jumps': nr_jumps,
                        'occupancy_factor_i': occupancy_factor_i,
                        'occupancy_factor_j': occupancy_factor_j
                    })
                    
                    # Add to histogram
                    dist_bin = int(np.ceil(dist / resolution)) - 1
                    if 0 <= dist_bin < len(jumps_vs_dist):
                        jumps_vs_dist[dist_bin] += nr_jumps
                        max_bin = max(max_bin, dist_bin)
                    
                    # Store for detailed analysis
                    jump_distances.append(dist)
                    jump_counts.append(nr_jumps)
                    site_pairs.append((i, j))
        
        # Trim histogram to actual data
        jumps_vs_dist = jumps_vs_dist[:max_bin + 1] if max_bin > 0 else jumps_vs_dist[:1]
        
        # Create distance axis for plotting
        distances = np.arange(0.5 * resolution, (max_bin + 1.5) * resolution, resolution) if max_bin > 0 else np.array([0.5 * resolution])
        
        # Prepare return data
        jump_data = {
            'distances': distances,
            'jumps_vs_dist': jumps_vs_dist,
            'jump_distances': np.array(jump_distances),
            'jump_counts': np.array(jump_counts),
            'site_pairs': site_pairs,
            'total_jumps': np.sum(jump_counts) if jump_counts else 0,
            'max_distance': np.max(jump_distances) if jump_distances else 0.0,
            'mean_distance': np.average(jump_distances, weights=jump_counts) if jump_distances else 0.0,
            'jump_data_for_diffusion': jump_data_for_diffusion,
            'all_trans': self.results.all_trans  # Raw jump events for compatibility
        }
        
        # Prepare occupancy data
        occupancy_data = {
            'site_occupancies': self.results.site_occupancies,
            'occupancy_factors': self.results.average_occupancy_factors,
            'residence_time_stats': self.results.residence_time_stats,
            'jump_specific_occupancy_factors': self.get_jump_specific_occupancy_factors()
        }
        
        # Print statistics
        self.print_occupancy_statistics()
        self.print_jump_statistics(jump_data)
        
        # Save data if requested
        if save_data:
            self._save_jump_data(jump_data, occupancy_data, diffusion_dim, 
                               resolution, output_dir)
        
        # Generate plot if requested
        if show_pics and len(jump_distances) > 0:
            self._plot_jumps_vs_distance(distances, jumps_vs_dist, jump_data, output_dir)
        
        return jump_data, occupancy_data

    def _calculate_occupancy_factors(self):
        """Calculate occupancy factors for each site."""
        print("Calculating occupancy factors...")
        
        n_diffusing_atoms = self.nr_diffusing
        
        # Calculate time-averaged occupancy for each site
        for site_idx in range(self.nr_sites):
            # Average number of atoms at this site across all frames
            avg_occupancy = np.mean(self.site_occupancy_frames[:, site_idx])
            self.results.site_occupancies[site_idx] = avg_occupancy
            
            # Occupancy factor: fraction of time site is occupied relative to maximum possible
            self.results.average_occupancy_factors[site_idx] = avg_occupancy / n_diffusing_atoms
        
        # Calculate residence time statistics
        self.results.residence_time_stats = {}
        for site_idx in range(self.nr_sites):
            if self.site_residence_times[site_idx]:
                times = np.array(self.site_residence_times[site_idx])
                self.results.residence_time_stats[site_idx] = {
                    'mean': np.mean(times),
                    'std': np.std(times),
                    'min': np.min(times),
                    'max': np.max(times),
                    'count': len(times)
                }
            else:
                self.results.residence_time_stats[site_idx] = {
                    'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'count': 0
                }

    def get_site_occupancy_factor(self, site_index):
        """Get occupancy factor for a specific site."""
        if hasattr(self.results, 'average_occupancy_factors') and len(self.results.average_occupancy_factors) > site_index:
            return self.results.average_occupancy_factors[site_index]
        else:
            print("Warning: Occupancy factors not calculated. Run analyze_trajectory() first.")
            return 1.0

    def get_jump_specific_occupancy_factors(self):
        """Get occupancy factors for each jump type based on transitions."""
        if not hasattr(self.results, 'average_occupancy_factors'):
            print("Warning: Occupancy factors not calculated.")
            return {}
        
        occupancy_factors = {}
        
        # Process all non-zero transitions
        for i in range(self.nr_sites):
            for j in range(self.nr_sites):
                if i != j and self.transitions[i, j] > 0:
                    jump_type = f"Site_{i}->Site_{j}"
                    occupancy_factors[jump_type] = self.results.average_occupancy_factors[i]
        
        return occupancy_factors

    def print_occupancy_statistics(self):
        """Print detailed occupancy statistics."""
        if not hasattr(self.results, 'site_occupancies'):
            print("No occupancy data available. Run analyze_trajectory() first.")
            return
        
        print("\n" + "="*60)
        print("SITE OCCUPANCY ANALYSIS")
        print("="*60)
        
        for site_idx in range(self.nr_sites):
            avg_occ = self.results.site_occupancies[site_idx]
            occ_factor = self.results.average_occupancy_factors[site_idx]
            
            print(f"\nSite {site_idx}:")
            print(f"  Average Occupancy: {avg_occ:.3f} atoms")
            print(f"  Occupancy Factor: {occ_factor:.4f}")
            
            stats = self.results.residence_time_stats[site_idx]
            if stats['count'] > 0:
                print(f"  Residence Time: {stats['mean']:.2f} ± {stats['std']:.2f} s")
                print(f"  Residence Events: {stats['count']}")
            else:
                print(f"  No residence events recorded")
        
        print(f"\nOverall Statistics:")
        print(f"  Total Sites: {self.nr_sites}")
        print(f"  Average Occupancy Factor: {np.mean(self.results.average_occupancy_factors):.4f}")
        print(f"  Most Occupied Site: Site_{np.argmax(self.results.site_occupancies)} "
              f"({np.max(self.results.site_occupancies):.3f} atoms)")
        print(f"  Least Occupied Site: Site_{np.argmin(self.results.site_occupancies)} "
              f"({np.min(self.results.site_occupancies):.3f} atoms)")

    def print_jump_statistics(self, jump_data):
        """Print jump statistics summary."""
        print("\n" + "="*60)
        print("JUMP STATISTICS")
        print("="*60)
        print(f"Total jump events detected: {len(self.results.all_trans)}")
        print(f"Total transitions in matrix: {np.sum(self.transitions)}")
        print(f"Active site pairs: {len(jump_data['site_pairs'])}")
        if jump_data['total_jumps'] > 0:
            print(f"Mean jump distance: {jump_data['mean_distance']:.3f} Å")
            print(f"Max jump distance: {jump_data['max_distance']:.3f} Å")

    def _plot_jumps_vs_distance(self, distances, jumps_vs_dist, jump_data, output_dir=None):
        """Generate and display the jumps vs distance histogram."""
        plt.figure(figsize=(10, 6))
        plt.bar(distances, jumps_vs_dist, width=distances[1]-distances[0] if len(distances) > 1 else 0.1, 
                align='center', edgecolor='black', alpha=0.7, color='skyblue')
        
        plt.title('Jumps vs. Distance')
        plt.xlabel('Distance (Angstrom)')
        plt.ylabel('Number of Jumps')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add statistics to plot
        total_jumps = jump_data['total_jumps']
        mean_dist = jump_data['mean_distance']
        max_dist = jump_data['max_distance']
        
        if mean_dist > 0:
            plt.axvline(mean_dist, color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {mean_dist:.2f} Å')
            plt.legend()
        
        # Add statistics text box
        stats_text = (f'Total jumps: {total_jumps:.0f}\n'
                     f'Mean distance: {mean_dist:.2f} Å\n'
                     f'Max distance: {max_dist:.2f} Å')
        
        plt.text(0.98, 0.98, stats_text, transform=plt.gca().transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, "jumps_vs_distance.png"),
                       dpi=300, bbox_inches='tight')
        
        plt.show()

    def _save_jump_data(self, jump_data, occupancy_data, diffusion_dim, 
                       resolution, output_dir=None):
        """Save jump analysis data to files."""
        if output_dir is None:
            output_dir = "."
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Save raw jump events (compatible with JumpDiffusivityAnalyzer)
        jumps_file = os.path.join(output_dir, "all_jump_events.dat")
        if len(self.results.all_trans) > 0:
            np.savetxt(jumps_file, self.results.all_trans,
                      header="Atom_ID From_Site To_Site Time",
                      fmt="%d %d %d %.6f")
            print(f"Saved jump events: {jumps_file}")
        
        # 2. Save transition matrix
        matrix_file = os.path.join(output_dir, "transition_matrix.dat")
        np.savetxt(matrix_file, self.transitions,
                  header="Transition matrix (from_site, to_site)",
                  fmt="%d")
        print(f"Saved transition matrix: {matrix_file}")
        
        # 3. Save histogram data
        histogram_file = os.path.join(output_dir, "jumps_vs_distance_histogram.dat")
        hist_data = np.column_stack([jump_data['distances'], jump_data['jumps_vs_dist']])
        np.savetxt(histogram_file, hist_data,
                  header="Distance(Angstrom) Number_of_Jumps",
                  fmt="%.6f %d")
        print(f"Saved histogram data: {histogram_file}")
        
        # 4. Save occupancy statistics
        occupancy_file = os.path.join(output_dir, "site_occupancy_analysis.dat")
        with open(occupancy_file, 'w') as f:
            f.write("# Site Occupancy Analysis\n")
            f.write("# Site_Index Average_Occupancy Occupancy_Factor Residence_Mean Residence_Std Residence_Count\n")
            for site_idx in range(self.nr_sites):
                avg_occ = occupancy_data['site_occupancies'][site_idx]
                occ_factor = occupancy_data['occupancy_factors'][site_idx]
                res_stats = occupancy_data['residence_time_stats'][site_idx]
                f.write(f"{site_idx:6d} {avg_occ:15.6f} {occ_factor:15.6f} "
                       f"{res_stats['mean']:12.6f} {res_stats['std']:12.6f} {res_stats['count']:6d}\n")
        print(f"Saved occupancy data: {occupancy_file}")

    # Compatibility method to work with your existing JumpDiffusivityAnalyzer
    def get_results_for_diffusivity_analyzer(self):
        """
        Get results in format compatible with JumpDiffusivityAnalyzer.
        
        Returns:
            Object with all_trans attribute containing jump events
        """
        results_obj = type('Results', (), {})()
        results_obj.all_trans = self.results.all_trans
        return results_obj
