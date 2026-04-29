"""Analysis of collective jump events in ion diffusion."""

import numpy as np
from collections import Counter
from ..core.utils import calc_dist_sqrd_frac
from ..core import AnalysisBase
from ..core.plot_io import save_plot_data

class CollectiveJumpAnalyzer(AnalysisBase):
    """Analyzes jump events to find pairs that are correlated in time and space."""

    def __init__(self, sim_data, jump_analysis_results, site_coordinates_cart,
                 coll_dist, coll_steps_manual=None, *, show_plot: bool = True, save_data: bool = False, output_dir: str | None = None, save_payload_dir: str | None = None):
        """
        Args:
            sim_data (SimData): The main simulation data object.
            jump_analysis_results: The results object from the completed SiteAnalyzerMDA run.
            site_coordinates_cart (np.ndarray): The Cartesian coordinates of the discovered sites.
            coll_dist (float): The maximum distance (in Angstroms) for spatial correlation.
            coll_steps_manual (int, optional): Manually set time window in simulation steps.
        """
        print("\n--- Stage: Analyzing Collective Jumps ---")
        super().__init__(show_plot=show_plot, save_data=save_data, output_dir=output_dir, save_payload_dir=save_payload_dir)
        self.sim_data = sim_data
        self.all_jumps = jump_analysis_results.all_trans
        self.sites_cart = site_coordinates_cart
        self.coll_dist = coll_dist
        self.coll_steps_manual = coll_steps_manual

        # Pre-calculate fractional coordinates for sites
        lattice_T = self.sim_data.lattice.T
        self.inv_lattice = np.linalg.inv(lattice_T)
        self.sites_frac = np.dot(self.inv_lattice, self.sites_cart.T).T

    def analyze(self):
        """
        Identify and analyze possibly collective jumps based on criteria.
        
        Returns:
            dict: A dictionary containing the analysis results. Returns None if analysis cannot proceed.
        """
        if self.all_jumps.size == 0:
            print("WARNING: Cannot analyze collective jumps. No jumps were detected.")
            return None

        # 1. Define Correlation Criteria
        # Convert time window to actual time units (picoseconds)
        if self.coll_steps_manual is not None:
            # Convert manual steps to time
            ts = self.sim_data.time_step * 1e12  # Convert to picoseconds
            coll_time_window = self.coll_steps_manual * ts
            print(f"INFO: Using manual time window for collective jumps: {self.coll_steps_manual} steps ({coll_time_window:.2f} ps).")
        elif self.sim_data.attempt_freq > 0:
            # Use inverse of attempt frequency as time window
            coll_time_window = (1.0 / self.sim_data.attempt_freq) * 1e12  # Convert to picoseconds
            equivalent_steps = coll_time_window / (self.sim_data.time_step * 1e12)
            print(f"INFO: Using calculated time window for collective jumps: {equivalent_steps:.1f} steps ({coll_time_window:.2f} ps).")
        else:
            print("WARNING: Attempt frequency is zero. Cannot calculate time window. Defaulting to 1 step.")
            ts = self.sim_data.time_step * 1e12  # Convert to picoseconds
            coll_time_window = ts

        coll_dist_sq = self.coll_dist**2
        print(f"INFO: Using spatial window for collective jumps: {self.coll_dist} Å.")

        # 2. Prepare Jump Data
        sorted_indices = np.argsort(self.all_jumps[:, 3])
        sorted_jumps = self.all_jumps[sorted_indices]
        nr_jumps = len(sorted_jumps)

        # 3. Find Correlated Pairs
        collective_pairs_indices = []
        is_jump_collective = np.zeros(nr_jumps, dtype=bool)
        
        # Debug counters
        time_passed = 0
        time_failed = 0
        same_atom_skipped = 0
        spatial_passed = 0
        spatial_failed = 0
        invalid_site_idx = 0

        for i in range(nr_jumps - 1):
            for j in range(i + 1, nr_jumps):
                jump_i = sorted_jumps[i]
                jump_j = sorted_jumps[j]

                time_diff = jump_j[3] - jump_i[3]  # Time difference in picoseconds
                if time_diff > coll_time_window:
                    # time_failed += 1
                    break  # Optimization: No need to check further for jump_i
                
                # time_passed += 1

                if jump_i[0] == jump_j[0]:
                    # same_atom_skipped += 1
                    continue  # Skip pairs from the same atom

                # Check for spatial correlation using Cartesian coordinates
                start_site_i, end_site_i = int(jump_i[1]) - 1, int(jump_i[2]) - 1
                start_site_j, end_site_j = int(jump_j[1]) - 1, int(jump_j[2]) - 1
                
                # Check for valid site indices
                if (start_site_i < 0 or start_site_i >= len(self.sites_cart) or
                    start_site_j < 0 or start_site_j >= len(self.sites_cart)):
                    # invalid_site_idx += 1
                    continue

                # Use direct Cartesian distance (more reliable than fractional with PBC)
                pos_si_cart = self.sites_cart[start_site_i]
                pos_sj_cart = self.sites_cart[start_site_j]
                
                # Simple Cartesian distance
                dist_sq = np.sum((pos_si_cart - pos_sj_cart)**2)
                dist = np.sqrt(dist_sq)

                if dist_sq <= coll_dist_sq:
                    spatial_passed += 1
                    # Found a collective pair
                    original_idx_i, original_idx_j = sorted_indices[i], sorted_indices[j]
                    collective_pairs_indices.append((original_idx_i, original_idx_j))
                    is_jump_collective[i] = True
                    is_jump_collective[j] = True
                    
                    # Show first few successful matches
                    if len(collective_pairs_indices) <= 3:
                        print(f"  ✅ Collective pair {len(collective_pairs_indices)}: "
                              f"Δt={time_diff:.1f}ps, distance={dist:.1f}Å, "
                              f"sites {start_site_i+1}→{end_site_i+1} & {start_site_j+1}→{end_site_j+1}")
                else:
                    spatial_failed += 1
                    # Sample distances for first few failures to guide parameter selection
                    if spatial_failed <= 5:
                        print(f"  ⚠️  Close in time ({time_diff:.1f}ps) but far in space ({dist:.1f}Å > {self.coll_dist}Å)")
        
        # Analysis summary for parameter optimization
        print(f"\n📊 ANALYSIS SUMMARY:")
        print(f"   Temporal correlation: {spatial_passed + spatial_failed} jump pairs within {coll_time_window:.1f}ps")
        print(f"   Spatial correlation: {spatial_passed} pairs within {self.coll_dist}Å")
        if spatial_failed > 0 and spatial_passed == 0:
            print(f"   💡 Suggestion: Consider increasing distance threshold (current failures: {spatial_failed} pairs)")

        coll_count = len(collective_pairs_indices)
        uncoll_count = nr_jumps - np.sum(is_jump_collective)

        print(f"Total number of jumps: {nr_jumps}")
        print(f"Number of possibly collective jump pairs found: {coll_count}")

        # 4. Analyze Collective Jump Types
        correlation_matrix, all_jump_types = self._build_correlation_matrix(collective_pairs_indices)

        # 5. Find Jumps Involved in Multiple Correlations
        multi_collective_jumps = self._find_multi_collective_jumps(collective_pairs_indices)

        results = {
            'collective_pairs': np.array(collective_pairs_indices),
            'uncollective_count': uncoll_count,
            'correlation_matrix': correlation_matrix,
            'jump_types': all_jump_types,
            'multi_collective_jumps': multi_collective_jumps
        }

        return results

    def _build_correlation_matrix(self, collective_pairs_indices):
        """Build correlation matrix for jump types."""
        if not collective_pairs_indices:
            return np.array([]), []

        def get_jump_name(jump_event):
            return f"Site_{int(jump_event[1])-1}->Site_{int(jump_event[2])-1}"

        all_jump_types = sorted(list(set(get_jump_name(j) for j in self.all_jumps)))
        type_to_idx = {name: i for i, name in enumerate(all_jump_types)}
        matrix_size = len(all_jump_types)
        corr_matrix = np.zeros((matrix_size, matrix_size), dtype=int)

        for idx_i, idx_j in collective_pairs_indices:
            jump_name_i = get_jump_name(self.all_jumps[idx_i])
            jump_name_j = get_jump_name(self.all_jumps[idx_j])
            mat_idx_i, mat_idx_j = type_to_idx[jump_name_i], type_to_idx[jump_name_j]
            corr_matrix[mat_idx_i, mat_idx_j] += 1
            if mat_idx_i != mat_idx_j:
                corr_matrix[mat_idx_j, mat_idx_i] += 1

        return corr_matrix, all_jump_types

    def _find_multi_collective_jumps(self, collective_pairs_indices):
        """Find jumps involved in multiple correlations."""
        if not collective_pairs_indices:
            return []

        all_correlated_indices = [idx for pair in collective_pairs_indices for idx in pair]
        counts = Counter(all_correlated_indices)
        multi_jumps = [idx for idx, count in counts.items() if count > 1]

        print(f"Number of jumps involved in multiple correlations: {len(multi_jumps)}")
        return multi_jumps

