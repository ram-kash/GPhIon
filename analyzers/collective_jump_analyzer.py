"""Analysis of collective jump events in ion diffusion."""

import numpy as np
from collections import Counter
from ..core.utils import calc_dist_sqrd_frac

class CollectiveJumpAnalyzer:
    """Analyzes jump events to find pairs that are correlated in time and space."""

    def __init__(self, sim_data, jump_analysis_results, site_coordinates_cart,
                 coll_dist, coll_steps_manual=None):
        """
        Args:
            sim_data (SimData): The main simulation data object.
            jump_analysis_results: The results object from the completed SiteAnalyzerMDA run.
            site_coordinates_cart (np.ndarray): The Cartesian coordinates of the discovered sites.
            coll_dist (float): The maximum distance (in Angstroms) for spatial correlation.
            coll_steps_manual (int, optional): Manually set time window in simulation steps.
        """
        print("\n--- Stage: Analyzing Collective Jumps ---")
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
        if self.coll_steps_manual is not None:
            coll_steps = self.coll_steps_manual
            print(f"INFO: Using manual time window for collective jumps: {coll_steps} steps.")
        elif self.sim_data.attempt_freq > 0:
            ts = self.sim_data.time_step
            coll_steps = np.ceil((1.0 / self.sim_data.attempt_freq) / ts)
            print(f"INFO: Using calculated time window for collective jumps: {int(coll_steps)} steps.")
        else:
            print("WARNING: Attempt frequency is zero. Cannot calculate time window. Defaulting to 1 step.")
            coll_steps = 1

        coll_dist_sq = self.coll_dist**2
        print(f"INFO: Using spatial window for collective jumps: {self.coll_dist} Å.")

        # 2. Prepare Jump Data
        sorted_indices = np.argsort(self.all_jumps[:, 3])
        sorted_jumps = self.all_jumps[sorted_indices]
        nr_jumps = len(sorted_jumps)

        # 3. Find Correlated Pairs
        collective_pairs_indices = []
        is_jump_collective = np.zeros(nr_jumps, dtype=bool)

        for i in range(nr_jumps - 1):
            for j in range(i + 1, nr_jumps):
                jump_i = sorted_jumps[i]
                jump_j = sorted_jumps[j]

                time_diff_steps = jump_j[3] - jump_i[3]
                if time_diff_steps > coll_steps:
                    break  # Optimization: No need to check further for jump_i

                if jump_i[0] == jump_j[0]:
                    continue  # Skip pairs from the same atom

                # Check for spatial correlation
                start_site_i, end_site_i = int(jump_i[1]) - 1, int(jump_i[2]) - 1
                start_site_j, end_site_j = int(jump_j[1]) - 1, int(jump_j[2]) - 1

                pos_si = self.sites_frac[start_site_i]
                pos_sj = self.sites_frac[start_site_j]

                # Minimum image convention distance check between start sites
                dist_sq = calc_dist_sqrd_frac(pos_si, pos_sj, self.sim_data.lattice.T)

                if dist_sq <= coll_dist_sq:
                    # Found a collective pair
                    original_idx_i, original_idx_j = sorted_indices[i], sorted_indices[j]
                    collective_pairs_indices.append((original_idx_i, original_idx_j))
                    is_jump_collective[i] = True
                    is_jump_collective[j] = True

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

