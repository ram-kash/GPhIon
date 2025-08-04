"""Jump detection and analysis between discovered sites."""

import numpy as np
from MDAnalysis.analysis.base import AnalysisBase

class SiteAnalyzerMDA(AnalysisBase):
    """Finds atomic jumps based on a given set of site coordinates."""

    def __init__(self, diffusing_atom_group, site_coordinates_cart, **kwargs):
        super().__init__(diffusing_atom_group.universe.trajectory, **kwargs)
        self._ag = diffusing_atom_group
        self.site_cart_pos = site_coordinates_cart
        self.site_radius = None
        self._previous_sites = np.zeros(len(self._ag), dtype=int)

    def _prepare(self):
        self._trajectory.rewind()
        pos0 = self._ag.positions.copy()
        self._trajectory[1]
        pos1 = self._ag.positions.copy()
        self._trajectory.rewind()

        vibration_amp = np.mean(np.linalg.norm(pos1 - pos0, axis=1))
        self.site_radius = 2.0 * vibration_amp

        print(f"DEBUG: Using site radius for jump analysis: {self.site_radius:.4f} Å")
        self.results.all_trans = []

    def _single_frame(self):
        current_sites = np.zeros(len(self._ag), dtype=int)
        close_sqrd = self.site_radius**2

        # Use MDAnalysis's fast distance calculation capabilities
        for i, atom_pos in enumerate(self._ag.positions):
            distances_sq = np.sum((self.site_cart_pos - atom_pos)**2, axis=1)
            min_dist_sq = np.min(distances_sq)
            
            if min_dist_sq < close_sqrd:
                found_site_idx = np.argmin(distances_sq)
                current_sites[i] = found_site_idx + 1  # 1-based index

        # Log jumps
        for i in range(len(self._ag)):
            prev_site, curr_site = self._previous_sites[i], current_sites[i]
            if prev_site > 0 and curr_site > 0 and curr_site != prev_site:
                self.results.all_trans.append([i, prev_site, curr_site, self._trajectory.time])

        self._previous_sites = current_sites

    def _conclude(self):
        self.results.all_trans = np.array(self.results.all_trans)

