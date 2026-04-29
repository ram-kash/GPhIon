"""Site discovery using Time-Averaged Occupancy Grid (TAOG)."""

import numpy as np
import os
from scipy.ndimage import maximum_filter, label as nd_label
from scipy.spatial.distance import cdist  # <-- NEW: Used for distance filtering

class AmorphousSiteFinder:
    """Discovers ion hopping sites in amorphous materials using a 3D occupancy grid."""

    def __init__(self, sim_data, grid_resolution=100, min_peak_height=None, save_path='.'):
        """
        Args:
            sim_data (SimData): A populated simulation data object.
            grid_resolution (int): The number of bins along each axis for the density grid.
            min_peak_height (float, optional): The minimum relative height for a density peak.
            save_path (str): Directory to save the output .npy files.
        """
        self.sim_data = sim_data
        self.grid_resolution = grid_resolution
        self.min_peak_height = min_peak_height
        self.save_path = save_path
        # Set the minimum site separation as requested (1.2 Å)
        self.min_site_separation = 1.2 
        os.makedirs(self.save_path, exist_ok=True)

    def find_sites_from_density(self):
        """Execute the site discovery workflow, including distance-based filtering."""
        print("\n--- Stage 2: Discovering Sites from Time-Averaged Occupancy Grid ---")
        
        # 1. Build the 3D density grid
        density_grid, edges = self._build_occupancy_grid()
        
        # 2. Find peaks in the density grid (need all outputs for potential radius calculation later)
        labeled_peaks, num_features, peak_indices = self._find_grid_peaks(density_grid, return_labels=True)
        
        # 3. Convert peak grid indices to initial real-space Cartesian coordinates
        initial_sites_cart = self._convert_indices_to_coords(peak_indices, edges)
        
        # 4. Filter sites based on minimum separation (1.2 Å) <-- CORE NEW LOGIC
        discovered_sites_cart = self._filter_sites_by_distance(initial_sites_cart, self.min_site_separation)
        
        sites_filepath = os.path.join(self.save_path, 'discovered_sites_cart.npy')
        np.save(sites_filepath, discovered_sites_cart)
        print(f"DEBUG: Found {len(initial_sites_cart)} initial potential sites.")
        print(f"DEBUG: Filtered to {len(discovered_sites_cart)} distinct hopping sites with >{self.min_site_separation} Å separation.")
        print(f"DEBUG: Saved discovered site coordinates to {sites_filepath}")
        
        return discovered_sites_cart

    def _filter_sites_by_distance(self, sites_cart, min_dist):
        """
        Filters a list of site coordinates to ensure minimum separation (min_dist).
        """
        if sites_cart.shape[0] == 0:
            return sites_cart
        
        # Accept the first site unconditionally
        accepted_sites = [sites_cart[0]]
        
        for i in range(1, sites_cart.shape[0]):
            candidate_site = sites_cart[i].reshape(1, 3)
            # Calculate distance from candidate to all already accepted sites
            distances = cdist(candidate_site, accepted_sites)
            
            # If the candidate is farther than min_dist from ALL accepted sites, accept it
            if np.all(distances >= min_dist):
                accepted_sites.append(candidate_site.flatten())
                
        return np.array(accepted_sites)

    def _build_occupancy_grid(self):
        """Calculate and save the 3D histogram and its edges."""
        print(f"DEBUG: Building {self.grid_resolution}x{self.grid_resolution}x{self.grid_resolution} occupancy grid...")
        
        # Use positions of all diffusing atoms over all time steps
        all_positions = self.sim_data.cart_pos.transpose(1, 2, 0).reshape(-1, 3)
        
        box_dims = self.sim_data.universe.dimensions
        box_range = [[0, box_dims[0]], [0, box_dims[1]], [0, box_dims[2]]]
        
        density, edges = np.histogramdd(all_positions, bins=self.grid_resolution, range=box_range)
        
        # Save both the density and the edges
        grid_filepath = os.path.join(self.save_path, 'occupancy_density_grid.npy')
        edges_filepath = os.path.join(self.save_path, 'occupancy_grid_edges.npy')
        
        np.save(grid_filepath, density)
        np.save(edges_filepath, edges, allow_pickle=True)
        print(f"DEBUG: Saved grid edges to {edges_filepath}")
        
        return density, edges

    def _find_grid_peaks(self, density_grid, return_labels=False):
        """Identify local maxima in the 3D density grid."""
        if self.min_peak_height is None:
            non_zero_density = density_grid[density_grid > 0]
            if non_zero_density.size == 0:
                threshold = 0.0
            else:
                threshold = 1 * non_zero_density.mean()
        else:
            threshold = self.min_peak_height

        print(f"DEBUG: Using density threshold of {threshold:.2f} to find peaks.")

        # Use a maximum filter to find local maxima
        neighborhood = np.ones((3, 3, 3))
        local_max = (density_grid == maximum_filter(density_grid, footprint=neighborhood))

        # Apply the threshold
        local_max[density_grid < threshold] = False

        # Label connected regions of peaks to get distinct sites
        labeled_peaks, num_features = nd_label(local_max)

        # Find the center of each labeled region
        peak_indices = np.array([
            np.mean(np.argwhere(labeled_peaks == i), axis=0)
            for i in range(1, num_features + 1)
        ])

        if return_labels:
            return labeled_peaks, num_features, peak_indices
        else:
            return peak_indices


    def _convert_indices_to_coords(self, indices, edges):
        """Convert grid indices to Cartesian coordinates using the bin edges."""
        if indices.size == 0:
            return np.empty((0, 3))

        # Calculate the center of each bin from the edges
        bin_centers = [(e[:-1] + e[1:]) / 2 for e in edges]

        # Map float indices to the nearest bin center coordinate
        cart_coords = np.array([
            [bin_centers[dim][int(round(idx))] for dim, idx in enumerate(coord)]
            for coord in indices
        ])

        return cart_coords
    
    # ------------------------------------------------------------------
    #  Robust, grid-aware site-radius estimator (Kept for completeness)
    # ------------------------------------------------------------------
    def get_site_radii(self, labeled_peaks, num_features, edges,
                    discovered_sites_cart, pct=95):
        """
        Estimate a radius for each site from the 3-D occupancy grid.

        Parameters
        ----------
        labeled_peaks : ndarray[int]
            3-D array returned by nd_label: each voxel has 0 or a peak label.
        num_features : int
            Number of distinct peak regions (labels).
        edges : list[np.ndarray]
            Bin edges from np.histogramdd (x, y, z).
        discovered_sites_cart : (N,3) ndarray
            Cartesian coordinates of site centres (same order as labels).
        pct : int
            Percentile of voxel distances to keep (default = 95).
        Returns
        -------
        radii : (N,) ndarray
            One radius per site in Å.
        """
        # --- grid geometry -------------------------------------------
        bin_centres = [(e[:-1] + e[1:]) / 2.0 for e in edges]
        voxel_size  = np.array([e[1] - e[0] for e in edges])
        voxel_diag  = np.linalg.norm(voxel_size)          # full body diagonal

        radii = []
        for lbl in range(1, num_features + 1):
            # voxels belonging to this peak
            vox = np.argwhere(labeled_peaks == lbl)
            if vox.size == 0:        # paranoia
                print(f"WARNING: label {lbl} has no voxels – fallback radius used")
                radii.append(0.01)
                continue

            # voxel-centre → Cartesian
            coords = np.array([[bin_centres[d][idx] for d, idx in enumerate(v)]
                            for v in vox])

            centre = discovered_sites_cart[lbl - 1]
            dists  = np.linalg.norm(coords - centre, axis=1)

            # percentile + half-diagonal compensates centre-to-edge offset
            r = np.percentile(dists, pct) + 0.5 * voxel_diag
            radii.append(r)

        return np.asarray(radii)