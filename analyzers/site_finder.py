"""Site discovery using Time-Averaged Occupancy Grid (TAOG)."""

import numpy as np
import os
from scipy.ndimage import maximum_filter, label as nd_label

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
        os.makedirs(self.save_path, exist_ok=True)

    def find_sites_from_density(self):
        """Execute the site discovery workflow."""
        print("\n--- Stage 2: Discovering Sites from Time-Averaged Occupancy Grid ---")
        
        # 1. Build the 3D density grid
        density_grid, edges = self._build_occupancy_grid()
        
        # 2. Find peaks in the density grid
        peak_indices = self._find_grid_peaks(density_grid)
        
        # 3. Convert peak grid indices to real-space Cartesian coordinates
        discovered_sites_cart = self._convert_indices_to_coords(peak_indices, edges)
        
        sites_filepath = os.path.join(self.save_path, 'discovered_sites_cart.npy')
        np.save(sites_filepath, discovered_sites_cart)
        print(f"DEBUG: Found {len(discovered_sites_cart)} potential sites.")
        print(f"DEBUG: Saved discovered site coordinates to {sites_filepath}")
        
        return discovered_sites_cart

    def _build_occupancy_grid(self):
        """Calculate and save the 3D histogram and its edges."""
        print(f"DEBUG: Building {self.grid_resolution}x{self.grid_resolution}x{self.grid_resolution} occupancy grid...")
        
        diff_atom_mask = self.sim_data.diffusing_atom
        all_positions = self.sim_data.cart_pos[:, diff_atom_mask, :].transpose(1, 2, 0).reshape(-1, 3)
        
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

    def _find_grid_peaks(self, density_grid):
        """Identify local maxima in the 3D density grid."""
        if self.min_peak_height is None:
            threshold = 5 * density_grid[density_grid > 0].mean()
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

