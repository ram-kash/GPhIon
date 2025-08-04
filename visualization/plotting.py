"""3D visualization utilities for TAOG density grids."""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_taog_density(density_grid, edges, threshold_ratio=0.1, output_file=None):
    """
    Generate and optionally save a 3D scatter plot of the TAOG density.
    
    Args:
        density_grid (np.ndarray): 3D array of density values from the histogram.
        edges (list of np.ndarray): A list of three arrays defining the bin edges.
        threshold_ratio (float): Fraction of the maximum density to use as a plotting threshold.
        output_file (str, optional): Path to save the figure.
    """
    if density_grid.max() == 0:
        print("Warning: Density grid is empty. Nothing to plot.")
        return

    # Set a threshold to visualize only the most significant density peaks
    threshold = density_grid.max() * threshold_ratio

    # Find the grid indices of all voxels with a density above the threshold
    indices = np.array(np.nonzero(density_grid > threshold)).T

    if indices.shape[0] == 0:
        print(f"No density points found above the threshold ratio of {threshold_ratio}.")
        return

    # Calculate the center of each bin from the provided edges
    bin_centers_x = (edges[0][:-1] + edges[0][1:]) / 2
    bin_centers_y = (edges[1][:-1] + edges[1][1:]) / 2
    bin_centers_z = (edges[2][:-1] + edges[2][1:]) / 2

    # Map the grid indices to their real-space Cartesian coordinates
    x_coords = bin_centers_x[indices[:, 0]]
    y_coords = bin_centers_y[indices[:, 1]]
    z_coords = bin_centers_z[indices[:, 2]]

    # Get the density values for coloring the points
    color_values = density_grid[indices[:, 0], indices[:, 1], indices[:, 2]]

    # Create the 3D Plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Create the scatter plot
    scatter = ax.scatter(x_coords, y_coords, z_coords, c=color_values, 
                        cmap='viridis', s=40, alpha=0.7)

    # Add a color bar to show the mapping of colors to density
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.6)
    cbar.set_label('Occupancy Density (a.u.)')

    # Set plot labels and title
    ax.set_xlabel('X-axis (Å)')
    ax.set_ylabel('Y-axis (Å)')
    ax.set_zlabel('Z-axis (Å)')
    ax.set_title('3D Time-Averaged Occupancy Grid (TAOG) of Diffusing Ions')

    # Set axis limits to match the simulation box
    ax.set_xlim(edges[0][0], edges[0][-1])
    ax.set_ylim(edges[1][0], edges[1][-1])
    ax.set_zlim(edges[2][0], edges[2][-1])

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    else:
        plt.show()

