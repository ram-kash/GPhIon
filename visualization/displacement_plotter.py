"""Displacement-based plotting utilities for TAOG analysis."""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter
import os

class DisplacementPlotter:
    """
    A dedicated class for plotting displacement-based analysis results.
    
    This class provides various displacement visualizations including:
    - Individual atom displacement trajectories
    - Displacement histograms
    - Average displacement per element
    - 3D density plots with isosurfaces
    - Site overlay on density plots
    """
    
    def __init__(self):
        """Initialize the Displacement plotter."""
        self.default_figure_size = (12, 8)
        self.default_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        
    def plot_individual_displacements(self, sim_data, timestep=None, max_atoms=None, 
                                    output_file=None):
        """
        Plot displacement trajectories for each diffusing atom.
        
        Args:
            sim_data (SimData): The simulation data object
            timestep (float, optional): Timestep to convert frames to real time
            max_atoms (int, optional): Maximum number of atoms to plot (for clarity)
            output_file (str, optional): Path to save the plot
        """
        plt.figure(figsize=self.default_figure_size)
        
        # Prepare time axis
        time_frames = np.arange(sim_data.nr_steps)
        if timestep is not None:
            time_axis = time_frames * timestep * 1e12  # Convert to ps
            xlabel = 'Time (ps)'
        else:
            time_axis = time_frames
            xlabel = 'Time step'
        
        plotted_count = 0
        for i in range(sim_data.nr_atoms):
            if sim_data.diffusing_atom[i]:
                if max_atoms is not None and plotted_count >= max_atoms:
                    break
                plt.plot(time_axis, sim_data.displacement[i, :], alpha=0.7, linewidth=1)
                plotted_count += 1
        
        plt.title('Displacement of Diffusing Element')
        plt.xlabel(xlabel)
        plt.ylabel('Displacement (Å)')
        plt.grid(True, alpha=0.3)
        
        if max_atoms is not None and plotted_count == max_atoms:
            plt.text(0.02, 0.98, f'Showing {max_atoms} of {sim_data.nr_diffusing} atoms', 
                    transform=plt.gca().transAxes, verticalalignment='top', 
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Plot saved: {output_file}")
        
        plt.show()
    
    def plot_displacement_histogram(self, sim_data, bins=50, output_file=None):
        """
        Plot histogram of final displacements for diffusing atoms.
        
        Args:
            sim_data (SimData): The simulation data object
            bins (int): Number of histogram bins
            output_file (str, optional): Path to save the plot
        """
        # Extract final displacements for diffusing atoms
        final_displacements = []
        for i in range(sim_data.nr_atoms):
            if sim_data.diffusing_atom[i]:
                final_displacements.append(sim_data.displacement[i, -1])
        
        final_displacements = np.array(final_displacements)
        
        plt.figure(figsize=self.default_figure_size)
        plt.hist(final_displacements, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('Histogram of Final Displacement of Diffusing Element')
        plt.xlabel('Displacement (Å)')
        plt.ylabel('Number of atoms')
        plt.grid(True, alpha=0.3)
        
        # Add statistics
        mean_disp = np.mean(final_displacements)
        std_disp = np.std(final_displacements)
        plt.axvline(mean_disp, color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {mean_disp:.2f} Å')
        plt.legend()
        
        # Add text box with statistics
        stats_text = f'Mean: {mean_disp:.2f} Å\nStd: {std_disp:.2f} Å\nAtoms: {len(final_displacements)}'
        plt.text(0.98, 0.98, stats_text, transform=plt.gca().transAxes, 
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Plot saved: {output_file}")
        
        plt.show()
    
    def plot_displacement_per_element(self, sim_data, elements=None, timestep=None, 
                                    output_file=None):
        """
        Plot average displacement per element type.
        
        Args:
            sim_data (SimData): The simulation data object
            elements (list, optional): List of element names. If None, extracted from sim_data
            timestep (float, optional): Timestep to convert frames to real time
            output_file (str, optional): Path to save the plot
        """
        plt.figure(figsize=self.default_figure_size)
        
        # Prepare time axis
        time_frames = np.arange(sim_data.nr_steps)
        if timestep is not None:
            time_axis = time_frames * timestep * 1e12  # Convert to ps
            xlabel = 'Time (ps)'
        else:
            time_axis = time_frames
            xlabel = 'Time step'
        
        # Get unique elements
        if hasattr(sim_data.universe, 'atoms') and hasattr(sim_data.universe.atoms, 'elements'):
            unique_elements = np.unique(sim_data.universe.atoms.elements)
        elif elements is not None:
            unique_elements = elements
        else:
            unique_elements = [sim_data.diff_elem]  # Fallback to diffusing element only
        
        # Calculate average displacement per element
        for i, element in enumerate(unique_elements):
            if hasattr(sim_data.universe, 'atoms'):
                element_mask = sim_data.universe.atoms.elements == element
                element_indices = np.where(element_mask)[0]
            else:
                # Fallback: assume only diffusing element
                if element == sim_data.diff_elem:
                    element_indices = np.where(sim_data.diffusing_atom)[0]
                else:
                    continue
            
            if len(element_indices) > 0:
                avg_displacement = np.mean(sim_data.displacement[element_indices, :], axis=0)
                color = self.default_colors[i % len(self.default_colors)]
                plt.plot(time_axis, avg_displacement, color=color, linewidth=2.5, 
                        label=f'{element} (n={len(element_indices)})')
        
        plt.title('Average Displacement per Element')
        plt.xlabel(xlabel)
        plt.ylabel('Displacement (Å)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Plot saved: {output_file}")
        
        plt.show()
    
    def plot_3d_density_with_sites(self, sim_data, sites_cart=None, resolution=0.5, 
                                  iso_values=[0.25, 0.10, 0.02], colors=['red', 'yellow', 'cyan'],
                                  alphas=[0.5, 0.3, 0.1], smooth_sigma=1.0, 
                                  angstrom_spacing=2, output_file=None):
        """
        Create 3D density plot with isosurfaces and optional site overlay.
        
        Args:
            sim_data (SimData): The simulation data object
            sites_cart (np.array, optional): Site coordinates to overlay
            resolution (float): Grid resolution in Angstroms
            iso_values (list): Isosurface values as fractions of maximum density
            colors (list): Colors for isosurfaces
            alphas (list): Alpha values for isosurfaces
            smooth_sigma (float): Gaussian smoothing parameter
            angstrom_spacing (int): Spacing for axis tick labels
            output_file (str, optional): Path to save the plot
        """
        # Get box dimensions
        box_dims = sim_data.universe.dimensions[:3]
        
        # Calculate grid dimensions
        grid_dims = [int(np.ceil(dim / resolution)) for dim in box_dims]
        
        print(f"Creating 3D density grid: {grid_dims[0]} x {grid_dims[1]} x {grid_dims[2]}")
        
        # Initialize density grid
        density_grid = np.zeros(grid_dims)
        
        # Fill density grid
        for atom_idx in range(sim_data.nr_atoms):
            if sim_data.diffusing_atom[atom_idx]:
                for time_idx in range(sim_data.nr_steps):
                    pos = sim_data.cart_pos[:, atom_idx, time_idx]
                    
                    # Convert to grid indices
                    grid_pos = np.floor(pos / resolution).astype(int)
                    
                    # Handle boundary conditions
                    for i in range(3):
                        if grid_pos[i] < 0:
                            grid_pos[i] += grid_dims[i]
                        elif grid_pos[i] >= grid_dims[i]:
                            grid_pos[i] -= grid_dims[i]
                    
                    density_grid[grid_pos[0], grid_pos[1], grid_pos[2]] += 1
        
        # Apply Gaussian smoothing
        if smooth_sigma > 0:
            density_grid = gaussian_filter(density_grid, sigma=smooth_sigma)
        
        # Create 3D plot
        fig = plt.figure(figsize=(14, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # Draw lattice box
        self._draw_lattice_box(ax, grid_dims, resolution)
        
        # Create isosurfaces
        max_density = np.max(density_grid)
        print(f"Maximum density: {max_density}")
        
        # Fix axis permutation issue (as mentioned in MATLAB code)
        density_grid_plot = np.transpose(density_grid, (1, 0, 2))
        
        # Create meshgrid for isosurfaces
        x, y, z = np.meshgrid(np.arange(grid_dims[1]), 
                             np.arange(grid_dims[0]), 
                             np.arange(grid_dims[2]))
        
        for i, iso_val in enumerate(iso_values):
            if i >= len(colors) or i >= len(alphas):
                break
                
            threshold = iso_val * max_density
            if threshold > 0:
                try:
                    from skimage.measure import marching_cubes
                    verts, faces, _, _ = marching_cubes(density_grid_plot, level=threshold)
                    
                    # Scale vertices to real coordinates
                    verts = verts * resolution
                    
                    # Create 3D surface
                    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
                    mesh = Poly3DCollection(verts[faces])
                    mesh.set_facecolor(colors[i])
                    mesh.set_alpha(alphas[i])
                    mesh.set_edgecolor('none')
                    ax.add_collection3d(mesh)
                    
                except ImportError:
                    print("Warning: scikit-image not available for isosurface generation")
                    # Fallback: show density as scatter plot
                    indices = np.where(density_grid > threshold)
                    if len(indices[0]) > 0:
                        ax.scatter(indices[1] * resolution, indices[0] * resolution, 
                                 indices[2] * resolution, c=colors[i], alpha=alphas[i], s=20)
        
        # Add sites if provided
        if sites_cart is not None:
            ax.scatter(sites_cart[:, 0], sites_cart[:, 1], sites_cart[:, 2], 
                      c='blue', s=100, marker='o', alpha=0.8, label='Sites')
            
            # Add site labels
            delta = 0.2
            for i, site_pos in enumerate(sites_cart):
                ax.text(site_pos[0] + delta, site_pos[1] + delta, site_pos[2] + delta, 
                       f'S{i+1}', fontsize=8)
        
        # Set labels and limits
        ax.set_xlabel('X (Å)')
        ax.set_ylabel('Y (Å)')
        ax.set_zlabel('Z (Å)')
        ax.set_title('3D Density of Diffusing Element')
        
        # Set axis limits
        ax.set_xlim(0, box_dims[0])
        ax.set_ylim(0, box_dims[1])
        ax.set_zlim(0, box_dims[2])
        
        # Set tick spacing
        x_ticks = np.arange(0, box_dims[0] + angstrom_spacing, angstrom_spacing)
        y_ticks = np.arange(0, box_dims[1] + angstrom_spacing, angstrom_spacing)
        z_ticks = np.arange(0, box_dims[2] + angstrom_spacing, angstrom_spacing)
        
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        ax.set_zticks(z_ticks)
        
        ax.view_init(elev=10, azim=-10)
        ax.grid(True)
        
        if sites_cart is not None:
            ax.legend()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Plot saved: {output_file}")
        
        plt.show()
    
    def _draw_lattice_box(self, ax, grid_dims, resolution):
        """Draw the simulation box outline."""
        # Convert grid dimensions to real coordinates
        dims = [dim * resolution for dim in grid_dims]
        
        # Define box vertices
        vertices = [
            [0, 0, 0], [dims[0], 0, 0], [dims[0], dims[1], 0], [0, dims[1], 0],  # bottom
            [0, 0, dims[2]], [dims[0], 0, dims[2]], [dims[0], dims[1], dims[2]], [0, dims[1], dims[2]]  # top
        ]
        
        # Define box edges
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # bottom edges
            [4, 5], [5, 6], [6, 7], [7, 4],  # top edges
            [0, 4], [1, 5], [2, 6], [3, 7]   # vertical edges
        ]
        
        # Draw edges
        for edge in edges:
            start, end = vertices[edge[0]], vertices[edge[1]]
            ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 
                   'k-', linewidth=1, alpha=0.3)
    
    def plot_all_displacement_analyses(self, sim_data, sites_cart=None, timestep=None,
                                     resolution=0.5, output_dir=None):
        """
        Generate all displacement-based plots in one function call.
        
        Args:
            sim_data (SimData): The simulation data object
            sites_cart (np.array, optional): Site coordinates for 3D plot
            timestep (float, optional): Timestep for time conversion
            resolution (float): Grid resolution for 3D density plot
            output_dir (str, optional): Directory to save all plots
        """
        print("🔄 Generating displacement analysis plots...")
        
        # Individual displacements
        self.plot_individual_displacements(
            sim_data, timestep=timestep, max_atoms=50,
            output_file=f"{output_dir}/individual_displacements.png" if output_dir else None
        )
        
        # Displacement histogram
        self.plot_displacement_histogram(
            sim_data, bins=50,
            output_file=f"{output_dir}/displacement_histogram.png" if output_dir else None
        )
        
        # Displacement per element
        self.plot_displacement_per_element(
            sim_data, timestep=timestep,
            output_file=f"{output_dir}/displacement_per_element.png" if output_dir else None
        )
        
        # 3D density plot
        self.plot_3d_density_with_sites(
            sim_data, sites_cart=sites_cart, resolution=resolution,
            output_file=f"{output_dir}/3d_density_with_sites.png" if output_dir else None
        )
        
        print("✅ All displacement plots generated!")
    
    def set_plot_style(self, figure_size=None, colors=None):
        """
        Customize the default plot styling.
        
        Args:
            figure_size (tuple, optional): Figure size as (width, height)
            colors (list, optional): List of colors for different elements
        """
        if figure_size is not None:
            self.default_figure_size = figure_size
            
        if colors is not None:
            self.default_colors = colors
            
        print(f"Plot style updated: figure_size={self.default_figure_size}")

