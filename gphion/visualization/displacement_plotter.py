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
        
    # -----------------
    # Utility functions
    # -----------------
    def _ensure_dir(self, path):
        if path and not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

    def _save_txt(self, filepath, array, header=None, fmt='%.8g'):
        self._ensure_dir(os.path.dirname(filepath) if os.path.dirname(filepath) else '.')
        np.savetxt(filepath, array, header=header if header else '', fmt=fmt)
        print(f"Data saved: {filepath}")
        
    # ------------------------------------
    # Plotting Functions (Plot and Save)
    # ------------------------------------
        
    def plot_individual_displacements(self, sim_data, timestep=None, max_atoms=None, 
                                    output_file=None, data_output_dir=None):
        """
        Plot displacement trajectories for each diffusing atom and save the combined data.
        """
        # --- Data Preparation ---
        time_frames = np.arange(sim_data.nr_steps)
        time_axis = time_frames if timestep is None else time_frames * timestep
        
        if sim_data.diffusing_atom is None:
            indices = range(sim_data.nr_atoms)
        else:
            indices = [i for i in range(sim_data.nr_atoms) if sim_data.diffusing_atom[i]]
        if max_atoms is not None:
            indices = list(indices)[:max_atoms]
            
        data_matrix = [time_axis]
        headers = ['time']
        for idx in indices:
            data_matrix.append(sim_data.displacement[idx, :])
            headers.append(f'atom_{idx}')
        data_matrix = np.vstack(data_matrix).T

        # --- Plotting ---
        plt.figure(figsize=self.default_figure_size)
        
        xlabel = 'Time (ps)' if timestep is not None else 'Time step'
        
        for i, idx in enumerate(indices):
            # Plot using the prepared data
            plt.plot(time_axis, sim_data.displacement[idx, :], alpha=0.7, linewidth=1)
        
        plt.title('Displacement of Diffusing Element')
        plt.xlabel(xlabel)
        plt.ylabel('Displacement (Å)')
        plt.grid(True, alpha=0.3)
        
        if max_atoms is not None and len(indices) == max_atoms:
            plt.text(0.02, 0.98, f'Showing {max_atoms} of {sim_data.nr_diffusing} atoms', 
                    transform=plt.gca().transAxes, verticalalignment='top', 
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Plot saved: {output_file}")
        
        plt.show()

        # --- Data Saving (Integration) ---
        if data_output_dir:
            base = os.path.join(data_output_dir, 'individual_displacements')
            self._save_txt(os.path.join(base, 'displacements_matrix.txt'), data_matrix,
                           header='\t'.join(headers))


    def plot_displacement_histogram(self, sim_data, bins=50, output_file=None, data_output_dir=None):
        """
        Plot histogram of final displacements for diffusing atoms and save the data.
        """
        # --- Data Preparation ---
        final_displacements = []
        if sim_data.diffusing_atom is None:
            final_displacements = [sim_data.displacement[i, -1] for i in range(sim_data.nr_atoms)]
        else:
            final_displacements = [sim_data.displacement[i, -1] for i in range(sim_data.nr_atoms) if sim_data.diffusing_atom[i]]
        final_displacements = np.array(final_displacements)
        
        # Calculate histogram data (used for both plot and save)
        counts, bin_edges = np.histogram(final_displacements, bins=bins)
        
        # --- Plotting ---
        plt.figure(figsize=self.default_figure_size)
        plt.hist(final_displacements, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('Histogram of Final Displacement of Diffusing Element')
        plt.xlabel('Displacement (Å)')
        plt.ylabel('Number of atoms')
        plt.grid(True, alpha=0.3)
        
        mean_disp = np.mean(final_displacements)
        std_disp = np.std(final_displacements)
        
        stats_text = f'Mean: {mean_disp:.2f} Å\nStd: {std_disp:.2f} Å\nAtoms: {len(final_displacements)}'
        plt.text(0.98, 0.98, stats_text, transform=plt.gca().transAxes, 
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Plot saved: {output_file}")
        
        plt.show()

        # --- Data Saving (Integration) ---
        if data_output_dir:
            base = os.path.join(data_output_dir, 'displacement_histogram')
            self._ensure_dir(base)
            
            # Consolidate into a single re-plottable file: bin_left_edge and count
            hist_data = np.column_stack([bin_edges[:-1], counts])
            self._save_txt(os.path.join(base, 'histogram_data.txt'), hist_data, 
                           header='bin_left_edge\tcount', fmt='%.8g')

    
    def plot_displacement_per_element(self, sim_data, elements=None, timestep=None, 
                                    output_file=None, data_output_dir=None):
        """
        Plot average displacement per element type and save the data.
        """
        # --- Data Preparation ---
        time_frames = np.arange(sim_data.nr_steps)
        time_axis = time_frames if timestep is None else time_frames * timestep
        xlabel = 'Time (ps)' if timestep is not None else 'Time step'

        if hasattr(sim_data.universe, 'atoms') and hasattr(sim_data.universe.atoms, 'elements'):
            unique_elements = np.unique(sim_data.universe.atoms.elements)
        elif elements is not None:
            unique_elements = elements
        else:
            unique_elements = [sim_data.diff_elem]
        
        colors = self.default_colors
        color_map = {element: colors[i % len(colors)] for i, element in enumerate(unique_elements)}
        
        data_matrix = [time_axis]
        headers = ['time']

        # --- Plotting ---
        plt.figure(figsize=self.default_figure_size)
        
        for element in unique_elements:
            element_indices = []
            if hasattr(sim_data.universe, 'atoms'):
                element_mask = sim_data.universe.atoms.elements == element
                element_indices = np.where(element_mask)[0]
            elif element == sim_data.diff_elem:
                element_indices = np.where(sim_data.diffusing_atom)[0] if sim_data.diffusing_atom is not None else np.arange(sim_data.nr_atoms)
            
            if len(element_indices) > 0:
                avg_displacement = np.mean(sim_data.displacement[element_indices, :], axis=0)
                
                # Plot
                plt.plot(time_axis, avg_displacement, label=element, color=color_map[element], linewidth=2)
                
                # Collect for saving
                data_matrix.append(avg_displacement)
                headers.append(str(element))

        plt.title('Average Displacement of Elements')
        plt.xlabel(xlabel)
        plt.ylabel('Displacement (Å)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Plot saved: {output_file}")
        
        plt.show()

        # --- Data Saving (Integration) ---
        if data_output_dir and len(data_matrix) > 1:
            base = os.path.join(data_output_dir, 'displacement_per_element')
            data_matrix = np.vstack(data_matrix).T
            self._save_txt(os.path.join(base, 'avg_displacement_per_element.txt'), data_matrix, header='\t'.join(headers))
    
    
    def plot_3d_density_with_sites(self, sim_data, sites_cart=None, resolution=0.5, output_file=None, data_output_dir=None):
        """
        Plot 3D isosurface density and overlay sites if provided, and save the data.
        """
        try:
            # --- Data Preparation ---
            if not hasattr(sim_data, 'atom_positions'):
                print("SimData missing 'atom_positions' attribute for 3D density plot.")
                return
            
            positions = sim_data.atom_positions
            positions_flat = positions.reshape(-1, 3)
            
            xyz_min = positions_flat.min(axis=0) - resolution
            xyz_max = positions_flat.max(axis=0) + resolution
            bins = [np.arange(xyz_min[i], xyz_max[i], resolution) for i in range(3)]
            H, edges = np.histogramdd(positions_flat, bins=bins)
            H_smooth = gaussian_filter(H, sigma=1)
            
            # --- Plotting ---
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            x_centers = (edges[0][:-1] + edges[0][1:]) / 2
            y_centers = (edges[1][:-1] + edges[1][1:]) / 2
            z_centers = (edges[2][:-1] + edges[2][1:]) / 2
            
            X, Y, Z = np.meshgrid(x_centers, y_centers, z_centers, indexing='ij')
            
            isovalue = H_smooth.max() * 0.2
            voxels = H_smooth > isovalue
            
            ax.voxels(X, Y, Z, voxels, facecolors='cyan', edgecolor='k', alpha=0.3)
            
            if sites_cart is not None:
                ax.scatter(sites_cart[:, 0], sites_cart[:, 1], sites_cart[:, 2], color='red', s=50, label='Sites')
            
            ax.set_title('3D Density of Displacements with Sites')
            ax.set_xlabel('X (Å)')
            ax.set_ylabel('Y (Å)')
            ax.set_zlabel('Z (Å)')
            ax.legend()
            
            if output_file:
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                print(f"Plot saved: {output_file}")
            
            plt.show()

            # --- Data Saving (Integration) ---
            if data_output_dir:
                base = os.path.join(data_output_dir, '3d_density_data')
                self._ensure_dir(base)

                # Save the 3D data array and edges to a single compressed .npz file
                filepath = os.path.join(base, '3d_density_data.npz')
                np.savez_compressed(filepath, 
                                   density_smooth=H_smooth,
                                   edges_x=edges[0],
                                   edges_y=edges[1],
                                   edges_z=edges[2])
                print(f"3D density data (smoothed array and bin edges) saved to single file: {filepath}")
        
        except Exception as e:
            print(f"Error generating 3D density plot: {e}")

    # ------------------------------------
    # Master Function
    # ------------------------------------

    def plot_all_displacement_analyses(self, sim_data, sites_cart=None, timestep=None,
                                       resolution=0.5, output_dir=None, export_data=False):
        """
        Generate all displacement-based plots. Data is automatically exported if
        output_dir is provided and export_data is True.
        
        Args:
            sim_data (SimData): Simulation data
            sites_cart (np.ndarray, optional): Site coordinates for overlay
            timestep (float, optional): Timestep size
            resolution (float): Resolution for 3D density grid
            output_dir (str, optional): Directory to save plots and data
            export_data (bool): Whether to export data files (triggers data_output_dir argument)
        """
        if output_dir:
            self._ensure_dir(output_dir)
        
        # Pass output_dir to data_output_dir only if export_data is True
        data_dir_arg = output_dir if export_data else None

        print("🔄 Generating displacement analysis plots...")
        
        # Individual displacement plots
        self.plot_individual_displacements(sim_data, timestep=timestep, max_atoms=50,
                                           output_file=os.path.join(output_dir, 'individual_displacements.png') if output_dir else None,
                                           data_output_dir=data_dir_arg)
        
        # Histogram
        self.plot_displacement_histogram(sim_data, bins=50,
                                         output_file=os.path.join(output_dir, 'displacement_histogram.png') if output_dir else None,
                                         data_output_dir=data_dir_arg)
        
        # Per element displacement
        self.plot_displacement_per_element(sim_data, timestep=timestep,
                                           output_file=os.path.join(output_dir, 'displacement_per_element.png') if output_dir else None,
                                           data_output_dir=data_dir_arg)
        
        # 3D density with sites
        self.plot_3d_density_with_sites(sim_data, sites_cart=sites_cart, resolution=resolution,
                                       output_file=os.path.join(output_dir, '3d_density_with_sites.png') if output_dir else None,
                                       data_output_dir=data_dir_arg)
        
        print("✅ All displacement plots generated!")