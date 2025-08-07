"""
3D Atomic Migration Path Visualization in Crystal/Amorphous Systems
Combines jump analysis data with site discovery for scientific visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
from scipy.spatial.distance import cdist
from scipy.interpolate import interp1d
import os

class MigrationPathVisualizer:
    """
    Visualize 3D atomic migration paths combining jump analysis and site discovery data.
    """
    
    def __init__(self, jump_analysis, site_finder, sim_data):
        """
        Initialize the visualizer with analysis results.
        
        Args:
            jump_analysis: JumpAnalyzer instance with completed analysis
            site_finder: AmorphousSiteFinder instance with discovered sites
            sim_data: SimData object with trajectory information
        """
        self.jump_analysis = jump_analysis
        self.site_finder = site_finder
        self.sim_data = sim_data
        
        # Load discovered sites
        self.site_coordinates = site_finder.find_sites_from_density()
        self.n_sites = len(self.site_coordinates)
        
        # Get box dimensions
        self.box_dimensions = sim_data.universe.dimensions[:3]  # x, y, z dimensions
        
        # Build migration network
        self.migration_network = None
        self.migration_paths = []
        
    def build_migration_network(self):
        """Create a network graph of migration pathways."""
        print("Building migration network from jump data...")
        
        self.migration_network = nx.DiGraph()
        
        # Add sites as nodes with properties
        for i, site_pos in enumerate(self.site_coordinates):
            occupancy = self.jump_analysis.results.site_occupancies[i] if i < len(self.jump_analysis.results.site_occupancies) else 0
            occupancy_factor = self.jump_analysis.results.average_occupancy_factors[i] if i < len(self.jump_analysis.results.average_occupancy_factors) else 0
            
            self.migration_network.add_node(i, 
                                          pos=site_pos,
                                          occupancy=occupancy,
                                          occupancy_factor=occupancy_factor)
        
        # Add edges from jump events
        jump_counts = {}
        total_residence_times = {}
        
        if self.jump_analysis.results.all_trans.size > 0:
            for jump in self.jump_analysis.results.all_trans:
                atom_id, from_site, to_site, time = jump
                from_idx = int(from_site) - 1  # Convert to 0-based
                to_idx = int(to_site) - 1
                
                if 0 <= from_idx < self.n_sites and 0 <= to_idx < self.n_sites:
                    edge_key = (from_idx, to_idx)
                    
                    # Count jumps
                    jump_counts[edge_key] = jump_counts.get(edge_key, 0) + 1
                    
                    # Calculate path properties
                    path_vector = self._get_minimum_image_vector(
                        self.site_coordinates[from_idx],
                        self.site_coordinates[to_idx]
                    )
                    path_length = np.linalg.norm(path_vector)
                    
                    # Add or update edge
                    if self.migration_network.has_edge(from_idx, to_idx):
                        self.migration_network[from_idx][to_idx]['weight'] += 1
                        self.migration_network[from_idx][to_idx]['times'].append(time)
                    else:
                        self.migration_network.add_edge(from_idx, to_idx,
                                                      weight=1,
                                                      path_length=path_length,
                                                      path_vector=path_vector,
                                                      times=[time])
        
        # Calculate edge statistics
        for edge in self.migration_network.edges():
            from_idx, to_idx = edge
            times = self.migration_network[from_idx][to_idx]['times']
            self.migration_network[from_idx][to_idx]['frequency'] = len(times)
            self.migration_network[from_idx][to_idx]['time_span'] = max(times) - min(times) if len(times) > 1 else 0
            
        print(f"Migration network built: {self.migration_network.number_of_nodes()} sites, "
              f"{self.migration_network.number_of_edges()} pathways")
    
    def _get_minimum_image_vector(self, pos1, pos2):
        """Calculate minimum image vector considering periodic boundaries."""
        delta = pos2 - pos1
        
        # Apply minimum image convention
        for i in range(3):
            if delta[i] > self.box_dimensions[i] / 2:
                delta[i] -= self.box_dimensions[i]
            elif delta[i] < -self.box_dimensions[i] / 2:
                delta[i] += self.box_dimensions[i]
                
        return delta
    
    def create_3d_static_visualization(self, save_path='migration_3d_static.png', 
                                     show_all_sites=True, min_jump_frequency=1):
        """Create a static 3D visualization of migration paths."""
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Draw simulation box
        self._draw_simulation_box(ax)
        
        # Plot all sites
        if show_all_sites:
            occupancies = np.array([self.migration_network.nodes[i].get('occupancy', 0) 
                                  for i in range(self.n_sites)])
            
            # Normalize sizes
            site_sizes = 50 + 200 * (occupancies / np.max(occupancies) if np.max(occupancies) > 0 else 0)
            
            scatter = ax.scatter(self.site_coordinates[:, 0],
                               self.site_coordinates[:, 1], 
                               self.site_coordinates[:, 2],
                               s=site_sizes,
                               c=occupancies,
                               cmap='viridis',
                               alpha=0.6,
                               label='Sites')
            plt.colorbar(scatter, label='Average Occupancy (atoms)', shrink=0.8)
        
        # Draw migration paths
        if self.migration_network:
            for edge in self.migration_network.edges():
                from_idx, to_idx = edge
                frequency = self.migration_network[from_idx][to_idx]['frequency']
                
                if frequency >= min_jump_frequency:
                    start_pos = self.site_coordinates[from_idx]
                    end_pos = self.site_coordinates[to_idx]
                    
                    # Calculate actual path considering PBC
                    path_vector = self._get_minimum_image_vector(start_pos, end_pos)
                    actual_end_pos = start_pos + path_vector
                    
                    # Line width proportional to frequency
                    line_width = 1 + np.log10(frequency)
                    
                    ax.plot([start_pos[0], actual_end_pos[0]],
                           [start_pos[1], actual_end_pos[1]],
                           [start_pos[2], actual_end_pos[2]],
                           'r-', linewidth=line_width, alpha=0.7)
                    
                    # Add arrow for direction
                    mid_point = start_pos + 0.5 * path_vector
                    arrow_vec = 0.1 * path_vector / np.linalg.norm(path_vector)
                    ax.quiver(mid_point[0], mid_point[1], mid_point[2],
                             arrow_vec[0], arrow_vec[1], arrow_vec[2],
                             color='red', alpha=0.8, arrow_length_ratio=0.1)
        
        ax.set_xlabel('X (Å)')
        ax.set_ylabel('Y (Å)')
        ax.set_zlabel('Z (Å)')
        ax.set_title('3D Atomic Migration Pathways')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Static 3D visualization saved to: {save_path}")
        
        return fig, ax
    
    def create_interactive_3d_visualization(self, save_path='migration_3d_interactive.html',
                                          min_jump_frequency=1, show_time_evolution=True):
        """Create an interactive 3D visualization using Plotly."""
        fig = go.Figure()
        
        # Add simulation box outline
        box_traces = self._create_box_traces()
        for trace in box_traces:
            fig.add_trace(trace)
        
        # Add sites
        occupancies = np.array([self.migration_network.nodes[i].get('occupancy', 0) 
                              for i in range(self.n_sites)])
        occupancy_factors = np.array([self.migration_network.nodes[i].get('occupancy_factor', 0) 
                                    for i in range(self.n_sites)])
        
        # Site sizes proportional to occupancy
        site_sizes = 5 + 15 * (occupancies / np.max(occupancies) if np.max(occupancies) > 0 else 0)
        
        fig.add_trace(go.Scatter3d(
            x=self.site_coordinates[:, 0],
            y=self.site_coordinates[:, 1],
            z=self.site_coordinates[:, 2],
            mode='markers',
            marker=dict(
                size=site_sizes,
                color=occupancies,
                colorscale='Viridis',
                colorbar=dict(title="Average Occupancy (atoms)", x=1.02),
                opacity=0.8,
                line=dict(width=1, color='black')
            ),
            text=[f"Site {i}<br>Occupancy: {occ:.3f}<br>Factor: {factor:.4f}"
                  for i, (occ, factor) in enumerate(zip(occupancies, occupancy_factors))],
            hovertemplate='%{text}<extra></extra>',
            name='Sites'
        ))
        
        # Add migration paths
        if self.migration_network:
            path_x, path_y, path_z = [], [], []
            path_info = []
            
            for edge in self.migration_network.edges():
                from_idx, to_idx = edge
                frequency = self.migration_network[from_idx][to_idx]['frequency']
                
                if frequency >= min_jump_frequency:
                    start_pos = self.site_coordinates[from_idx]
                    path_vector = self._get_minimum_image_vector(
                        start_pos, self.site_coordinates[to_idx]
                    )
                    end_pos = start_pos + path_vector
                    
                    # Create smooth path with multiple points
                    n_points = max(5, int(frequency / 2))
                    for i in range(n_points + 1):
                        t = i / n_points
                        point = start_pos + t * path_vector
                        path_x.append(point[0])
                        path_y.append(point[1])
                        path_z.append(point[2])
                    
                    # Add separator (None) between paths
                    path_x.append(None)
                    path_y.append(None)
                    path_z.append(None)
                    
                    path_info.extend([f"Path {from_idx}→{to_idx}<br>Frequency: {frequency}"] * (n_points + 1) + [None])
            
            if path_x:
                fig.add_trace(go.Scatter3d(
                    x=path_x,
                    y=path_y,
                    z=path_z,
                    mode='lines',
                    line=dict(color='red', width=4),
                    text=path_info,
                    hovertemplate='%{text}<extra></extra>',
                    name='Migration Paths'
                ))
        
        # Layout
        fig.update_layout(
            title="Interactive 3D Atomic Migration Pathways",
            scene=dict(
                xaxis_title="X (Å)",
                yaxis_title="Y (Å)",
                zaxis_title="Z (Å)",
                aspectmode='cube',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            width=1000,
            height=800,
            showlegend=True
        )
        
        # Save interactive plot
        fig.write_html(save_path)
        print(f"Interactive 3D visualization saved to: {save_path}")
        
        return fig
    
    def create_migration_time_analysis(self, save_path='migration_time_analysis.png'):
        """Analyze and visualize temporal aspects of migration."""
        if self.jump_analysis.results.all_trans.size == 0:
            print("No jump data available for time analysis.")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Jump frequency over time
        times = self.jump_analysis.results.all_trans[:, 3]
        time_bins = np.linspace(times.min(), times.max(), 50)
        counts, _ = np.histogram(times, bins=time_bins)
        ax1.plot(time_bins[:-1], counts, 'b-', linewidth=2)
        ax1.set_xlabel('Time (ps)')
        ax1.set_ylabel('Number of Jumps')
        ax1.set_title('Jump Frequency vs Time')
        ax1.grid(True, alpha=0.3)
        
        # 2. Site occupancy time series (if available)
        if hasattr(self.jump_analysis.results, 'occupancy_time_series'):
            time_points = np.arange(len(self.jump_analysis.results.occupancy_time_series))
            time_points = time_points * self.jump_analysis._trajectory.dt / self.jump_analysis.frame_gap
            
            for i in range(min(5, self.n_sites)):  # Show first 5 sites
                occupancy_series = self.jump_analysis.results.occupancy_time_series[:, i]
                ax2.plot(time_points, occupancy_series, label=f'Site {i}', alpha=0.7)
            
            ax2.set_xlabel('Time (ps)')
            ax2.set_ylabel('Site Occupancy (atoms)')
            ax2.set_title('Site Occupancy Evolution')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. Residence time distribution
        all_residence_times = []
        for site_idx in range(self.n_sites):
            if site_idx in self.jump_analysis.results.residence_time_stats:
                stats = self.jump_analysis.results.residence_time_stats[site_idx]
                if stats['count'] > 0:
                    # Approximate distribution from mean and std
                    mean_rt = stats['mean']
                    all_residence_times.extend([mean_rt] * stats['count'])
        
        if all_residence_times:
            ax3.hist(all_residence_times, bins=30, alpha=0.7, color='green', edgecolor='black')
            ax3.set_xlabel('Residence Time (ps)')
            ax3.set_ylabel('Frequency')
            ax3.set_title('Residence Time Distribution')
            ax3.grid(True, alpha=0.3)
        
        # 4. Migration distance distribution
        if self.migration_network:
            distances = []
            frequencies = []
            for edge in self.migration_network.edges():
                from_idx, to_idx = edge
                path_length = self.migration_network[from_idx][to_idx]['path_length']
                frequency = self.migration_network[from_idx][to_idx]['frequency']
                distances.extend([path_length] * frequency)
                frequencies.append(frequency)
            
            if distances:
                ax4.hist(distances, bins=30, alpha=0.7, color='orange', edgecolor='black')
                ax4.set_xlabel('Migration Distance (Å)')
                ax4.set_ylabel('Frequency')
                ax4.set_title('Migration Distance Distribution')
                ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Migration time analysis saved to: {save_path}")
        
        return fig
    
    def _draw_simulation_box(self, ax):
        """Draw the simulation box outline."""
        # Define box corners
        corners = np.array([
            [0, 0, 0], [self.box_dimensions[0], 0, 0],
            [self.box_dimensions[0], self.box_dimensions[1], 0],
            [0, self.box_dimensions[1], 0],
            [0, 0, self.box_dimensions[2]], [self.box_dimensions[0], 0, self.box_dimensions[2]],
            [self.box_dimensions[0], self.box_dimensions[1], self.box_dimensions[2]],
            [0, self.box_dimensions[1], self.box_dimensions[2]]
        ])
        
        # Define edges
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # bottom face
            [4, 5], [5, 6], [6, 7], [7, 4],  # top face
            [0, 4], [1, 5], [2, 6], [3, 7]   # vertical edges
        ]
        
        for edge in edges:
            start, end = corners[edge[0]], corners[edge[1]]
            ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 
                   'k--', alpha=0.3, linewidth=1)
    
    def _create_box_traces(self):
        """Create Plotly traces for simulation box outline."""
        traces = []
        
        # Box edges
        edges = [
            # Bottom face
            [[0, 0, 0], [self.box_dimensions[0], 0, 0]],
            [[self.box_dimensions[0], 0, 0], [self.box_dimensions[0], self.box_dimensions[1], 0]],
            [[self.box_dimensions[0], self.box_dimensions[1], 0], [0, self.box_dimensions[1], 0]],
            [[0, self.box_dimensions[1], 0], [0, 0, 0]],
            # Top face
            [[0, 0, self.box_dimensions[2]], [self.box_dimensions[0], 0, self.box_dimensions[2]]],
            [[self.box_dimensions[0], 0, self.box_dimensions[2]], [self.box_dimensions[0], self.box_dimensions[1], self.box_dimensions[2]]],
            [[self.box_dimensions[0], self.box_dimensions[1], self.box_dimensions[2]], [0, self.box_dimensions[1], self.box_dimensions[2]]],
            [[0, self.box_dimensions[1], self.box_dimensions[2]], [0, 0, self.box_dimensions[2]]],
            # Vertical edges
            [[0, 0, 0], [0, 0, self.box_dimensions[2]]],
            [[self.box_dimensions[0], 0, 0], [self.box_dimensions[0], 0, self.box_dimensions[2]]],
            [[self.box_dimensions[0], self.box_dimensions[1], 0], [self.box_dimensions[0], self.box_dimensions[1], self.box_dimensions[2]]],
            [[0, self.box_dimensions[1], 0], [0, self.box_dimensions[1], self.box_dimensions[2]]]
        ]
        
        for edge in edges:
            start, end = edge
            traces.append(go.Scatter3d(
                x=[start[0], end[0]],
                y=[start[1], end[1]],
                z=[start[2], end[2]],
                mode='lines',
                line=dict(color='gray', width=2),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        return traces
    
    def generate_migration_report(self, output_dir='migration_analysis'):
        """Generate comprehensive migration analysis report."""
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nGenerating comprehensive migration analysis in: {output_dir}")
        
        # Build network if not already done
        if self.migration_network is None:
            self.build_migration_network()
        
        # 1. Static 3D visualization
        static_path = os.path.join(output_dir, 'migration_3d_static.png')
        self.create_3d_static_visualization(static_path)
        
        # 2. Interactive 3D visualization
        interactive_path = os.path.join(output_dir, 'migration_3d_interactive.html')
        self.create_interactive_3d_visualization(interactive_path)
        
        # 3. Time analysis
        time_analysis_path = os.path.join(output_dir, 'migration_time_analysis.png')
        self.create_migration_time_analysis(time_analysis_path)
        
        # 4. Network analysis report
        self._generate_network_analysis_report(os.path.join(output_dir, 'migration_network_analysis.txt'))
        
        # 5. Migration statistics
        self._save_migration_statistics(os.path.join(output_dir, 'migration_statistics.txt'))
        
        print(f"Migration analysis complete! Check {output_dir}/ for all outputs.")

    def _generate_network_analysis_report(self, filepath):
        """Generate detailed network analysis report."""
        with open(filepath, 'w') as f:
            f.write("=== MIGRATION NETWORK ANALYSIS ===\n\n")
            
            f.write(f"Network Statistics:\n")
            f.write(f"- Number of sites: {self.migration_network.number_of_nodes()}\n")
            f.write(f"- Number of pathways: {self.migration_network.number_of_edges()}\n")
            f.write(f"- Network density: {nx.density(self.migration_network):.4f}\n")
            
            # Most connected sites
            degrees = dict(self.migration_network.degree())
            most_connected = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:5]
            
            f.write(f"\nMost Connected Sites:\n")
            for site, degree in most_connected:
                f.write(f"- Site {site}: {degree} connections\n")
            
            # Most frequent pathways
            edge_weights = [(edge, self.migration_network[edge[0]][edge[1]]['frequency']) 
                           for edge in self.migration_network.edges()]
            most_frequent = sorted(edge_weights, key=lambda x: x[1], reverse=True)[:10]
            
            f.write(f"\nMost Frequent Migration Pathways:\n")
            for (from_site, to_site), frequency in most_frequent:
                f.write(f"- Site {from_site} → Site {to_site}: {frequency} jumps\n")

    def _save_migration_statistics(self, filepath):
        """Save comprehensive migration statistics."""
        with open(filepath, 'w') as f:
            f.write("=== MIGRATION STATISTICS ===\n\n")
            
            total_jumps = len(self.jump_analysis.results.all_trans)
            f.write(f"Total Jumps: {total_jumps}\n")
            
            if total_jumps > 0:
                times = self.jump_analysis.results.all_trans[:, 3]
                f.write(f"Time Range: {times.min():.2f} - {times.max():.2f} ps\n")
                f.write(f"Average Jump Rate: {total_jumps / (times.max() - times.min()):.4f} jumps/ps\n")
                
                # Migration distances
                if self.migration_network:
                    distances = [self.migration_network[edge[0]][edge[1]]['path_length'] 
                               for edge in self.migration_network.edges()]
                    if distances:
                        f.write(f"\nMigration Distances:\n")
                        f.write(f"- Average: {np.mean(distances):.4f} Å\n")
                        f.write(f"- Min: {np.min(distances):.4f} Å\n")
                        f.write(f"- Max: {np.max(distances):.4f} Å\n")
                        f.write(f"- Std: {np.std(distances):.4f} Å\n")

# Usage Example
def create_migration_visualization(jump_analysis, site_finder, sim_data, output_dir='migration_visualization'):
    """
    Main function to create comprehensive migration visualization.
    
    Args:
        jump_analysis: Completed JumpAnalyzer instance
        site_finder: AmorphousSiteFinder instance 
        sim_data: SimData object
        output_dir: Directory for outputs
    """
    
    # Create visualizer
    visualizer = MigrationPathVisualizer(jump_analysis, site_finder, sim_data)
    
    # Generate complete analysis
    visualizer.generate_migration_report(output_dir)
    
    return visualizer

