"""Site connectivity and hopping pathway visualization utilities."""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
# Fix for Line3DCollection import
try:
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
except ImportError:
    # Fallback for older matplotlib versions
    from matplotlib.collections import LineCollection
    Line3DCollection = None

class ConnectivityPlotter:
    """
    A dedicated class for plotting site connectivity and hopping pathways.
    
    This class visualizes:
    - Site positions as nodes
    - Jump connections as edges with thickness/color based on probability
    - 3D network representation of diffusion pathways
    - 2D projections of connectivity networks
    """
    
    def __init__(self):
        """Initialize the Connectivity plotter."""
        self.default_figure_size = (14, 10)
        
    def plot_3d_site_connectivity(self, sites_cart, jump_data, connectivity_matrix=None,
                                 min_jumps=1, edge_width_scale=5, node_size=100,
                                 show_site_labels=True, colormap='plasma', 
                                 background_alpha=0.1, output_file=None):
        """
        Plot 3D site connectivity network with edges representing hopping pathways.
        
        Args:
            sites_cart (np.array): Site coordinates (N x 3)
            jump_data (np.array): Jump data with columns [atom_id, from_site, to_site, time]
            connectivity_matrix (np.array, optional): Pre-computed connectivity matrix
            min_jumps (int): Minimum number of jumps to show connection
            edge_width_scale (float): Scaling factor for edge thickness
            node_size (int): Size of site nodes
            show_site_labels (bool): Whether to show site labels
            colormap (str): Colormap for edge coloring based on jump frequency
            background_alpha (float): Alpha for background elements
            output_file (str, optional): Path to save the plot
        """
        if connectivity_matrix is None:
            connectivity_matrix = self._build_connectivity_matrix(sites_cart, jump_data)
        
        fig = plt.figure(figsize=self.default_figure_size)
        ax = fig.add_subplot(111, projection='3d')
        
        n_sites = len(sites_cart)
        
        # Plot site nodes
        ax.scatter(sites_cart[:, 0], sites_cart[:, 1], sites_cart[:, 2], 
                  s=node_size, c='blue', alpha=0.8, marker='o', 
                  edgecolors='darkblue', linewidth=2, label='Hopping Sites')
        
        # Add site labels
        if show_site_labels:
            for i, pos in enumerate(sites_cart):
                ax.text(pos[0], pos[1], pos[2], f'S{i+1}', 
                       fontsize=10, fontweight='bold', color='darkblue')
        
        # Prepare and plot edges
        max_jumps = np.max(connectivity_matrix[connectivity_matrix > 0]) if np.any(connectivity_matrix > 0) else 1
        
        # Plot edges using individual lines (more compatible approach)
        edge_colors_list = []
        for i in range(n_sites):
            for j in range(i+1, n_sites):  # Avoid duplicate edges
                jump_count = connectivity_matrix[i, j] + connectivity_matrix[j, i]  # Bidirectional
                
                if jump_count >= min_jumps:
                    # Create edge
                    start_pos = sites_cart[i]
                    end_pos = sites_cart[j]
                    
                    # Edge properties based on jump frequency
                    normalized_weight = jump_count / max_jumps
                    line_width = normalized_weight * edge_width_scale
                    
                    # Choose color based on jump frequency
                    cmap = plt.cm.get_cmap(colormap)
                    color = cmap(normalized_weight)
                    
                    # Plot individual line
                    ax.plot([start_pos[0], end_pos[0]], 
                           [start_pos[1], end_pos[1]], 
                           [start_pos[2], end_pos[2]], 
                           color=color, linewidth=line_width, alpha=0.7)
                    
                    edge_colors_list.append(normalized_weight)
        
        # Add colorbar if we have edges
        if edge_colors_list:
            # Create a dummy mappable for colorbar
            import matplotlib.cm as cm
            cmap = cm.get_cmap(colormap)
            norm = plt.Normalize(vmin=0, vmax=1)
            mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
            mappable.set_array([])
            
            cbar = plt.colorbar(mappable, ax=ax, shrink=0.6, pad=0.1)
            cbar.set_label('Relative Jump Frequency', fontsize=12)
        
        # Customize plot
        ax.set_xlabel('X (Å)', fontsize=12)
        ax.set_ylabel('Y (Å)', fontsize=12)
        ax.set_zlabel('Z (Å)', fontsize=12)
        ax.set_title('3D Site Connectivity Network\n(Edge thickness ∝ Jump Frequency)', fontsize=14)
        
        # Set equal aspect ratio
        max_range = np.array([sites_cart[:,0].max()-sites_cart[:,0].min(),
                             sites_cart[:,1].max()-sites_cart[:,1].min(),
                             sites_cart[:,2].max()-sites_cart[:,2].min()]).max() / 2.0
        mid_x = (sites_cart[:,0].max()+sites_cart[:,0].min()) * 0.5
        mid_y = (sites_cart[:,1].max()+sites_cart[:,1].min()) * 0.5
        mid_z = (sites_cart[:,2].max()+sites_cart[:,2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        ax.legend()
        ax.grid(True, alpha=background_alpha)
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"3D connectivity plot saved: {output_file}")
        
        plt.show()
        
        return connectivity_matrix
    
    def plot_2d_site_connectivity(self, sites_cart, jump_data, projection='xy',
                                 connectivity_matrix=None, min_jumps=1, 
                                 edge_width_scale=10, node_size=200,
                                 show_site_labels=True, output_file=None):
        """
        Plot 2D projection of site connectivity network.
        
        Args:
            sites_cart (np.array): Site coordinates
            jump_data (np.array): Jump data
            projection (str): Projection plane ('xy', 'xz', 'yz')
            connectivity_matrix (np.array, optional): Pre-computed connectivity matrix
            min_jumps (int): Minimum jumps to show connection
            edge_width_scale (float): Edge thickness scaling
            node_size (int): Node size
            show_site_labels (bool): Whether to show site labels
            output_file (str, optional): Path to save plot
        """
        if connectivity_matrix is None:
            connectivity_matrix = self._build_connectivity_matrix(sites_cart, jump_data)
        
        # Select projection coordinates
        if projection == 'xy':
            x_idx, y_idx = 0, 1
            xlabel, ylabel = 'X (Å)', 'Y (Å)'
        elif projection == 'xz':
            x_idx, y_idx = 0, 2
            xlabel, ylabel = 'X (Å)', 'Z (Å)'
        elif projection == 'yz':
            x_idx, y_idx = 1, 2
            xlabel, ylabel = 'Y (Å)', 'Z (Å)'
        else:
            raise ValueError("Projection must be 'xy', 'xz', or 'yz'")
        
        plt.figure(figsize=self.default_figure_size)
        
        # Plot connections first (so they appear behind nodes)
        max_jumps = np.max(connectivity_matrix[connectivity_matrix > 0]) if np.any(connectivity_matrix > 0) else 1
        
        for i in range(len(sites_cart)):
            for j in range(i+1, len(sites_cart)):
                jump_count = connectivity_matrix[i, j] + connectivity_matrix[j, i]
                
                if jump_count >= min_jumps:
                    x_coords = [sites_cart[i, x_idx], sites_cart[j, x_idx]]
                    y_coords = [sites_cart[i, y_idx], sites_cart[j, y_idx]]
                    
                    # Line properties
                    line_width = (jump_count / max_jumps) * edge_width_scale
                    alpha = 0.3 + 0.7 * (jump_count / max_jumps)  # Variable transparency
                    
                    plt.plot(x_coords, y_coords, 'r-', linewidth=line_width, 
                            alpha=alpha, solid_capstyle='round')
        
        # Plot site nodes
        plt.scatter(sites_cart[:, x_idx], sites_cart[:, y_idx], 
                   s=node_size, c='blue', alpha=0.8, marker='o', 
                   edgecolors='darkblue', linewidth=2, zorder=5)
        
        # Add site labels
        if show_site_labels:
            for i, pos in enumerate(sites_cart):
                plt.annotate(f'S{i+1}', (pos[x_idx], pos[y_idx]), 
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=10, fontweight='bold', color='darkblue')
        
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.title(f'Site Connectivity Network ({projection.upper()} projection)\n'
                 f'Line thickness ∝ Jump Frequency', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
                   markersize=10, label='Hopping Sites'),
            Line2D([0], [0], color='red', linewidth=3, alpha=0.7, 
                   label='Jump Connections')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"2D connectivity plot saved: {output_file}")
        
        plt.show()
    
    def plot_connectivity_network_graph(self, sites_cart, jump_data, 
                                       connectivity_matrix=None, min_jumps=1,
                                       layout='spring', node_size=500, 
                                       show_edge_labels=False, output_file=None):
        """
        Plot connectivity as a network graph using NetworkX.
        
        Args:
            sites_cart (np.array): Site coordinates
            jump_data (np.array): Jump data
            connectivity_matrix (np.array, optional): Pre-computed connectivity matrix
            min_jumps (int): Minimum jumps to show connection
            layout (str): Network layout ('spring', 'circular', 'random')
            node_size (int): Size of nodes
            show_edge_labels (bool): Whether to show jump counts on edges
            output_file (str, optional): Path to save plot
        """
        if connectivity_matrix is None:
            connectivity_matrix = self._build_connectivity_matrix(sites_cart, jump_data)
        
        # Create NetworkX graph
        G = nx.Graph()
        
        # Add nodes
        for i in range(len(sites_cart)):
            G.add_node(i, pos=sites_cart[i], label=f'S{i+1}')
        
        # Add edges
        edge_weights = {}
        max_jumps = np.max(connectivity_matrix[connectivity_matrix > 0]) if np.any(connectivity_matrix > 0) else 1
        
        for i in range(len(sites_cart)):
            for j in range(i+1, len(sites_cart)):
                jump_count = connectivity_matrix[i, j] + connectivity_matrix[j, i]
                
                if jump_count >= min_jumps:
                    weight = jump_count / max_jumps
                    G.add_edge(i, j, weight=weight, jumps=jump_count)
                    edge_weights[(i, j)] = weight
        
        plt.figure(figsize=self.default_figure_size)
        
        # Choose layout
        if layout == 'spring':
            pos = nx.spring_layout(G, k=2, iterations=50)
        elif layout == 'circular':
            pos = nx.circular_layout(G)
        elif layout == 'random':
            pos = nx.random_layout(G)
        else:
            pos = nx.spring_layout(G)
        
        # Draw network
        # Draw edges with varying thickness
        edges = G.edges()
        weights = [G[u][v]['weight'] * 10 for u, v in edges]  # Scale for visibility
        
        nx.draw_networkx_edges(G, pos, width=weights, alpha=0.7, edge_color='red')
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color='lightblue', 
                              edgecolors='blue', linewidths=2)
        
        # Draw labels
        labels = {i: f'S{i+1}' for i in range(len(sites_cart))}
        nx.draw_networkx_labels(G, pos, labels, font_size=12, font_weight='bold')
        
        # Draw edge labels if requested
        if show_edge_labels:
            edge_labels = {(u, v): f"{G[u][v]['jumps']}" for u, v in G.edges()}
            nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=10)
        
        plt.title('Site Connectivity Network Graph\n(Edge thickness ∝ Jump Frequency)', 
                 fontsize=14)
        plt.axis('off')
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Network graph saved: {output_file}")
        
        plt.show()
    
    def _build_connectivity_matrix(self, sites_cart, jump_data):
        """Build connectivity matrix from jump data."""
        n_sites = len(sites_cart)
        connectivity_matrix = np.zeros((n_sites, n_sites), dtype=int)
        
        if jump_data.size == 0:
            return connectivity_matrix
        
        for jump in jump_data:
            from_site = int(jump[1]) - 1  # Convert from 1-based to 0-based
            to_site = int(jump[2]) - 1
            
            if 0 <= from_site < n_sites and 0 <= to_site < n_sites:
                connectivity_matrix[from_site, to_site] += 1
        
        return connectivity_matrix
    
    def plot_all_connectivity_analyses(self, sites_cart, jump_data, min_jumps=1,
                                     output_dir=None):
        """
        Generate all connectivity plots in one function call.
        
        Args:
            sites_cart (np.array): Site coordinates
            jump_data (np.array): Jump data
            min_jumps (int): Minimum jumps to show connections
            output_dir (str, optional): Directory to save all plots
        """
        print("🔄 Generating site connectivity plots...")
        
        # Build connectivity matrix once
        connectivity_matrix = self._build_connectivity_matrix(sites_cart, jump_data)
        
        # 3D connectivity plot
        self.plot_3d_site_connectivity(
            sites_cart, jump_data, connectivity_matrix=connectivity_matrix,
            min_jumps=min_jumps,
            output_file=f"{output_dir}/3d_site_connectivity.png" if output_dir else None
        )
        
        # 2D projections
        for projection in ['xy', 'xz', 'yz']:
            self.plot_2d_site_connectivity(
                sites_cart, jump_data, projection=projection,
                connectivity_matrix=connectivity_matrix, min_jumps=min_jumps,
                output_file=f"{output_dir}/2d_connectivity_{projection}.png" if output_dir else None
            )
        
        # Network graph
        self.plot_connectivity_network_graph(
            sites_cart, jump_data, connectivity_matrix=connectivity_matrix,
            min_jumps=min_jumps,
            output_file=f"{output_dir}/network_graph.png" if output_dir else None
        )
        
        print("✅ All connectivity plots generated!")
        
        # Print connectivity statistics
        total_connections = np.sum(connectivity_matrix > 0)
        total_jumps = np.sum(connectivity_matrix)
        
        print(f"\n📊 CONNECTIVITY STATISTICS:")
        print(f"   Total Site Pairs with Jumps:       {total_connections}")
        print(f"   Total Jumps Recorded:              {total_jumps}")
        print(f"   Average Jumps per Connection:      {total_jumps/total_connections if total_connections > 0 else 0:.2f}")
        print(f"   Sites with Outgoing Jumps:         {np.sum(np.sum(connectivity_matrix, axis=1) > 0)}")
        print(f"   Sites with Incoming Jumps:         {np.sum(np.sum(connectivity_matrix, axis=0) > 0)}")

