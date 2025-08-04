"""Migration pathway visualization with probability-weighted connectivity."""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import networkx as nx
from scipy.spatial.distance import pdist, squareform
import warnings

class MigrationPathwayPlotter:
    """
    A specialized class for plotting migration pathways with probability-weighted connectivity.
    
    This class visualizes ion diffusion pathways by:
    - Showing hopping sites as nodes
    - Connecting sites with edges whose thickness/color represents migration probability
    - Supporting both 2D and 3D pathway visualization
    - Calculating migration probabilities from jump frequency data
    - Creating isoprobability surfaces for pathway analysis
    """
    
    def __init__(self):
        """Initialize the Migration Pathway plotter."""
        self.default_figure_size = (14, 10)
        
    def calculate_migration_probabilities(self, jump_data, sites_cart, 
                                        normalization='outgoing', symmetrize=False):
        """
        Calculate migration probability matrix from jump data.
        
        Args:
            jump_data (np.array): Jump data with [atom_id, from_site, to_site, time]
            sites_cart (np.array): Site coordinates (N x 3)
            normalization (str): 'outgoing', 'total', or 'raw'
            symmetrize (bool): Whether to make probability matrix symmetric
            
        Returns:
            dict: Migration analysis results
        """
        n_sites = len(sites_cart)
        migration_counts = np.zeros((n_sites, n_sites), dtype=int)
        
        if jump_data.size == 0:
            print("Warning: No jump data available for migration probability calculation")
            return {'probability_matrix': migration_counts, 'jump_counts': migration_counts}
        
        # Count jumps between sites
        for jump in jump_data:
            from_site = int(jump[1]) - 1  # Convert from 1-based to 0-based
            to_site = int(jump[2]) - 1
            
            if 0 <= from_site < n_sites and 0 <= to_site < n_sites:
                migration_counts[from_site, to_site] += 1
        
        # Calculate probability matrix based on normalization method
        if normalization == 'outgoing':
            # Normalize by outgoing jumps from each site
            row_sums = migration_counts.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1  # Avoid division by zero
            probability_matrix = migration_counts / row_sums
            
        elif normalization == 'total':
            # Normalize by total number of jumps
            total_jumps = migration_counts.sum()
            probability_matrix = migration_counts / (total_jumps if total_jumps > 0 else 1)
            
        elif normalization == 'raw':
            # Use raw counts as probabilities
            probability_matrix = migration_counts.astype(float)
        else:
            raise ValueError(f"Unknown normalization method: {normalization}")
        
        # Symmetrize if requested
        if symmetrize:
            probability_matrix = (probability_matrix + probability_matrix.T) / 2
        
        # Calculate additional statistics
        total_jumps = migration_counts.sum()
        active_sites = np.sum(migration_counts.sum(axis=1) > 0)
        max_probability = np.max(probability_matrix)
        
        results = {
            'probability_matrix': probability_matrix,
            'jump_counts': migration_counts,
            'total_jumps': total_jumps,
            'active_sites': active_sites,
            'max_probability': max_probability,
            'normalization': normalization
        }
        
        print(f"Migration Probability Analysis:")
        print(f"  Total jumps: {total_jumps}")
        print(f"  Active sites: {active_sites}/{n_sites}")
        print(f"  Max probability: {max_probability:.4f}")
        print(f"  Normalization: {normalization}")
        
        return results
    
    def plot_3d_migration_pathways(self, sites_cart, probability_matrix, 
                                  min_probability=0.01, edge_width_scale=10,
                                  node_size=100, colormap='plasma', 
                                  show_site_labels=True, transparency=0.7,
                                  output_file=None):
        """
        Plot 3D migration pathways with probability-weighted edges.
        
        Args:
            sites_cart (np.array): Site coordinates
            probability_matrix (np.array): Migration probability matrix
            min_probability (float): Minimum probability to show edge
            edge_width_scale (float): Scaling factor for edge width
            node_size (int): Size of site nodes
            colormap (str): Colormap for edge colors
            show_site_labels (bool): Whether to show site labels
            transparency (float): Edge transparency
            output_file (str, optional): Path to save plot
        """
        fig = plt.figure(figsize=self.default_figure_size)
        ax = fig.add_subplot(111, projection='3d')
        
        n_sites = len(sites_cart)
        
        # Plot site nodes
        ax.scatter(sites_cart[:, 0], sites_cart[:, 1], sites_cart[:, 2], 
                  s=node_size, c='blue', alpha=0.9, marker='o', 
                  edgecolors='darkblue', linewidth=2, label='Hopping Sites')
        
        # Add site labels
        if show_site_labels:
            for i, pos in enumerate(sites_cart):
                ax.text(pos[0], pos[1], pos[2], f'S{i+1}', 
                       fontsize=9, fontweight='bold', color='darkblue')
        
        # Prepare edge data
        edges = []
        edge_weights = []
        edge_colors = []
        
        max_prob = np.max(probability_matrix[probability_matrix > 0]) if np.any(probability_matrix > 0) else 1
        
        for i in range(n_sites):
            for j in range(n_sites):
                if i != j and probability_matrix[i, j] >= min_probability:
                    # Create directed edge from i to j
                    start_pos = sites_cart[i]
                    end_pos = sites_cart[j]
                    edges.append([start_pos, end_pos])
                    
                    # Edge properties based on migration probability
                    prob = probability_matrix[i, j]
                    normalized_prob = prob / max_prob
                    edge_weights.append(normalized_prob * edge_width_scale)
                    edge_colors.append(normalized_prob)
        
        # Plot edges if any exist
        if edges:
            line_collection = Line3DCollection(edges, linewidths=edge_weights, 
                                             cmap=colormap, alpha=transparency)
            line_collection.set_array(np.array(edge_colors))
            ax.add_collection3d(line_collection)
            
            # Add colorbar
            cbar = plt.colorbar(line_collection, ax=ax, shrink=0.6, pad=0.1)
            cbar.set_label('Relative Migration Probability', fontsize=12)
            
            print(f"Plotted {len(edges)} migration pathways")
        else:
            print("No migration pathways above threshold to plot")
        
        # Customize plot
        ax.set_xlabel('X (Å)', fontsize=12)
        ax.set_ylabel('Y (Å)', fontsize=12)
        ax.set_zlabel('Z (Å)', fontsize=12)
        ax.set_title('3D Migration Pathways\n(Edge width ∝ Migration Probability)', fontsize=14)
        
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
        ax.grid(True, alpha=0.3)
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"3D migration pathway plot saved: {output_file}")
        
        plt.show()
    
    def plot_migration_network_analysis(self, sites_cart, probability_matrix,
                                       min_probability=0.01, layout='spring',
                                       node_size=500, show_probabilities=True,
                                       output_file=None):
        """
        Create network analysis visualization of migration pathways.
        
        Args:
            sites_cart (np.array): Site coordinates
            probability_matrix (np.array): Migration probability matrix
            min_probability (float): Minimum probability for edge
            layout (str): Network layout algorithm
            node_size (int): Size of nodes
            show_probabilities (bool): Whether to show edge labels
            output_file (str, optional): Path to save plot
        """
        # Create NetworkX directed graph
        G = nx.DiGraph()
        
        n_sites = len(sites_cart)
        
        # Add nodes
        for i in range(n_sites):
            G.add_node(i, pos=sites_cart[i], label=f'S{i+1}')
        
        # Add edges with probabilities as weights
        max_prob = np.max(probability_matrix[probability_matrix > 0]) if np.any(probability_matrix > 0) else 1
        
        for i in range(n_sites):
            for j in range(n_sites):
                prob = probability_matrix[i, j]
                if i != j and prob >= min_probability:
                    G.add_edge(i, j, weight=prob, normalized_weight=prob/max_prob)
        
        plt.figure(figsize=self.default_figure_size)
        
        # Choose layout
        if layout == 'spring':
            pos = nx.spring_layout(G, k=2, iterations=50)
        elif layout == 'circular':
            pos = nx.circular_layout(G)
        elif layout == '3d_projection':
            # Project 3D coordinates to 2D
            pos = {i: sites_cart[i][:2] for i in range(n_sites)}
        else:
            pos = nx.spring_layout(G)
        
        # Draw edges with varying thickness
        edges = G.edges()
        weights = [G[u][v]['normalized_weight'] * 10 for u, v in edges]
        
        nx.draw_networkx_edges(G, pos, width=weights, alpha=0.7, 
                              edge_color='red', arrows=True, arrowsize=20)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=node_size, 
                              node_color='lightblue', edgecolors='blue', 
                              linewidths=2)
        
        # Draw labels
        labels = {i: f'S{i+1}' for i in range(n_sites)}
        nx.draw_networkx_labels(G, pos, labels, font_size=12, font_weight='bold')
        
        # Draw edge labels if requested
        if show_probabilities and len(edges) < 20:  # Avoid cluttering
            edge_labels = {(u, v): f"{G[u][v]['weight']:.3f}" for u, v in G.edges()}
            nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8)
        
        plt.title('Migration Network Analysis\n(Directed edges with probability weights)', 
                 fontsize=14)
        plt.axis('off')
        
        # Add network statistics
        stats_text = f"""Network Statistics:
Nodes: {G.number_of_nodes()}
Edges: {G.number_of_edges()}
Avg. degree: {sum(dict(G.degree()).values())/G.number_of_nodes():.2f}
"""
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Network analysis plot saved: {output_file}")
        
        plt.show()
    
    def analyze_migration_pathways(self, sites_cart, probability_matrix, 
                                  min_probability=0.01):
        """
        Analyze migration pathway connectivity and identify key routes.
        
        Args:
            sites_cart (np.array): Site coordinates
            probability_matrix (np.array): Migration probability matrix
            min_probability (float): Minimum probability threshold
            
        Returns:
            dict: Pathway analysis results
        """
        n_sites = len(sites_cart)
        
        # Create graph for analysis
        G = nx.DiGraph()
        for i in range(n_sites):
            for j in range(n_sites):
                prob = probability_matrix[i, j]
                if i != j and prob >= min_probability:
                    G.add_edge(i, j, weight=prob)
        
        # Calculate network metrics
        try:
            # Centrality measures
            in_degree_centrality = nx.in_degree_centrality(G)
            out_degree_centrality = nx.out_degree_centrality(G)
            betweenness_centrality = nx.betweenness_centrality(G)
            
            # Most important nodes
            most_central_in = max(in_degree_centrality.items(), key=lambda x: x[1])
            most_central_out = max(out_degree_centrality.items(), key=lambda x: x[1])
            most_between = max(betweenness_centrality.items(), key=lambda x: x[1])
            
        except:
            print("Warning: Could not calculate centrality measures")
            in_degree_centrality = {}
            out_degree_centrality = {}
            betweenness_centrality = {}
            most_central_in = (0, 0)
            most_central_out = (0, 0)
            most_between = (0, 0)
        
        # Find highest probability pathways
        max_prob = np.max(probability_matrix)
        high_prob_edges = []
        
        for i in range(n_sites):
            for j in range(n_sites):
                prob = probability_matrix[i, j]
                if prob > max_prob * 0.5:  # Top 50% of probabilities
                    high_prob_edges.append((i, j, prob))
        
        high_prob_edges.sort(key=lambda x: x[2], reverse=True)
        
        results = {
            'network_graph': G,
            'in_degree_centrality': in_degree_centrality,
            'out_degree_centrality': out_degree_centrality,
            'betweenness_centrality': betweenness_centrality,
            'most_central_in': most_central_in,
            'most_central_out': most_central_out,
            'most_between': most_between,
            'high_probability_edges': high_prob_edges[:10],  # Top 10
            'total_edges': G.number_of_edges(),
            'total_nodes': G.number_of_nodes()
        }
        
        print(f"\nMigration Pathway Analysis:")
        print(f"  Active migration network: {G.number_of_nodes()} sites, {G.number_of_edges()} pathways")
        print(f"  Most central sink (incoming): Site {most_central_in[0]+1} ({most_central_in[1]:.3f})")
        print(f"  Most central source (outgoing): Site {most_central_out[0]+1} ({most_central_out[1]:.3f})")
        print(f"  Most important bridge: Site {most_between[0]+1} ({most_between[1]:.3f})")
        
        if high_prob_edges:
            print(f"  Top migration pathways:")
            for i, (from_site, to_site, prob) in enumerate(high_prob_edges[:5]):
                print(f"    {i+1}. Site {from_site+1} → Site {to_site+1}: {prob:.4f}")
        
        return results
    
    def plot_comprehensive_migration_analysis(self, sites_cart, jump_data,
                                            min_probability=0.01, output_dir=None):
        """
        Generate comprehensive migration pathway analysis with multiple visualizations.
        
        Args:
            sites_cart (np.array): Site coordinates
            jump_data (np.array): Jump data
            min_probability (float): Minimum probability threshold
            output_dir (str, optional): Directory to save plots
        """
        print("🔄 Generating comprehensive migration pathway analysis...")
        
        # 1. Calculate migration probabilities
        migration_results = self.calculate_migration_probabilities(
            jump_data, sites_cart, normalization='outgoing'
        )
        
        probability_matrix = migration_results['probability_matrix']
        
        # 2. 3D pathway visualization
        self.plot_3d_migration_pathways(
            sites_cart, probability_matrix, min_probability=min_probability,
            output_file=f"{output_dir}/3d_migration_pathways.png" if output_dir else None
        )
        
        # 3. Network analysis
        self.plot_migration_network_analysis(
            sites_cart, probability_matrix, min_probability=min_probability,
            output_file=f"{output_dir}/migration_network.png" if output_dir else None
        )
        
        # 4. Pathway analysis
        pathway_analysis = self.analyze_migration_pathways(
            sites_cart, probability_matrix, min_probability=min_probability
        )
        
        # 5. Generate analysis report
        if output_dir:
            self._generate_migration_report(
                migration_results, pathway_analysis, 
                f"{output_dir}/migration_analysis_report.txt"
            )
        
        print("✅ Comprehensive migration pathway analysis complete!")
        
        return {
            'migration_probabilities': migration_results,
            'pathway_analysis': pathway_analysis
        }
    
    def _generate_migration_report(self, migration_results, pathway_analysis, output_file):
        """Generate detailed migration analysis report."""
        with open(output_file, 'w') as f:
            f.write("MIGRATION PATHWAY ANALYSIS REPORT\n")
            f.write("="*50 + "\n\n")
            
            # Migration statistics
            f.write("MIGRATION STATISTICS:\n")
            f.write(f"  Total jumps: {migration_results['total_jumps']}\n")
            f.write(f"  Active sites: {migration_results['active_sites']}\n")
            f.write(f"  Max probability: {migration_results['max_probability']:.4f}\n")
            f.write(f"  Normalization method: {migration_results['normalization']}\n\n")
            
            # Network analysis
            f.write("NETWORK ANALYSIS:\n")
            f.write(f"  Total nodes: {pathway_analysis['total_nodes']}\n")
            f.write(f"  Total edges: {pathway_analysis['total_edges']}\n")
            
            most_in = pathway_analysis['most_central_in']
            most_out = pathway_analysis['most_central_out']
            most_between = pathway_analysis['most_between']
            
            f.write(f"  Most central sink: Site {most_in[0]+1} ({most_in[1]:.3f})\n")
            f.write(f"  Most central source: Site {most_out[0]+1} ({most_out[1]:.3f})\n")
            f.write(f"  Most important bridge: Site {most_between[0]+1} ({most_between[1]:.3f})\n\n")
            
            # High probability pathways
            f.write("TOP MIGRATION PATHWAYS:\n")
            for i, (from_site, to_site, prob) in enumerate(pathway_analysis['high_probability_edges']):
                f.write(f"  {i+1:2d}. Site {from_site+1:2d} → Site {to_site+1:2d}: {prob:.4f}\n")
        
        print(f"Migration analysis report saved: {output_file}")

    def calculate_time_weighted_migration_probabilities(self, jump_data, sites_cart, 
                                                   time_window=None, decay_factor=0.1):
        """
        Calculate migration probabilities with time-based weighting.
        
        Args:
            jump_data (np.array): Jump data with [atom_id, from_site, to_site, time]
            sites_cart (np.array): Site coordinates
            time_window (float, optional): Time window for correlation (ps)
            decay_factor (float): Exponential decay factor for time weighting
            
        Returns:
            dict: Enhanced migration analysis with time weighting
        """
        n_sites = len(sites_cart)
        
        # Time-weighted migration matrix
        time_weighted_counts = np.zeros((n_sites, n_sites))
        raw_counts = np.zeros((n_sites, n_sites), dtype=int)
        
        if jump_data.size == 0:
            print("Warning: No jump data available")
            return {'probability_matrix': raw_counts, 'time_weighted_matrix': time_weighted_counts}
        
        # Sort jumps by time
        sorted_jumps = jump_data[np.argsort(jump_data[:, 3])]
        total_time = np.max(sorted_jumps[:, 3]) - np.min(sorted_jumps[:, 3])
        
        for i, jump in enumerate(sorted_jumps):
            from_site = int(jump[1]) - 1  # Convert to 0-based
            to_site = int(jump[2]) - 1
            jump_time = jump[3]
            
            if 0 <= from_site < n_sites and 0 <= to_site < n_sites:
                raw_counts[from_site, to_site] += 1
                
                # Time-based weighting: recent jumps get higher weight
                if time_window is not None:
                    # Weight based on recency within time window
                    time_weight = np.exp(-decay_factor * (total_time - jump_time) / time_window)
                else:
                    # Weight based on position in simulation (later = higher weight)
                    time_weight = 1.0 + (jump_time / total_time)
                
                time_weighted_counts[from_site, to_site] += time_weight
        
        # Normalize probabilities
        # Method 1: Normalize by outgoing connections
        row_sums = time_weighted_counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        time_weighted_prob = time_weighted_counts / row_sums
        
        # Method 2: Also create frequency-based probability
        raw_row_sums = raw_counts.sum(axis=1, keepdims=True)
        raw_row_sums[raw_row_sums == 0] = 1  
        frequency_prob = raw_counts / raw_row_sums
        
        return {
            'probability_matrix': frequency_prob,
            'time_weighted_matrix': time_weighted_prob,
            'raw_counts': raw_counts,
            'time_weighted_counts': time_weighted_counts,
            'total_jumps': raw_counts.sum()
        }

    def plot_enhanced_3d_migration_pathways(self, sites_cart, jump_data, 
                                        min_probability=0.001, 
                                        use_time_weighting=True,
                                        connection_threshold=2,  # Minimum number of jumps
                                        edge_width_scale=10,
                                        output_file=None):
        """
        Enhanced 3D migration pathway plot with better connection visibility.
        """
        # Calculate both types of probabilities
        if use_time_weighting:
            migration_results = self.calculate_time_weighted_migration_probabilities(
                jump_data, sites_cart
            )
            probability_matrix = migration_results['time_weighted_matrix']
            raw_counts = migration_results['raw_counts']
        else:
            migration_results = self.calculate_migration_probabilities(
                jump_data, sites_cart, normalization='outgoing'
            )
            probability_matrix = migration_results['probability_matrix']
            raw_counts = migration_results['jump_counts']
        
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        n_sites = len(sites_cart)
        
        # Plot site nodes with size based on activity
        site_activity = raw_counts.sum(axis=0) + raw_counts.sum(axis=1)  # Total jumps per site
        max_activity = np.max(site_activity) if np.max(site_activity) > 0 else 1
        node_sizes = 50 + (site_activity / max_activity) * 200  # Scale node size by activity
        
        scatter = ax.scatter(sites_cart[:, 0], sites_cart[:, 1], sites_cart[:, 2], 
                            s=node_sizes, c=site_activity, cmap='viridis', 
                            alpha=0.8, edgecolors='darkblue', linewidth=2)
        
        # Add colorbar for node activity
        cbar_nodes = plt.colorbar(scatter, ax=ax, shrink=0.6, pad=0.02)
        cbar_nodes.set_label('Site Activity (Total Jumps)', fontsize=10)
        
        # Prepare edge data with multiple criteria
        edges = []
        edge_weights = []
        edge_colors = []
        
        max_prob = np.max(probability_matrix[probability_matrix > 0]) if np.any(probability_matrix > 0) else 1
        
        for i in range(n_sites):
            for j in range(n_sites):
                if i != j:
                    prob = probability_matrix[i, j]
                    jump_count = raw_counts[i, j]
                    
                    # Multiple criteria for showing connections:
                    # 1. Probability above threshold, OR
                    # 2. Minimum number of jumps, OR  
                    # 3. High probability connections regardless of threshold
                    show_connection = (
                        prob >= min_probability or 
                        jump_count >= connection_threshold or
                        prob > max_prob * 0.1  # Top 10% of probabilities
                    )
                    
                    if show_connection:
                        start_pos = sites_cart[i]
                        end_pos = sites_cart[j]
                        edges.append([start_pos, end_pos])
                        
                        # Edge width based on probability
                        normalized_prob = prob / max_prob
                        edge_weights.append(max(0.5, normalized_prob * edge_width_scale))
                        edge_colors.append(normalized_prob)
        
        # Plot edges
        if edges:
            from mpl_toolkits.mplot3d.art3d import Line3DCollection
            
            line_collection = Line3DCollection(edges, linewidths=edge_weights, 
                                            cmap='plasma', alpha=0.7)
            line_collection.set_array(np.array(edge_colors))
            ax.add_collection3d(line_collection)
            
            # Add colorbar for edge weights
            cbar_edges = plt.colorbar(line_collection, ax=ax, shrink=0.6, pad=0.15)
            cbar_edges.set_label('Migration Probability', fontsize=10)
            
            print(f"Plotted {len(edges)} migration pathways")
            
            # Add statistics text
            stats_text = f"""Migration Statistics:
    Total pathways: {len(edges)}
    Probability range: {np.min(edge_colors):.4f} - {np.max(edge_colors):.4f}
    Threshold used: {min_probability:.4f}
    """
            ax.text2D(0.02, 0.98, stats_text, transform=ax.transAxes,
                    verticalalignment='top', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        else:
            print("No migration pathways found with current criteria")
            print(f"Try lowering min_probability below {min_probability}")
            print(f"Or lowering connection_threshold below {connection_threshold}")
        
        # Enhanced plot formatting
        ax.set_xlabel('X (Å)', fontsize=12)
        ax.set_ylabel('Y (Å)', fontsize=12)
        ax.set_zlabel('Z (Å)', fontsize=12)
        ax.set_title('3D Migration Pathways with Time-Based Probability\n'
                    f'(Node size ∝ Activity, Edge thickness ∝ Probability)', fontsize=14)
        
        # Better axis limits
        margin = 2.0  # Angstrom margin
        ax.set_xlim(sites_cart[:, 0].min() - margin, sites_cart[:, 0].max() + margin)
        ax.set_ylim(sites_cart[:, 1].min() - margin, sites_cart[:, 1].max() + margin)
        ax.set_zlim(sites_cart[:, 2].min() - margin, sites_cart[:, 2].max() + margin)
        
        ax.grid(True, alpha=0.3)
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Enhanced 3D migration plot saved: {output_file}")
        
        plt.show()
