"""Collective jump behavior plotting utilities for TAOG analysis."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class CollectivePlotter:
    """
    A dedicated class for plotting collective jump behavior analysis results.
    
    This class provides visualizations for collective jump analysis including:
    - Jump frequency histograms over time
    - Collective jump correlation matrices
    - Jump type combination analysis
    """
    
    def __init__(self):
        """Initialize the Collective plotter."""
        self.default_figure_size = (12, 8)
        
    def plot_jump_histogram_vs_time(self, jump_data, timestep=None, bin_width=500, 
                                   output_file=None):
        """
        Plot histogram of number of jumps vs. timestep.
        
        Args:
            jump_data (np.array): Jump data array with time information in column 3 (0-indexed)
            timestep (float, optional): Timestep to convert frames to real time
            bin_width (int): Width of histogram bins in frame units
            output_file (str, optional): Path to save the plot
        """
        if jump_data.size == 0:
            print("Warning: No jump data available for histogram plotting.")
            return
            
        # Extract jump times (assuming column 3 contains time data)
        jump_times = jump_data[:, 3]
        
        # Create bin centers
        max_time = np.max(jump_times)
        bin_centers = np.arange(bin_width/2, max_time + bin_width, bin_width)
        bin_edges = np.arange(0, max_time + bin_width + 1, bin_width)
        
        plt.figure(figsize=self.default_figure_size)
        
        # Create histogram
        plt.hist(jump_times, bins=bin_edges, alpha=0.7, color='skyblue', edgecolor='black')
        
        # Labels and title
        if timestep is not None:
            # Convert to real time units
            time_ps = jump_times * timestep * 1e12  # Convert to ps
            xlabel = 'Time (ps)'
            plt.figure(figsize=self.default_figure_size)
            bin_edges_ps = bin_edges * timestep * 1e12
            plt.hist(time_ps, bins=bin_edges_ps, alpha=0.7, color='skyblue', edgecolor='black')
        else:
            xlabel = 'Time (steps)'
            
        plt.title('Histogram of Jumps vs. Time')
        plt.xlabel(xlabel)
        plt.ylabel('Number of jumps')
        plt.grid(True, alpha=0.3)
        
        # Add statistics
        total_jumps = len(jump_times)
        if timestep is not None:
            total_time = max_time * timestep
            avg_rate = total_jumps / total_time
            rate_unit = 'jumps/s'
        else:
            avg_rate = total_jumps / max_time
            rate_unit = 'jumps/step'
            
        stats_text = f'Total jumps: {total_jumps}\nAverage rate: {avg_rate:.2e} {rate_unit}'
        plt.text(0.98, 0.98, stats_text, transform=plt.gca().transAxes, 
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Plot saved: {output_file}")
        
        plt.show()
    
    def plot_collective_jump_matrix(self, correlation_matrix, jump_types, 
                                   colormap='viridis', output_file=None):
        """
        Plot correlation matrix showing collective jump type combinations.
        
        Args:
            correlation_matrix (np.array): Matrix of collective jump correlations
            jump_types (list): List of jump type names
            colormap (str): Colormap for the matrix visualization
            output_file (str, optional): Path to save the plot
        """
        if correlation_matrix.size == 0:
            print("Warning: No correlation matrix data available for plotting.")
            return
            
        plt.figure(figsize=(max(10, len(jump_types) * 0.8), max(8, len(jump_types) * 0.6)))
        
        # Create the matrix plot
        im = plt.imshow(correlation_matrix, cmap=colormap, aspect='auto', 
                       interpolation='nearest')
        plt.colorbar(im, label='Number of cooperative jumps')
        
        # Set ticks and labels
        plt.xticks(range(len(jump_types)), jump_types, rotation=90)
        plt.yticks(range(len(jump_types)), jump_types)
        
        # Add text annotations for non-zero values
        for i in range(len(jump_types)):
            for j in range(len(jump_types)):
                value = correlation_matrix[i, j]
                if value > 0:
                    # Choose text color based on background
                    text_color = 'white' if value > correlation_matrix.max() * 0.5 else 'black'
                    plt.text(j, i, f'{int(value)}', ha='center', va='center', 
                            color=text_color, fontweight='bold')
        
        plt.title('Number of Cooperative Jumps per Jump-Type Combination')
        plt.xlabel('Jump Type')
        plt.ylabel('Jump Type')
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Plot saved: {output_file}")
        
        plt.show()
    
    def plot_collective_jump_heatmap(self, correlation_matrix, jump_types, 
                                    annot=True, fmt='d', output_file=None):
        """
        Plot correlation matrix as a seaborn heatmap with better formatting.
        
        Args:
            correlation_matrix (np.array): Matrix of collective jump correlations
            jump_types (list): List of jump type names
            annot (bool): Whether to annotate cells with values
            fmt (str): Format for annotations
            output_file (str, optional): Path to save the plot
        """
        if correlation_matrix.size == 0:
            print("Warning: No correlation matrix data available for heatmap plotting.")
            return
            
        plt.figure(figsize=(max(10, len(jump_types) * 0.8), max(8, len(jump_types) * 0.6)))
        
        # Create heatmap
        sns.heatmap(correlation_matrix, 
                   xticklabels=jump_types, 
                   yticklabels=jump_types,
                   annot=annot, 
                   fmt=fmt, 
                   cmap='Blues',
                   square=True,
                   cbar_kws={'label': 'Number of cooperative jumps'})
        
        plt.title('Cooperative Jump Correlation Matrix')
        plt.xlabel('Jump Type')
        plt.ylabel('Jump Type')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Plot saved: {output_file}")
        
        plt.show()
    
    def plot_collective_summary(self, collective_results, jump_data, timestep=None, 
                               output_file=None):
        """
        Create a summary plot showing collective jump statistics.
        
        Args:
            collective_results (dict): Results from collective jump analysis
            jump_data (np.array): All jump data
            timestep (float, optional): Timestep for time conversion
            output_file (str, optional): Path to save the plot
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Jump histogram
        jump_times = jump_data[:, 3]
        if timestep is not None:
            jump_times_ps = jump_times * timestep * 1e12
            ax1.hist(jump_times_ps, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            ax1.set_xlabel('Time (ps)')
        else:
            ax1.hist(jump_times, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            ax1.set_xlabel('Time (steps)')
        ax1.set_ylabel('Number of jumps')
        ax1.set_title('Jump Frequency Distribution')
        ax1.grid(True, alpha=0.3)
        
        # 2. Collective vs non-collective pie chart
        collective_pairs = len(collective_results.get('collective_pairs', []))
        total_jumps = len(jump_data)
        collective_jumps = collective_pairs * 2  # Each pair involves 2 jumps
        non_collective_jumps = collective_results.get('uncollective_count', total_jumps - collective_jumps)
        
        labels = ['Collective', 'Non-collective']
        sizes = [collective_jumps, non_collective_jumps]
        colors = ['lightcoral', 'lightblue']
        
        ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Collective vs Non-collective Jumps')
        
        # 3. Multi-correlation statistics
        multi_jumps = collective_results.get('multi_collective_jumps', [])
        correlation_counts = {}
        
        # Count how many correlations each jump is involved in
        for jump_idx in multi_jumps:
            count = sum(1 for pair in collective_results.get('collective_pairs', []) 
                       if jump_idx in pair)
            correlation_counts[count] = correlation_counts.get(count, 0) + 1
        
        if correlation_counts:
            counts = list(correlation_counts.keys())
            frequencies = list(correlation_counts.values())
            ax3.bar(counts, frequencies, alpha=0.7, color='orange')
            ax3.set_xlabel('Number of correlations per jump')
            ax3.set_ylabel('Number of jumps')
            ax3.set_title('Multi-correlation Statistics')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'No multi-correlated jumps', ha='center', va='center', 
                    transform=ax3.transAxes, fontsize=12)
            ax3.set_title('Multi-correlation Statistics')
        
        # 4. Summary statistics
        stats_text = f"""
        Total Jumps: {total_jumps}
        Collective Pairs: {collective_pairs}
        Collective Jumps: {collective_jumps}
        Non-collective Jumps: {non_collective_jumps}
        Multi-correlated Jumps: {len(multi_jumps)}
        Collective Fraction: {collective_jumps/total_jumps*100:.1f}%
        """
        
        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        ax4.set_title('Summary Statistics')
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Plot saved: {output_file}")
        
        plt.show()
    
    def plot_all_collective_analyses(self, collective_results, jump_data, timestep=None,
                                   output_dir=None):
        """
        Generate all collective jump plots in one function call.
        
        Args:
            collective_results (dict): Results from collective jump analysis
            jump_data (np.array): All jump data
            timestep (float, optional): Timestep for time conversion
            output_dir (str, optional): Directory to save all plots
        """
        print("🔄 Generating collective jump analysis plots...")
        
        # Jump histogram
        self.plot_jump_histogram_vs_time(
            jump_data, timestep=timestep,
            output_file=f"{output_dir}/jump_histogram_vs_time.png" if output_dir else None
        )
        
        # Collective jump matrix
        if 'correlation_matrix' in collective_results and 'jump_types' in collective_results:
            self.plot_collective_jump_matrix(
                collective_results['correlation_matrix'],
                collective_results['jump_types'],
                output_file=f"{output_dir}/collective_jump_matrix.png" if output_dir else None
            )
            
            # Also create heatmap version
            self.plot_collective_jump_heatmap(
                collective_results['correlation_matrix'],
                collective_results['jump_types'],
                output_file=f"{output_dir}/collective_jump_heatmap.png" if output_dir else None
            )
        
        # Summary plot
        self.plot_collective_summary(
            collective_results, jump_data, timestep=timestep,
            output_file=f"{output_dir}/collective_summary.png" if output_dir else None
        )
        
        print("✅ All collective jump plots generated!")
    
    def set_plot_style(self, figure_size=None):
        """
        Customize the default plot styling.
        
        Args:
            figure_size (tuple, optional): Figure size as (width, height)
        """
        if figure_size is not None:
            self.default_figure_size = figure_size
            
        print(f"Plot style updated: figure_size={self.default_figure_size}")

