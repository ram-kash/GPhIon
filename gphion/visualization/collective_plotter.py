"""Collective jump behavior plotting utilities for TAOG analysis."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from ..core.plot_io import save_plot_data

class CollectivePlotter:
    """
    A dedicated class for plotting collective jump behavior analysis results.
    
    This class provides visualizations for collective jump analysis including:
    - Jump frequency histograms over time
    - Collective jump correlation matrices
    - Jump type combination analysis
    
    Data is robustly saved via plot_io for simple reloading and replotting.
    """
    
    def __init__(self, data_dir=None):
        """Initialize the Collective plotter."""
        self.default_figure_size = (12, 8)
        self.data_dir = Path(data_dir) if data_dir else None
        
    def plot_jump_histogram_vs_time(self, jump_data, timestep=None, bin_width=500, save_payload_dir=None):
        """Plot histogram of number of jumps vs. timestep."""
        if jump_data.size == 0:
            print("Warning: No jump data available for histogram plotting.")
            return
            
        jump_times = jump_data[:, 3]
        
        plt.figure(figsize=self.default_figure_size)
        
        min_time = np.min(jump_times)
        max_time = np.max(jump_times)
        
        # Use simple fixed number of bins like the old summary plot
        counts, bin_edges, _ = plt.hist(jump_times, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        
        plt.title('Jump Frequency Distribution')
        plt.xlabel('Time (ps)')
        plt.ylabel('Number of jumps')
        plt.grid(True, alpha=0.3)
        
        total_jumps = len(jump_times)
        time_span = max_time - min_time
        avg_rate = total_jumps / (time_span * 1e-12)
        
        stats_text = f'Total jumps: {total_jumps}\nTime span: {time_span:.1f} ps\nAverage rate: {avg_rate:.2e} jumps/s'
        plt.text(0.98, 0.98, stats_text, transform=plt.gca().transAxes, 
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        if save_payload_dir:
            out_file = Path(save_payload_dir) / "jump_histogram_vs_time.png"
            plt.savefig(out_file, dpi=300, bbox_inches='tight')
        plt.show()

        if save_payload_dir:
            save_plot_data(
                name="collective_jump_histogram",
                payload={"jump_times": jump_times, "counts": counts, "bins": np.array(bin_edges)},
                base_dir=save_payload_dir
            )
    
    def plot_collective_jump_matrix(self, correlation_matrix, jump_types, save_payload_dir=None):
        """Plot correlation matrix as an annotated seaborn heatmap."""
        if correlation_matrix.size == 0:
            print("Warning: No correlation matrix data available for plotting.")
            return
            
        plt.figure(figsize=(max(10, len(jump_types) * 0.8), max(8, len(jump_types) * 0.6)))
        
        sns.heatmap(correlation_matrix, 
                   xticklabels=jump_types, 
                   yticklabels=jump_types,
                   annot=True, 
                   fmt='d', 
                   cmap='Blues',
                   square=True,
                   cbar_kws={'label': 'Number of cooperative jumps'})
        
        plt.title('Cooperative Jump Correlation Matrix')
        plt.xlabel('Jump Type')
        plt.ylabel('Jump Type')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        if save_payload_dir:
            out_file = Path(save_payload_dir) / "collective_jump_matrix.png"
            plt.savefig(out_file, dpi=300, bbox_inches='tight')
        plt.show()

        if save_payload_dir:
            save_plot_data(
                name="collective_jump_matrix",
                payload={"correlation_matrix": correlation_matrix, "jump_types": jump_types},
                base_dir=save_payload_dir
            )
            
    def plot_collective_ratios(self, collective_results, total_jumps, save_payload_dir=None):
        """Plot pie chart comparing Collective to Non-collective jump ratios, and display summary stats."""
        collective_pairs = len(collective_results.get('collective_pairs', []))
        collective_jumps = collective_pairs * 2
        non_collective_jumps = collective_results.get('uncollective_count', total_jumps - collective_jumps)
        multi_jumps = collective_results.get('multi_collective_jumps', [])
        
        fig, (ax_pie, ax_text) = plt.subplots(1, 2, figsize=(14, 7))
        
        # 1. Pie Chart
        labels = ['Collective', 'Non-collective']
        sizes = [collective_jumps, non_collective_jumps]
        colors = ['lightcoral', 'lightblue']
        
        ax_pie.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax_pie.set_title('Collective vs Non-collective Jumps')
        
        # 2. Summary Statistics Text Box
        stats_text = f"""
        Total Jumps: {total_jumps}
        Collective Pairs: {collective_pairs}
        Collective Jumps: {collective_jumps}
        Non-collective Jumps: {non_collective_jumps}
        Multi-correlated Jumps: {len(multi_jumps)}
        Collective Fraction: {collective_jumps/total_jumps*100:.1f}%
        """
        
        ax_text.text(0.1, 0.9, stats_text, transform=ax_text.transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        ax_text.axis('off')
        ax_text.set_title('Summary Statistics')
        
        plt.tight_layout()
        if save_payload_dir:
            out_file = Path(save_payload_dir) / "collective_summary.png"
            plt.savefig(out_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        if save_payload_dir:
            save_plot_data(
                name="collective_jump_ratios",
                payload={
                    "collective_count": int(collective_jumps), 
                    "non_collective_count": int(non_collective_jumps),
                    "multi_correlated_count": int(len(multi_jumps)),
                    "total_jumps": int(total_jumps)
                },
                base_dir=save_payload_dir
            )

    def plot_multicorrelation_stats(self, collective_results, save_payload_dir=None):
        """Plot bar chart matching jumps to multiple correlation incidents."""
        multi_jumps = collective_results.get('multi_collective_jumps', [])
        correlation_counts = {}
        
        for jump_idx in multi_jumps:
            count = sum(1 for pair in collective_results.get('collective_pairs', []) 
                       if jump_idx in pair)
            correlation_counts[count] = correlation_counts.get(count, 0) + 1
        
        if not correlation_counts:
            print("No multi-correlated jumps available to plot.")
            return

        counts = list(correlation_counts.keys())
        frequencies = list(correlation_counts.values())
        
        plt.figure(figsize=self.default_figure_size)
        plt.bar(counts, frequencies, alpha=0.7, color='orange')
        plt.xlabel('Number of correlations per jump')
        plt.ylabel('Number of jumps')
        plt.title('Multi-correlation Statistics')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_payload_dir:
            out_file = Path(save_payload_dir) / "multi_correlation_stats.png"
            plt.savefig(out_file, dpi=300, bbox_inches='tight')
        plt.show()

        if save_payload_dir:
            save_plot_data(
                name="collective_multicorr_stats",
                payload={"counts": np.array(counts), "frequencies": np.array(frequencies)},
                base_dir=save_payload_dir
            )

    def plot_all_collective_analyses(self, collective_results, jump_data, timestep=None, output_dir="plot_data"):
        """
        Generate all collective jump plots independently and save arrays.

        Args:
            collective_results (dict): Results from collective jump analysis
            jump_data (np.array): All jump data
            timestep (float, optional): Timestep for time conversion
            output_dir (str, optional): Directory to save all payloads via plot_io
        """
        print("🔄 Generating isolated collective jump plots and packaging NPZ artifacts...")

        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            output_str = str(output_path)
        else:
            output_str = None

        # 1. Plot Histogram
        self.plot_jump_histogram_vs_time(jump_data, timestep=timestep, save_payload_dir=output_str)

        # 2. Plot Memory Matrix
        if 'correlation_matrix' in collective_results and 'jump_types' in collective_results:
            correlation_matrix = np.array(collective_results['correlation_matrix'])
            jump_types = collective_results['jump_types']
            self.plot_collective_jump_matrix(correlation_matrix, jump_types, save_payload_dir=output_str)

        # 3. Plot Pie Ratios
        total_jumps = len(jump_data)
        self.plot_collective_ratios(collective_results, total_jumps, save_payload_dir=output_str)

        # 4. Plot Multi-correlation Stats
        self.plot_multicorrelation_stats(collective_results, save_payload_dir=output_str)

        print(f"✅ All collective jump plots generated uniquely. Data correctly packaged at `{output_str}`!")
