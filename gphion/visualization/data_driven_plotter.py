"""Data-driven plotting utilities that use saved analysis results directly."""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from pathlib import Path

class DataDrivenPlotter:
    """
    A plotter class that reads saved analysis data files directly.
    
    This ensures we use the exact time values and data as calculated
    by the analysis modules without any conversion issues.
    """
    
    def __init__(self, data_dir):
        """
        Initialize the plotter with data directory.
        
        Args:
            data_dir (str): Path to directory containing saved analysis data
        """
        self.data_dir = Path(data_dir)
        self.default_figure_size = (12, 8)
        
    def plot_jump_histogram_from_saved_data(self, output_file=None, bin_width_ps=500):
        """
        Plot jump histogram using saved jump data.
        
        Args:
            output_file (str, optional): Path to save the plot
            bin_width_ps (float): Bin width in picoseconds
        """
        # Read the detailed jump data
        jump_file = self.data_dir / "detailed_jump_data.dat"
        if not jump_file.exists():
            print(f"Error: {jump_file} not found")
            return
            
        # Read the data, skipping header lines
        data = []
        with open(jump_file, 'r') as f:
            for line in f:
                if line.startswith('#') or line.strip() == '':
                    continue
                parts = line.strip().split()
                if len(parts) >= 4:
                    time = float(parts[3])  # Time column
                    data.append(time)
        
        if not data:
            print("No jump data found")
            return
            
        jump_times = np.array(data)
        
        plt.figure(figsize=self.default_figure_size)
        
        # Create histogram bins
        min_time = np.min(jump_times)
        max_time = np.max(jump_times)
        bins = np.arange(min_time, max_time + bin_width_ps, bin_width_ps)
        
        plt.hist(jump_times, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
        
        plt.title('Jump Frequency Distribution vs Time')
        plt.xlabel('Time (ps)')
        plt.ylabel('Number of jumps')
        plt.grid(True, alpha=0.3)
        
        # Add statistics
        total_jumps = len(jump_times)
        time_span = max_time - min_time
        avg_rate = total_jumps / (time_span * 1e-12)  # Convert ps to s for rate
        
        stats_text = f'Total jumps: {total_jumps}\\nTime span: {time_span:.1f} ps\\nAverage rate: {avg_rate:.2e} jumps/s'
        plt.text(0.98, 0.98, stats_text, transform=plt.gca().transAxes, 
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        print(f"✅ Jump histogram plotted using saved data")
        print(f"   Time range: {min_time:.1f} - {max_time:.1f} ps")
        print(f"   Total jumps: {total_jumps}")
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"   Plot saved: {output_file}")
        
        plt.show()
    
    def plot_jump_distances_vs_time(self, output_file=None):
        """
        Plot jump distances vs time using saved temporal data.
        
        Args:
            output_file (str, optional): Path to save the plot
        """
        # Read the jump distances vs time data
        temporal_file = self.data_dir / "jump_distances_vs_time.dat"
        if not temporal_file.exists():
            print(f"Error: {temporal_file} not found")
            return
            
        # Read the data
        times = []
        distances = []
        with open(temporal_file, 'r') as f:
            for line in f:
                if line.startswith('#') or line.strip() == '':
                    continue
                parts = line.strip().split()
                if len(parts) >= 2:
                    times.append(float(parts[0]))
                    distances.append(float(parts[1]))
        
        if not times:
            print("No temporal data found")
            return
            
        plt.figure(figsize=self.default_figure_size)
        plt.scatter(times, distances, alpha=0.6, s=20)
        
        plt.title('Jump Distances vs Time')
        plt.xlabel('Time (ps)')
        plt.ylabel('Jump Distance (Å)')
        plt.grid(True, alpha=0.3)
        
        # Add statistics
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        
        stats_text = f'Mean distance: {mean_dist:.3f} Å\\nStd distance: {std_dist:.3f} Å\\nTotal jumps: {len(times)}'
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        print(f"✅ Jump distances vs time plotted")
        print(f"   Time range: {np.min(times):.1f} - {np.max(times):.1f} ps")
        print(f"   Distance range: {np.min(distances):.3f} - {np.max(distances):.3f} Å")
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"   Plot saved: {output_file}")
        
        plt.show()
    
    def plot_jump_distance_histogram(self, output_file=None):
        """
        Plot jump distance histogram using saved histogram data.
        
        Args:
            output_file (str, optional): Path to save the plot
        """
        # Read histogram data
        hist_file = self.data_dir / "jump_distance_histogram.dat"
        
        if not hist_file.exists():
            print(f"Error: {hist_file} not found")
            return
            
        # Read histogram data (two columns: distance_center, count)
        bin_centers = []
        hist_counts = []
        
        with open(hist_file, 'r') as f:
            for line in f:
                if line.startswith('#') or line.strip() == '':
                    continue
                parts = line.strip().split()
                if len(parts) >= 2:
                    bin_centers.append(float(parts[0]))
                    hist_counts.append(float(parts[1]))
        
        if not bin_centers:
            print("No histogram data found")
            return
            
        bin_centers = np.array(bin_centers)
        hist_counts = np.array(hist_counts)
        
        plt.figure(figsize=self.default_figure_size)
        
        # Calculate bin width from the spacing
        if len(bin_centers) > 1:
            bin_width = bin_centers[1] - bin_centers[0]
        else:
            bin_width = 0.1
        
        # Plot histogram
        plt.bar(bin_centers, hist_counts, width=bin_width * 0.8, alpha=0.7, 
               color='lightcoral', edgecolor='black')
        
        plt.title('Jump Distance Distribution')
        plt.xlabel('Jump Distance (Å)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # Add statistics
        total_jumps = np.sum(hist_counts)
        if total_jumps > 0:
            weighted_mean = np.sum(bin_centers * hist_counts) / total_jumps
        else:
            weighted_mean = 0
        
        stats_text = f'Total jumps: {int(total_jumps)}\\nWeighted mean: {weighted_mean:.3f} Å\\nBins: {len(hist_counts)}'
        plt.text(0.98, 0.98, stats_text, transform=plt.gca().transAxes, 
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        print(f"✅ Jump distance histogram plotted")
        print(f"   Distance range: {bin_centers[0]:.3f} - {bin_centers[-1]:.3f} Å")
        print(f"   Total jumps: {int(total_jumps)}")
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"   Plot saved: {output_file}")
        
        plt.show()
    
    def read_statistics(self):
        """Read and display saved statistics."""
        stats_file = self.data_dir / "jump_distance_statistics.dat"
        if not stats_file.exists():
            print(f"Error: {stats_file} not found")
            return None
            
        stats = {}
        with open(stats_file, 'r') as f:
            for line in f:
                if '=' in line:
                    key, value = line.strip().split('=', 1)
                    try:
                        stats[key.strip()] = float(value.strip())
                    except ValueError:
                        stats[key.strip()] = value.strip()
        
        return stats
    
    def create_summary_plot(self, output_file=None):
        """
        Create a comprehensive summary plot with multiple subplots.
        
        Args:
            output_file (str, optional): Path to save the plot
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Jump histogram vs time
        jump_file = self.data_dir / "detailed_jump_data.dat"
        if jump_file.exists():
            data = []
            with open(jump_file, 'r') as f:
                for line in f:
                    if line.startswith('#') or line.strip() == '':
                        continue
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        time = float(parts[3])
                        data.append(time)
            
            if data:
                jump_times = np.array(data)
                ax1.hist(jump_times, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                ax1.set_xlabel('Time (ps)')
                ax1.set_ylabel('Number of jumps')
                ax1.set_title('Jump Frequency Distribution')
                ax1.grid(True, alpha=0.3)
        
        # 2. Jump distances vs time (scatter)
        temporal_file = self.data_dir / "jump_distances_vs_time.dat"
        if temporal_file.exists():
            times, distances = [], []
            with open(temporal_file, 'r') as f:
                for line in f:
                    if line.startswith('#') or line.strip() == '':
                        continue
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        times.append(float(parts[0]))
                        distances.append(float(parts[1]))
            
            if times:
                ax2.scatter(times, distances, alpha=0.6, s=10)
                ax2.set_xlabel('Time (ps)')
                ax2.set_ylabel('Jump Distance (Å)')
                ax2.set_title('Jump Distances vs Time')
                ax2.grid(True, alpha=0.3)
        
        # 3. Distance histogram
        hist_file = self.data_dir / "jump_distance_histogram.dat"
        if hist_file.exists():
            # Read histogram data (two columns: distance_center, count)
            bin_centers = []
            hist_counts = []
            
            with open(hist_file, 'r') as f:
                for line in f:
                    if line.startswith('#') or line.strip() == '':
                        continue
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        bin_centers.append(float(parts[0]))
                        hist_counts.append(float(parts[1]))
            
            if bin_centers:
                bin_centers = np.array(bin_centers)
                hist_counts = np.array(hist_counts)
                
                # Calculate bin width
                if len(bin_centers) > 1:
                    bin_width = bin_centers[1] - bin_centers[0]
                else:
                    bin_width = 0.1
                
                ax3.bar(bin_centers, hist_counts, width=bin_width * 0.8, 
                       alpha=0.7, color='lightcoral', edgecolor='black')
                ax3.set_xlabel('Jump Distance (Å)')
                ax3.set_ylabel('Frequency')
                ax3.set_title('Distance Distribution')
                ax3.grid(True, alpha=0.3)
        
        # 4. Summary statistics
        stats = self.read_statistics()
        if stats:
            stats_text = ""
            for key, value in stats.items():
                if isinstance(value, (int, float)):
                    if value > 1000 or (value < 0.01 and value != 0):
                        stats_text += f"{key}: {value:.2e}\\n"
                    else:
                        stats_text += f"{key}: {value:.3f}\\n"
                else:
                    stats_text += f"{key}: {value}\\n"
            
            ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=10,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            ax4.set_xlim(0, 1)
            ax4.set_ylim(0, 1)
            ax4.axis('off')
            ax4.set_title('Analysis Statistics')
        
        plt.suptitle('Jump Analysis Summary (Data-Driven)', fontsize=16)
        plt.tight_layout()
        
        print("✅ Summary plot created using saved data files")
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"   Plot saved: {output_file}")
        
        plt.show()
    
    def plot_all_analyses(self, output_dir=None):
        """
        Generate all available plots from saved data.
        
        Args:
            output_dir (str, optional): Directory to save plots
        """
        print("🔄 Generating plots from saved analysis data...")
        
        output_path = Path(output_dir) if output_dir else None
        
        # Jump histogram
        self.plot_jump_histogram_from_saved_data(
            output_file=output_path / "data_driven_jump_histogram.png" if output_path else None
        )
        
        # Jump distances vs time
        self.plot_jump_distances_vs_time(
            output_file=output_path / "data_driven_jump_distances_vs_time.png" if output_path else None
        )
        
        # Distance histogram
        self.plot_jump_distance_histogram(
            output_file=output_path / "data_driven_distance_histogram.png" if output_path else None
        )
        
        # Summary plot
        self.create_summary_plot(
            output_file=output_path / "data_driven_summary.png" if output_path else None
        )
        
        print("✅ All data-driven plots generated!")
